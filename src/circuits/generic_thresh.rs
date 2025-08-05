use crate::circuits::{
    arith::{linear, sum_circuit, sum_many_circuit, CircuitRing},
    builder::{new_builder, CircuitBuilder},
    circuit::TCircuit,
    elements::{compare_all, mask, CircuitElement},
    WireId,
};

pub fn init_circuit<T: CircuitRing>(
    num_parties: usize,
    threshold: usize,
    points: &[T],
) -> TCircuit<Vec<T>, Vec<T>> {
    let builder = new_builder();
    // each party inputs its share of alpha polynomial
    let (builder, fs_s) = builder.add_input_multi::<Vec<T>>(num_parties * threshold, None);

    // calculate sum of input fs
    let sum_fs_c = sum_many_circuit::<T>(threshold, num_parties);
    let (builder, fs) = builder.extend_circuit(&fs_s, &sum_fs_c, None);

    let const_poly_eval_c = T::const_poly_eval_circuit(threshold, points, true);
    let (builder, shares_alpha) = builder.extend_circuit(&fs, &const_poly_eval_c, None);

    builder
        .refine_input()
        .refine_output(&shares_alpha)
        .to_circuit()
}

fn input_sample_inner<T: CircuitRing, I, O>(
    builder: CircuitBuilder<I, O>,
    num_parties: usize,
    num_shares: usize,
    threshold: usize,
    points: &[T],
    alphas: &[WireId],
    xs: &[WireId],
    zs_s: &[WireId],
) -> TCircuit<I, Vec<Vec<(T, T)>>> {
    // sum alphas
    let sum_c = sum_circuit::<T>(num_parties);
    let (builder, alpha) = builder.extend_circuit(&alphas, &sum_c, Some("alpha"));

    // sum zs
    let sum_zs_c = sum_many_circuit::<T>(2 * num_shares * (threshold - 1), num_parties);
    let (builder, fs) = builder.extend_circuit(&zs_s, &sum_zs_c, None);
    // fs = f_1(.), f_2(.), ...
    // need to create [0_1]_i = f_1(i), [0_2]_i = f_2(i), ...
    let const_poly_eval_c = T::const_poly_eval_circuit(threshold, points, false);

    let (builder, zs) = builder.map_circuit(
        &fs,
        &const_poly_eval_c,
        2 * num_shares,
        |chunk| chunk.to_vec(),
        None,
    );

    // calculate ax_0, ax_1, ..
    let linear_c = linear::<T>(num_shares);
    let mut macs_in = alpha;
    macs_in.extend(xs);
    let (builder, macs) = builder.extend_circuit(&macs_in, &linear_c, None);

    // calculate party output shares
    let mut shares_in = Vec::with_capacity(T::BIT_SIZE * (2 * 2 * num_shares * num_parties));
    // want to make the input of the form x_0 * num_parties || m_0 * num_partys || x_1 * num_parties ...
    // to line up with the secret shares
    for (x_i, m_i) in xs
        .chunks_exact(T::BIT_SIZE)
        .zip(macs.chunks_exact(T::BIT_SIZE))
    {
        for _ in 0..num_parties {
            shares_in.extend(x_i);
        }
        for _ in 0..num_parties {
            shares_in.extend(m_i);
        }
    }

    shares_in.extend(zs);
    // sum the halves  (xs || macs)*num_parties + zs
    let sum_halves_c = sum_many_circuit::<T>(2 * num_shares * num_parties, 2);
    let (builder, outs) = builder.extend_circuit(&shares_in, &sum_halves_c, None);
    // need to shuffle the outputs so that that all party outputs are together
    // outs is currently [x_1] || [m_1] || [x_2] ...
    // want [x_1]_1 || [m_1]_1 || [x_2]_1 ... || [x_1]_n || ...
    let mut new_outs = Vec::with_capacity(outs.len());
    // each sharing is of `stride` size
    let stride = T::BIT_SIZE * num_parties;
    for i in 0..num_parties {
        // we are the `i`th party, so we are the `i`th element in each sharing
        let start = i * T::BIT_SIZE;
        for j in 0..num_shares {
            // each input has a value sharing and a mac sharing
            let xj = start + j * (2 * stride);
            let mj = xj + stride;
            // get [x_j]_i
            new_outs.extend(&outs[xj..xj + T::BIT_SIZE]);
            // get [m_j]_i
            new_outs.extend(&outs[mj..mj + T::BIT_SIZE]);
        }
    }

    let builder = builder.refine_output::<Vec<Vec<(T, T)>>>(&new_outs);

    builder.to_circuit()
}

/// The circuit used for producing authenticated shares in the `setup` instruction
/// of Fthresh using the generic threshold protocol
pub fn setup_circuit<T: CircuitRing, I, O: CircuitElement>(
    num_parties: usize,
    num_inputs: usize,
    threshold: usize,
    points: &[T],
    circuit: &TCircuit<I, O>,
) -> TCircuit<(I, Vec<(T, Vec<T>)>), Vec<Vec<(T, T)>>> {
    let builder = new_builder().with_consts();
    let consts = builder.get_const_wire_ids().unwrap();

    let (builder, xs) = builder.add_input_multi::<Vec<bool>>(num_parties * num_inputs, None);
    let mut builder = builder.refine_input::<(I, Vec<(T, Vec<T>)>)>();
    // each party inputs its share of alpha, 2*num_inputs*num_parties shares of zero
    let out_size = O::BIT_SIZE;
    let num_shares = (out_size + T::BIT_SIZE - 1) / T::BIT_SIZE;
    dbg!("setup_circuit", out_size, num_shares);
    let mut alphas = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut zs_s = Vec::with_capacity(T::BIT_SIZE * num_parties * (threshold - 1) * num_shares * 2);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, zs_i) = b2.add_input_multi::<Vec<T>>(2 * num_shares * (threshold - 1), None);
        // coalesce the input
        builder = b2.refine_input();

        alphas.extend(alpha_i);
        zs_s.extend(zs_i);
    }

    let (builder, mut ys) = builder.extend_circuit(&xs, circuit, None);
    ys.resize(T::BIT_SIZE * num_shares, consts[0]);

    input_sample_inner(
        builder,
        num_parties,
        num_shares,
        threshold,
        points,
        &alphas,
        &ys,
        &zs_s,
    )
}

/// The circuit used for producing authenticated shares in the `sample` instruction
/// of Fthresh using the generic threshold protocol
pub fn sample_circuit<T: CircuitRing>(
    num_parties: usize,
    num_inputs: usize,
    threshold: usize,
    points: &[T],
) -> TCircuit<Vec<(T, Vec<T>, Vec<bool>)>, Vec<Vec<(T, T)>>> {
    let builder = new_builder().with_consts();
    let consts = builder.get_const_wire_ids().unwrap();
    // each party inputs its share of alpha, 2*num_inputs*num_parties shares of zero, and num_inputs inputs
    let mut builder = builder.refine_input::<Vec<(T, Vec<T>, Vec<bool>)>>();
    let mut alphas = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut xs_s = Vec::with_capacity(num_parties * num_inputs);

    let num_shares = (num_inputs + T::BIT_SIZE - 1) / T::BIT_SIZE;
    let mut zs_s = Vec::with_capacity(T::BIT_SIZE * num_parties * (threshold - 1) * num_shares * 2);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, zs_i) = b2.add_input_multi::<Vec<T>>(2 * num_shares * (threshold - 1), None);
        let (b2, xs_i) = b2.add_input_multi::<Vec<bool>>(num_inputs, None);
        // coalesce the input
        builder = b2.refine_input();

        alphas.extend(alpha_i);
        zs_s.extend(zs_i);
        xs_s.extend(xs_i);
    }

    // sum inputs
    let sum_xs_c = sum_many_circuit::<[bool; 1]>(num_inputs, num_parties);
    let (builder, mut ys) = builder.extend_circuit(&xs_s, &sum_xs_c, None);
    ys.resize(T::BIT_SIZE * num_shares, consts[0]);

    input_sample_inner(
        builder,
        num_parties,
        num_shares,
        threshold,
        points,
        &alphas,
        &ys,
        &zs_s,
    )
}

/// Takes an arbitrary circuit with field element inputs with arbitrary output
/// then creates a circuit that takes in authenticated inputs, verifies the inputs,
/// and then returns an additional bit whether the input verification passed.
pub fn add_validation_project<T: CircuitRing, I, O>(
    num_parties: usize,
    num_shares: usize,
    bit_selection: &[usize],
    c: &TCircuit<I, O>,
) -> TCircuit<Vec<(T, Vec<T>, Vec<T>)>, Option<O>> {
    let builder = new_builder();
    // the values are input are (\shares{(a, \vec{xs}, \vec{m})}_1, \shares{(...)}_2, ...)
    // where each party has all of its input wires adjacent.
    let mut builder = builder.refine_input::<Vec<(T, Vec<T>, Vec<T>)>>();
    let mut alpha_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut inputs_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties * num_shares);
    let mut macs_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties * num_shares);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, xs_i) = b2.add_input_multi::<Vec<T>>(num_shares, None);
        let (b2, macs_i) = b2.add_input_multi::<Vec<T>>(num_shares, None);
        // coalesce the input
        builder = b2.refine_input::<Vec<(T, Vec<T>, Vec<T>)>>();

        alpha_shares_ids.extend(alpha_i);
        inputs_shares_ids.extend(xs_i);
        macs_shares_ids.extend(macs_i);
    }

    // Calculate the value of the mac key alpha by summing all of the shares
    let sum_p = sum_circuit::<T>(num_parties);
    let (builder, alpha_ids) = builder.extend_circuit(&alpha_shares_ids, &sum_p, Some("alpha"));

    // Reconstruct all of the values of the xs and all of the values of the macs
    let sum_x = sum_many_circuit::<T>(num_shares, num_parties);

    let (builder, xs_ids) = builder.extend_circuit(&inputs_shares_ids, &sum_x, Some("xs"));
    let (builder, macs_ids) = builder.extend_circuit(&macs_shares_ids, &sum_x, Some("macs"));

    dbg!("eval circuit", num_shares, c.inputs.len());
    // Re-calculate the MAC alpha * \vec{xs}
    let lin_x = linear::<T>(num_shares);
    let alpha_xs_ids: Vec<_> = alpha_ids
        .iter()
        .cloned()
        .chain(xs_ids.iter().cloned())
        .collect();

    let (builder, linear_ids) = builder.extend_circuit(&alpha_xs_ids, &lin_x, Some("check_macs"));

    // Check that all of the MACs are valid
    let checks_ids: Vec<_> = linear_ids.iter().cloned().chain(macs_ids.clone()).collect();
    let compare_n = compare_all::<T>(num_shares);

    let (builder, check_id) = builder.extend_circuit(&checks_ids, &compare_n, Some("valid_bit"));

    // gather the bits of input from all of the field elements
    let proj_xs: Vec<_> = bit_selection.iter().map(|&i| xs_ids[i].clone()).collect();

    // Actually compute the original circuit
    let (builder, original_output) = builder.extend_circuit(&proj_xs, c, Some("original_output"));

    // Mask the output if the MAC check failed
    let mask_ids: Vec<_> = check_id
        .iter()
        .cloned()
        .chain(original_output.into_iter())
        .collect();

    let mask_c = mask::<O>(mask_ids.len() - 1);
    let (builder, masked_output_ids) =
        builder.extend_circuit(&mask_ids, &mask_c, Some("masked_output"));

    // Re-specify which wires are output
    let builder = builder.refine_output::<Option<O>>(&masked_output_ids);

    builder.to_circuit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuits::{aes::aes_key_schedule, *},
        ff2_128::*,
    };

    fn example_setup_circuit<I: CircuitRing, O>(
        np: usize,
        cir: &TCircuit<I, O>,
    ) -> TCircuit<Vec<I>, (I, O)> {
        let b = new_builder();
        let (b, ids) = b.add_input_multi::<Vec<I>>(np, None);
        let sum = sum_circuit::<I>(np);
        let (b, k) = b.extend_circuit(&ids, &sum, None);
        let (b, mut keys) = b.extend_circuit(&k, cir, None);
        let mut outs = k;
        outs.append(&mut keys);

        b.refine_input().refine_output(&outs).to_circuit()
    }

    #[test]
    fn thresh_circuits_well_formed() {
        let points: Vec<_> = (1..=5).map(|i| FF2_128::from(i)).collect();

        init_circuit(5, 3, &points).well_formed();

        // deliberately not divisible by 128 inputs
        sample_circuit(5, 129, 3, &points).well_formed();

        let aes = aes_key_schedule();

        let c = setup_circuit(5, 128, 3, &points, &example_setup_circuit(5, &aes)).well_formed();
        println!("{}", c.describe());
    }
}
