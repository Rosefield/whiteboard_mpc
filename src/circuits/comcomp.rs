use crate::circuits::{
    arith::{linear, sum_circuit, sum_many_circuit, CircuitRing},
    builder::{new_builder, CircuitBuilder},
    circuit::TCircuit,
    elements::{compare_all, mask},
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
    num_inputs: usize,
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
    let sum_zs_c = sum_many_circuit::<T>(2 * num_inputs * (threshold - 1), num_parties);
    let (builder, fs) = builder.extend_circuit(&zs_s, &sum_zs_c, None);
    // fs = f_1(.), f_2(.), ...
    // need to create [0_1]_i = f_1(i), [0_2]_i = f_2(i), ...
    let const_poly_eval_c = T::const_poly_eval_circuit(threshold, points, false);

    let (builder, zs) = builder.map_circuit(
        &fs,
        &const_poly_eval_c,
        2 * num_inputs,
        |chunk| chunk.to_vec(),
        None,
    );

    // calculate ax_0, ax_1, ..
    let linear_c = linear::<T>(num_inputs);
    let mut macs_in = alpha;
    macs_in.extend(xs);
    let (builder, macs) = builder.extend_circuit(&macs_in, &linear_c, None);

    // calculate party output shares
    let mut shares_in = Vec::with_capacity(T::BIT_SIZE * (2 * 2 * num_inputs * num_parties));
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
    let sum_halves_c = sum_many_circuit::<T>(2 * num_inputs * num_parties, 2);
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
        for j in 0..num_inputs {
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

/// The circuit used for producing authenticated shares in the `input` instruction
/// of F_comcomp.
pub fn input_circuit<T: CircuitRing>(
    num_parties: usize,
    num_inputs: usize,
    threshold: usize,
    points: &[T],
) -> TCircuit<(Vec<T>, Vec<(T, Vec<T>)>), Vec<Vec<(T, T)>>> {
    let builder = new_builder();
    let (builder, xs) = builder.add_input_multi::<Vec<T>>(num_inputs, None);
    let mut builder = builder.refine_input::<(Vec<T>, Vec<(T, Vec<T>)>)>();
    // each party inputs its share of alpha, 2*num_inputs*num_parties shares of zero
    let mut alphas = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut zs_s = Vec::with_capacity(T::BIT_SIZE * num_parties * (threshold - 1) * num_inputs * 2);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, zs_i) = b2.add_input_multi::<Vec<T>>(2 * num_inputs * (threshold - 1), None);
        // coalesce the input
        builder = b2.refine_input::<(Vec<T>, Vec<(T, Vec<T>)>)>();

        alphas.extend(alpha_i);
        zs_s.extend(zs_i);
    }

    input_sample_inner(
        builder,
        num_parties,
        num_inputs,
        threshold,
        points,
        &alphas,
        &xs,
        &zs_s,
    )
}

/// The circuit used for producing authenticated shares in the `sample` instruction
/// of F_comcomp.
pub fn sample_circuit<T: CircuitRing>(
    num_parties: usize,
    num_inputs: usize,
    threshold: usize,
    points: &[T],
) -> TCircuit<Vec<(T, Vec<T>, Vec<T>)>, Vec<Vec<(T, T)>>> {
    let builder = new_builder();
    // each party inputs its share of alpha, 2*num_inputs*num_parties shares of zero, and num_inputs inputs
    let mut builder = builder.refine_input::<Vec<(T, Vec<T>, Vec<T>)>>();
    let mut alphas = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut xs_s = Vec::with_capacity(T::BIT_SIZE * num_parties * num_inputs);
    let mut zs_s = Vec::with_capacity(T::BIT_SIZE * num_parties * (threshold - 1) * num_inputs * 2);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, zs_i) = b2.add_input_multi::<Vec<T>>(2 * num_inputs * (threshold - 1), None);
        let (b2, xs_i) = b2.add_input_multi::<Vec<T>>(num_inputs, None);
        // coalesce the input
        builder = b2.refine_input::<Vec<(T, Vec<T>, Vec<T>)>>();

        alphas.extend(alpha_i);
        zs_s.extend(zs_i);
        xs_s.extend(xs_i);
    }

    // sum inputs
    let sum_xs_c = sum_many_circuit::<T>(num_inputs, num_parties);
    let (builder, xs) = builder.extend_circuit(&xs_s, &sum_xs_c, None);

    input_sample_inner(
        builder,
        num_parties,
        num_inputs,
        threshold,
        points,
        &alphas,
        &xs,
        &zs_s,
    )
}

/// Takes an arbitrary circuit with field element inputs with arbitrary output
/// then creates a circuit that takes in authenticated inputs, verifies the inputs,
/// and then returns an additional bit whether the input verification passed.
pub fn add_validation<T: CircuitRing, O>(
    num_parties: usize,
    num_inputs: usize,
    c: &TCircuit<Vec<T>, O>,
) -> TCircuit<Vec<(T, Vec<T>, Vec<T>)>, Option<O>> {
    let builder = new_builder();
    // the values are input are (\shares{(a, \vec{xs}, \vec{m})}_1, \shares{(...)}_2, ...)
    // where each party has all of its input wires adjacent.
    let mut builder = builder.refine_input::<Vec<(T, Vec<T>, Vec<T>)>>();
    let mut alpha_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties);
    let mut inputs_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties * num_inputs);
    let mut macs_shares_ids = Vec::with_capacity(T::BIT_SIZE * num_parties * num_inputs);
    for _ in 0..num_parties {
        let (b2, alpha_i) = builder.add_input::<T>(None);
        let (b2, xs_i) = b2.add_input_multi::<Vec<T>>(num_inputs, None);
        let (b2, macs_i) = b2.add_input_multi::<Vec<T>>(num_inputs, None);
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
    let sum_x = sum_many_circuit::<T>(num_inputs, num_parties);

    let (builder, xs_ids) = builder.extend_circuit(&inputs_shares_ids, &sum_x, Some("xs"));
    let (builder, macs_ids) = builder.extend_circuit(&macs_shares_ids, &sum_x, Some("macs"));

    // Re-calculate the MAC alpha * \vec{xs}
    let lin_x = linear::<T>(num_inputs);
    let alpha_xs_ids: Vec<_> = alpha_ids
        .iter()
        .cloned()
        .chain(xs_ids.iter().cloned())
        .collect();

    let (builder, linear_ids) = builder.extend_circuit(&alpha_xs_ids, &lin_x, Some("check_macs"));

    // Check that all of the MACs are valid
    let checks_ids: Vec<_> = linear_ids.iter().cloned().chain(macs_ids.clone()).collect();
    let compare_n = compare_all::<T>(num_inputs);

    let (builder, check_id) = builder.extend_circuit(&checks_ids, &compare_n, Some("valid_bit"));

    // Actually compute the original circuit
    let (builder, original_output) = builder.extend_circuit(&xs_ids, c, Some("original_output"));

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
