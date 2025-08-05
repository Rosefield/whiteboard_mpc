use crate::{
    circuits::{
        builder::new_builder,
        circuit::{Gate, TCircuit},
        elements::CircuitElement,
        get_def_circuit, WireId,
    },
    ff2_128::FF2_128,
    rr2_128::RR2_128,
};

use std::marker::PhantomData;

/// Trait that represents basic ring operations as boolean circuits
pub trait CircuitRing: CircuitElement {
    /// Produces the circuit that adds two elements
    fn add_circuit() -> TCircuit<(Self, Self), Self>;
    /// Produces the circuit that multiplies two elements
    fn mul_circuit() -> TCircuit<(Self, Self), Self>;
    /// Produces a circuit that given a polynomial, evaluates it on each of a list of constant inputs
    fn const_poly_eval_circuit(
        t: usize,
        consts: &[Self],
        use_zero_coeff: bool,
    ) -> TCircuit<Vec<Self>, Vec<Self>>;
}

fn ff2_128_reduce_circuit() -> TCircuit<[bool; 256], [bool; 128]> {
    let mul_hi = |lhs: &[WireId], xs: &[WireId], next_id: WireId| {
        // y[..7] = (x >> 57) ^ (x >> 62) ^ (x >> 63)
        // y[0] = x[57] ^ x[62} ^ x[63], y[1] = x[58] ^ x[63], y[2..7] = x[59..64]
        // output z = lhs[..64] ^ (y[..7] || 0)
        let ys: Vec<_> = [next_id + 1, next_id + 2]
            .into_iter()
            .chain(xs[59..64].iter().cloned())
            .collect();
        let gates: Vec<_> = [
            Gate::Xor(xs[57], xs[62], next_id),
            Gate::Xor(next_id, xs[63], next_id + 1),
            Gate::Xor(xs[58], xs[63], next_id + 2),
        ]
        .into_iter()
        .chain((0..7).map(|l| Gate::Xor(lhs[l], ys[l], next_id + 3 + l)))
        .collect();
        let out_wires: Vec<WireId> = (next_id + 3..next_id + 10)
            .chain(lhs[7..64].iter().cloned())
            .collect();

        (gates, out_wires, 10)
    };

    let mul_lo = |xs: &[WireId], next_id: WireId| {
        // y[..64] = x ^ (x <<1 ) ^ (x << 2) ^ (x << 7)
        // y[0] = x[0]
        // y[1] = x[1] ^ x[0]
        // y[2..7] = x[2..7] ^ x[1..6] ^ x[0..5]
        // y[8..] = x[8..] ^ x[7..63] ^ x[6..62] ^ x[..56]

        // m1[..63] = x[1..64] ^ x[0..63]
        let gates: Vec<_> = (0..63)
            .map(|l| Gate::Xor(xs[l + 1], xs[l], next_id + l))
            // m2[..62] = m1[1..63] ^ x[0..62]
            .chain((0..62).map(|l| Gate::Xor(next_id + 1 + l, xs[l], next_id + 63 + l)))
            // m3[..57] = m2[5..62] ^ x[0..57]
            .chain((0..57).map(|l| Gate::Xor(next_id + 68 + l, xs[l], next_id + 125 + l)))
            .collect();
        let out_wires: Vec<WireId> = [xs[0], next_id]
            .into_iter()
            .chain(next_id + 63..next_id + 68)
            .chain(next_id + 125..next_id + 182)
            .collect();

        (gates, out_wires, 63 + 62 + 57)
    };

    let poly_reduce_c: TCircuit<[bool; 256], [bool; 128]> = {
        let inputs = Vec::from_iter(0..256);
        let mut gates = Vec::new();
        let t1h = &inputs[192..256];
        let next_id = 256;
        let (g1, t1l, n1) = mul_hi(&inputs[128..192], t1h, next_id);
        let (g2, thl, n2) = mul_lo(&t1h, next_id + n1);
        let (g3, mh, n3) = mul_hi(&thl, &t1l, next_id + n1 + n2);
        let (g4, ml, n4) = mul_lo(&t1l, next_id + n1 + n2 + n3);
        let next_id = next_id + n1 + n2 + n3 + n4;
        let mut m = ml;
        m.extend(mh);

        let gout = (0..128).map(|l| Gate::Xor(inputs[l], m[l], next_id + l));
        gates.extend(g1);
        gates.extend(g2);
        gates.extend(g3);
        gates.extend(g4);
        gates.extend(gout);

        let outputs = (next_id..next_id + 128).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    };
    poly_reduce_c
}

impl<const N: usize> CircuitRing for [bool; N] {
    fn add_circuit() -> TCircuit<(Self, Self), Self> {
        let inputs: Vec<_> = (0..N + N).collect();
        let gates = (0..N).map(|i| Gate::Xor(i, N + i, N + N + i)).collect();

        let outputs = (2 * N..3 * N).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    }

    fn mul_circuit() -> TCircuit<(Self, Self), Self> {
        let inputs: Vec<_> = (0..N + N).collect();
        let gates = (0..N).map(|i| Gate::And(i, N + i, N + N + i)).collect();

        let outputs = (2 * N..3 * N).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    }

    fn const_poly_eval_circuit(
        _t: usize,
        _consts: &[Self],
        _use_zero_coeff: bool,
    ) -> TCircuit<Vec<Self>, Vec<Self>> {
        unimplemented!()
    }
}

impl CircuitRing for FF2_128 {
    fn add_circuit() -> TCircuit<(Self, Self), Self> {
        // addition is the same between this and RR2_128, just xor
        TCircuit::read_from_file_bf(&get_def_circuit("rr2_128_add.txt")).unwrap()
    }

    fn mul_circuit() -> TCircuit<(Self, Self), Self> {
        TCircuit::read_from_file_bf(&get_def_circuit("ff2_128_mul.txt")).unwrap()
    }

    /// Produces a circuit that given a polynomial, evaluates it on each of a list of constant inputs.
    /// if `use_zero_coeff` == false then the circuit will ignore the constant coefficient
    fn const_poly_eval_circuit(
        t: usize,
        consts: &[Self],
        use_zero_coeff: bool,
    ) -> TCircuit<Vec<Self>, Vec<Self>> {
        let builder = new_builder().with_consts();
        let c_wires = builder.get_const_wire_ids().unwrap();
        let n_coeffs = if use_zero_coeff { t } else { t - 1 };
        let (builder, coeffs) = builder.add_input_multi::<Vec<Self>>(n_coeffs, None);

        let reduce_c = ff2_128_reduce_circuit();

        let s = Self::BIT_SIZE;

        let mut builder = builder.refine_input::<Vec<Self>>();

        let mut cmul_in = coeffs.clone();
        cmul_in.push(c_wires[0]);

        let mut outputs = Vec::with_capacity(s * consts.len());

        for c in consts {
            // each of the coefficients, and 1 constant 0 input wire
            let inputs: Vec<_> = (0..n_coeffs * s + 1).collect();
            let zero_id = *inputs.last().unwrap();
            let mut bits = vec![false; Self::BIT_SIZE];

            let mut x = c.clone();

            // "Allocate" a set of wires to use as the full evaluation of the poly
            let mut sum = vec![zero_id; 2 * s];
            if use_zero_coeff {
                sum[..s].copy_from_slice(&inputs[..s]);
            }

            let mut next_id = zero_id + 1;
            let mut gates = Vec::with_capacity(s * t);

            let start = if use_zero_coeff { 1 } else { 0 };
            for (_, coeff) in inputs.chunks_exact(s).skip(start).enumerate() {
                x.to_bits(&mut bits);

                for (j, b) in bits.iter().enumerate() {
                    if *b {
                        for k in 0..s {
                            gates.push(Gate::Xor(sum[j + k], coeff[k], next_id));
                            sum[j + k] = next_id;
                            next_id += 1;
                        }
                    }
                }
                x *= c;
            }

            let c_mul: TCircuit<Vec<Self>, [bool; 256]> = TCircuit {
                _i: PhantomData,
                _o: PhantomData,
                inputs: inputs,
                gates: gates,
                outputs: sum,
            };

            let (b2, mul) = builder.extend_circuit(&cmul_in, &c_mul, None);
            let (b2, red) = b2.extend_circuit(&mul, &reduce_c, None);
            outputs.extend(red);

            //println!("cmul: in: {:?}, gates: {}, out: {:?}", c_mul.inputs, c_mul.gates.len(), c_mul.outputs);

            builder = b2.refine_output::<()>(&[]);
        }

        builder.refine_output::<Vec<Self>>(&outputs).to_circuit()
    }
}

impl CircuitRing for RR2_128 {
    fn add_circuit() -> TCircuit<(Self, Self), Self> {
        TCircuit::read_from_file_bf(&get_def_circuit("rr2_128_add.txt")).unwrap()
    }

    fn mul_circuit() -> TCircuit<(Self, Self), Self> {
        TCircuit::read_from_file_bf(&get_def_circuit("rr2_128_mul.txt")).unwrap()
    }

    /// Produces a circuit that given a polynomial, evaluates it on each of a list of constant inputs
    fn const_poly_eval_circuit(
        _t: usize,
        _consts: &[Self],
        _use_zero_coeff: bool,
    ) -> TCircuit<Vec<Self>, Vec<Self>> {
        unimplemented!();
    }
}

/// Produces a circuit that given an array of inputs of size `num' outputs the sum
/// of all of the inputs
pub fn sum_circuit<T: CircuitRing>(num: usize) -> TCircuit<Vec<T>, T> {
    let add_circuit = T::add_circuit();
    let builder = new_builder();
    let (builder, inputs) = builder.add_input_multi::<Vec<T>>(num, None);
    let (builder, outputs) = builder.fold_circuit(&inputs, &add_circuit, num, None);
    let builder = builder.refine_output(&outputs).refine_input();

    builder.to_circuit()
}

/// Produces a circuit that given an array of inputs of size num_inputs*num_shares, interprets as
/// A: [[T; inputs]; shares] and outputs the SIMD sum over the outer array, A[0] + A[1] + ...
pub fn sum_many_circuit<T: CircuitRing>(
    num_inputs: usize,
    num_shares: usize,
) -> TCircuit<Vec<T>, Vec<T>> {
    let add_circuit = T::add_circuit();
    let mut all_outputs: Vec<usize> = Vec::new();
    let el_size: usize = T::BIT_SIZE;

    let builder = new_builder();
    let (mut builder, inputs) = builder.add_input_multi::<Vec<T>>(num_inputs * num_shares, None);
    for i in 0..num_inputs {
        let interm_inputs: Vec<usize> = inputs
            .chunks_exact(el_size)
            .enumerate()
            .filter(|&(ind, _)| ind % num_inputs == i)
            .map(|(_, e)| e)
            .flatten()
            .cloned()
            .collect();

        let (builder2, outputs) =
            builder.fold_circuit(&interm_inputs, &add_circuit, num_shares, None);
        builder = builder2.refine_output(&[]);
        all_outputs.extend(outputs);
    }

    let builder = builder.refine_output::<Vec<T>>(&all_outputs).refine_input();

    builder.to_circuit()
}

/// Produces a circuit that given a scalar a and an array b of size num
/// outputs [a*b[0], a*b[1], ...]
pub fn linear<T: CircuitRing>(num: usize) -> TCircuit<(T, Vec<T>), Vec<T>> {
    let builder = new_builder();
    let (builder, a) = builder.add_input::<T>(None);
    let (builder, b) = builder.add_input_multi::<Vec<T>>(num, None);

    let mul_c = T::mul_circuit();
    let in_fn = |el: &[WireId]| {
        let mut ins = a.clone();
        ins.extend(el);
        ins
    };
    let (builder, macs) = builder.map_circuit(&b, &mul_c, num, in_fn, None);

    let builder = builder.refine_input().refine_output::<Vec<T>>(&macs);

    builder.to_circuit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_ff2_128_mul_circuit() {
        // A reimplementation of FF2_128::mul_assign as a boolean circuit
        let builder = new_builder();
        // inputs are the little-endian bits of the field elements
        // e.g. the polynomial x^5 + x -> 0100010....
        let (builder, in1) = builder.add_input::<FF2_128>(Some("in1"));
        let (builder, in2) = builder.add_input::<FF2_128>(Some("in2"));
        let builder = builder.refine_input::<(FF2_128, FF2_128)>();

        let half_mul_circuit: TCircuit<([bool; 64], [bool; 64]), [bool; 127]> =
            TCircuit::read_from_file_peralta(&get_def_circuit("poly_64b_mul_peralta.txt"))
                .unwrap()
                .checked();

        // Split input to (a1 || a0), (b1 || b0)
        // Want to compute (a0b0 * 2^128 + (a0+a1)(b0+b1) * 2^64 + (a0b0 + a1b1) * 2^64 + a1bl
        // then reduce by the polynomial x^128 + x^7 + x^2 + x + 1
        let (i1_hi, i1_lo) = (&in1[64..], &in1[..64]);
        let (i2_hi, i2_lo) = (&in2[64..], &in2[..64]);

        let hi_in: Vec<_> = i1_hi.iter().chain(i2_hi.iter()).cloned().collect();
        let lo_in: Vec<_> = i1_lo.iter().chain(i2_lo.iter()).cloned().collect();

        let (builder, hi_bits) = builder.extend_circuit(&hi_in, &half_mul_circuit, None);
        let (builder, lo_bits) = builder.extend_circuit(&lo_in, &half_mul_circuit, None);

        let sum_halves_c: TCircuit<([bool; 128], [bool; 128]), ([bool; 64], [bool; 64])> = {
            let inputs = (0..256).collect();
            let gates = (0..64)
                .map(|l| Gate::Xor(l, l + 64, 256 + l))
                .chain((128..192).map(|l| Gate::Xor(l, l + 64, 192 + l)))
                .collect();
            let outputs = (256..384).collect();
            TCircuit::from_parts(inputs, gates, outputs)
        };

        let all_ins: Vec<_> = in1.iter().chain(in2.iter()).cloned().collect();

        let (builder, mid_in) = builder.extend_circuit(&all_ins, &sum_halves_c, None);
        let (builder, mid_bits) = builder.extend_circuit(&mid_in, &half_mul_circuit, None);

        // input is hi[..127] || mid[..127]|| lo[..127]
        // output is the full multiplication of ab
        let sum_mul_c: TCircuit<[bool; 381], [bool; 256]> = {
            let inputs: Vec<_> = (0..381).collect();
            let hi = &inputs[..127];
            let mid = &inputs[127..254];
            let lo = &inputs[254..381];
            // first (hi + lo)
            let gates = (0..127)
                .map(|l| Gate::Xor(hi[l], lo[l], 381 + l))
                // then m = (hi +lo) + mid
                .chain((0..127).map(|l| Gate::Xor(mid[l], 381 + l, 509 + l)))
                // low bits of hi, high bits of m
                // then m1[..63] = hi[..63] + m[64..]
                .chain((0..63).map(|l| Gate::Xor(hi[l], 509 + 64 + l, 636 + l)))
                // high bits of lo, low bits of m
                // then m2[..63] = lo[64..] + m[..63]
                .chain((0..63).map(|l| Gate::Xor(lo[64 + l], 509 + l, 699 + l)))
                // make a zero
                // TODO: consts
                .chain([Gate::Xor(0, 0, 762)].into_iter())
                .collect();
            // output is lo[..64] || m2[..63] || m[63] || m1[..63] || hi[63..] || 0
            let outputs = (254..318)
                .chain(699..762)
                .chain([509 + 63].into_iter())
                .chain(636..699)
                .chain(63..127)
                .chain(762..763)
                .collect();
            TCircuit::from_parts(inputs, gates, outputs)
        };

        let mut muls_wires = hi_bits.clone();
        muls_wires.extend(mid_bits.iter());
        muls_wires.extend(lo_bits.iter());

        let builder = builder.refine_output::<[bool; 381]>(&muls_wires);
        let (builder, ts) = builder.extend_circuit(&muls_wires, &sum_mul_c, None);

        let poly_reduce_c = ff2_128_reduce_circuit();

        let (builder, res) = builder.extend_circuit(&ts, &poly_reduce_c, None);
        let builder = builder.refine_output::<FF2_128>(&res);

        let c = builder.to_circuit_checked();
        c.write_to_file_bf(&get_def_circuit("ff2_128_mul.txt"));
    }
}
