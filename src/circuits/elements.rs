use crate::{
    circuits::{
        builder::new_builder,
        circuit::{Gate, TCircuit},
        WireId,
    },
    ff2_128::FF2_128,
    field::ToFromBytes,
    rr2_128::RR2_128,
};

use std::marker::PhantomData;

/// Elements that can convert to/from boolean circuit representations
/// e.g. have a fixed size, and can convert to and from bit reprentations
pub trait CircuitElement: Sized {
    const BIT_SIZE: usize;

    fn to_bits(&self, b: &mut [bool]);
    fn from_bits(b: &[bool]) -> Self;
}

/// Collections of elements that can be converted to/from a circuit representation
/// Where the type of the collection does not allow statically knowing how many
/// elements are available.
pub trait CircuitCollection {
    fn total_size(num_els: usize) -> usize;
    fn to_bits(&self, b: &mut [bool]);
    fn from_bits(num_els: usize, b: &[bool]) -> Self;
}

impl<T: CircuitElement> CircuitCollection for Vec<T> {
    fn total_size(num_els: usize) -> usize {
        num_els * T::BIT_SIZE
    }

    fn to_bits(&self, b: &mut [bool]) {
        let s = self.len() * T::BIT_SIZE;
        assert!(b.len() >= s);
        self.iter()
            .zip((&mut b[..s]).chunks_exact_mut(T::BIT_SIZE))
            .for_each(|(i, buf)| i.to_bits(buf));
    }

    fn from_bits(num_els: usize, b: &[bool]) -> Self {
        let s = T::BIT_SIZE * num_els;
        assert!(b.len() >= s);

        (&b[..s])
            .chunks_exact(T::BIT_SIZE)
            .map(|c| T::from_bits(c))
            .collect()
    }
}

impl CircuitElement for bool {
    const BIT_SIZE: usize = 1;

    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= 1);
        b[0] = *self;
    }

    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= 1);
        b[0]
    }
}

impl<const N: usize> CircuitElement for [bool; N] {
    const BIT_SIZE: usize = N;

    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= N);
        b[..N].copy_from_slice(&self[..]);
    }

    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= N);
        let mut s = [false; N];
        s.copy_from_slice(&b[..N]);
        s
    }
}

impl CircuitElement for FF2_128 {
    const BIT_SIZE: usize = 128;

    // writes to b as little-endian bits
    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= 128);
        let mut bytes = [0; 16];
        self.to_bytes(&mut bytes);
        bytes.into_iter().enumerate().for_each(|(i, byte)| {
            for j in 0..8 {
                b[8 * i + j] = ((byte >> j) & 1) == 1;
            }
        });
    }

    // reads from b as little-endian bits
    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= 128);
        let mut bytes = [0; 16];

        for i in 0..16 {
            let mut byte = 0;
            for j in 0..8 {
                byte |= u8::from(b[8 * i + j]) << j;
            }
            bytes[i] = byte;
        }

        Self::from_bytes(&bytes)
    }
}

impl CircuitElement for RR2_128 {
    const BIT_SIZE: usize = 128;

    // writes to b as little-endian bits
    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= 128);
        let mut bytes = [0; 16];
        self.to_bytes(&mut bytes);
        bytes.into_iter().enumerate().for_each(|(i, byte)| {
            for j in 0..8 {
                b[8 * i + j] = ((byte >> j) & 1) == 1;
            }
        });
    }

    // reads from b as little-endian bits
    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= 128);
        let mut bytes = [0; 16];

        for i in 0..16 {
            let mut byte = 0;
            for j in 0..8 {
                byte |= u8::from(b[8 * i + j]) << j;
            }
            bytes[i] = byte;
        }

        Self::from_bytes(&bytes)
    }
}

impl<T: CircuitElement> CircuitElement for Option<T> {
    const BIT_SIZE: usize = 1 + T::BIT_SIZE;

    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= Self::BIT_SIZE);

        if let Some(t) = self {
            b[0] = true;
            t.to_bits(&mut b[1..]);
            return;
        } else {
            b[0] = false;
        }
    }

    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= Self::BIT_SIZE);

        if !b[0] {
            return None;
        }

        return Some(T::from_bits(&b[1..]));
    }
}

impl<T1: CircuitElement, T2: CircuitElement> CircuitElement for (T1, T2) {
    const BIT_SIZE: usize = T1::BIT_SIZE + T2::BIT_SIZE;

    fn to_bits(&self, b: &mut [bool]) {
        assert!(b.len() >= Self::BIT_SIZE);

        self.0.to_bits(b);
        self.1.to_bits(&mut b[T1::BIT_SIZE..])
    }

    fn from_bits(b: &[bool]) -> Self {
        assert!(b.len() >= Self::BIT_SIZE);

        (T1::from_bits(b), T2::from_bits(&b[T1::BIT_SIZE..]))
    }
}

/// Produces a circuit that given two arrays (a,b) of size num, calculates the predicate
/// a[0] == b[0] AND a[1] == b[1] AND ...
pub fn compare_all<T: CircuitElement>(num: usize) -> TCircuit<(Vec<T>, Vec<T>), bool> {
    let s: usize = T::BIT_SIZE;
    let half = s * num;
    let inputs: Vec<WireId> = (0..2 * half).collect();
    let a = &inputs[..half];
    let b = &inputs[half..];
    let next = 2 * half;
    let gates: Vec<Gate> = (0..half)
        // find bits that are different
        .map(|l| Gate::Xor(a[l], b[l], next + l))
        // flip to be bits that are the same
        .chain((0..half).map(|l| Gate::Not(next + l, next + half + l)))
        // reduce to check that all bits are the same
        .chain([Gate::And(next + half, next + half + 1, next + next)].into_iter())
        .chain(
            (2..half).map(|l| Gate::And(next + half + l, next + next + l - 2, next + next + l - 1)),
        )
        .collect();
    // output is the last wire in the reduction
    let outputs = vec![gates.last().unwrap().output_id()];

    TCircuit {
        _i: PhantomData,
        _o: PhantomData,
        inputs,
        gates,
        outputs,
    }
}

/// Produces a circuit that just reinterprets its input as an array of bits
/// instead of any particular integer
pub fn bit_decompose<T: CircuitElement, const S: usize>() -> TCircuit<T, [bool; S]> {
    assert!(T::BIT_SIZE == S);
    let b = new_builder();
    let (b, i) = b.add_input::<T>(None);
    let b = b.refine_input::<T>();
    let o = b.refine_output::<[bool; S]>(&i);

    o.to_circuit()
}

pub fn out_mask<I, O: CircuitElement>(c: &TCircuit<I, O>) -> TCircuit<(O, I), O> {
    let b = new_builder();
    let (b, mut mask) = b.add_input::<O>(None);
    let (b, input) = b.add_input_multi::<Vec<bool>>(c.inputs.len(), None);
    let (b, mut out1) = b.extend_circuit(&input[..], c, None);
    let osize = O::BIT_SIZE;
    let c2: TCircuit<(O, O), O> = {
        let inputs = (0..osize * 2).collect();
        let gates = (0..osize)
            .map(|i| Gate::Xor(i, i + osize, 2 * osize + i))
            .collect();
        let outputs = (osize * 2..osize * 3).collect();

        TCircuit::from_parts(inputs, gates, outputs)
    };

    mask.append(&mut out1);
    let (b, out2) = b.extend_circuit(&mask, &c2, None);

    b.refine_input().refine_output(&out2).to_circuit()
}

/// Produces the circuit that given an arbitrary bit string O and a mask bit b,
/// returns (b, O & b)
pub fn mask<O>(o_size: usize) -> TCircuit<(bool, O), Option<O>> {
    let inputs: Vec<_> = (0..=o_size).collect();
    let gates: Vec<_> = (1..=o_size).map(|r| Gate::And(0, r, o_size + r)).collect();
    let outputs: Vec<_> = [0]
        .into_iter()
        .chain(o_size + 1..=o_size + o_size)
        .collect();

    TCircuit {
        _i: PhantomData,
        _o: PhantomData,
        inputs,
        gates,
        outputs,
    }
}

pub fn hardcode_input<I: CircuitElement, I2: CircuitElement, O>(
    input: I,
    circuit: &TCircuit<(I, I2), O>,
) -> TCircuit<I2, O> {
    let builder = new_builder().with_consts();
    let consts = builder.get_const_wire_ids().unwrap();
    let (builder, mut remaining) = builder.add_input::<I2>(None);

    let mut ibits = vec![false; I::BIT_SIZE];
    input.to_bits(&mut ibits);

    let mut hardcoded: Vec<_> = ibits
        .iter()
        .map(|&x| consts[if x { 1 } else { 0 }])
        .collect();
    hardcoded.append(&mut remaining);

    let (builder, out) = builder.extend_circuit(&hardcoded[..], circuit, None);

    builder.refine_input().refine_output(&out).to_circuit()
}
