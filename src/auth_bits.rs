use crate::{
    circuits::CircuitElement,
    field::{Field, Ring},
    polynomial::lagrange_poly,
};

use serde::{Deserialize, Serialize};

#[derive(Debug)]
/// A single authenticated bit over a ring F
pub struct Abit<F> {
    pub bit: bool,
    pub macs: Vec<F>,
    pub keys: Vec<F>,
}

impl<F: Ring> Abit<F> {
    /// Add two abits together
    pub fn add_assign(&mut self, other: &Self) {
        self.bit ^= other.bit;
        self.macs
            .iter_mut()
            .zip(other.macs.iter())
            .for_each(|(m, o)| *m += o);

        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(m, o)| *m += o);
    }

    /// Add a public constant to this abit
    /// This is done by having the first party add the bit to their bit-share
    /// and each other party adjusts their key for the first party
    /// to incorporate the constant
    pub fn add_const(&mut self, c: bool, delta: &F, is_first: bool) {
        if is_first {
            self.bit ^= c;
        } else {
            self.keys[0] += delta;
        }
    }
}

#[derive(Debug)]
/// Multiple authenticated bits, in a Struct-of-Arrays form
/// where macs/keys is a (n-1) x k matrix, i.e. `abits.macs[j]`
/// is this party's macs under the j'th party's keys for each of the `bits`
pub struct Abits<F> {
    pub bits: Vec<bool>,
    pub macs: Vec<Vec<F>>,
    pub keys: Vec<Vec<F>>,
}

impl<F> Abits<F> {
    pub fn empty(np: usize) -> Self {
        Abits {
            bits: Vec::new(),
            macs: (0..np).map(|_| Vec::new()).collect(),
            keys: (0..np).map(|_| Vec::new()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn append(&mut self, mut other: Self) {
        self.bits.append(&mut other.bits);
        let n = self.macs.len();
        assert!(n == other.macs.len());

        for i in 0..n {
            self.macs[i].append(&mut other.macs[i]);
            self.keys[i].append(&mut other.keys[i]);
        }
    }
}

impl<F: Ring> Abits<F> {
    pub fn add_assign(&mut self, other: &Self) {
        self.bits
            .iter_mut()
            .zip(other.bits.iter())
            .for_each(|(b, o)| *b ^= o);

        let f_add = |s: &mut Vec<Vec<F>>, o: &Vec<Vec<F>>| {
            s.iter_mut()
                .flat_map(|x| x.iter_mut())
                .zip(o.iter().flatten())
                .for_each(|(x, y)| *x += y);
        };

        f_add(&mut self.macs, &other.macs);
        f_add(&mut self.keys, &other.keys);
    }
}

impl<F: Clone> From<&Abits<F>> for Vec<Abit<F>> {
    fn from(other: &Abits<F>) -> Self {
        let nbits = other.len();
        let np = other.macs.len();

        (0..nbits)
            .map(|i| Abit {
                bit: other.bits[i],
                macs: (0..np).map(|j| other.macs[j][i].clone()).collect(),
                keys: (0..np).map(|j| other.keys[j][i].clone()).collect(),
            })
            .collect()
    }
}

impl<F: Clone> From<&[Abit<F>]> for Abits<F> {
    fn from(other: &[Abit<F>]) -> Self {
        let mut bits = Vec::with_capacity(other.len());
        let mut macs: Vec<Vec<F>> = Vec::new();
        let mut keys: Vec<Vec<F>> = Vec::new();

        if other.len() > 0 {
            let np = other[0].macs.len();

            macs.extend((0..np).map(|_| Vec::with_capacity(other.len())));
            keys.extend((0..np).map(|_| Vec::with_capacity(other.len())));
        }

        for a in other.iter() {
            bits.push(a.bit);

            for (ms, m) in macs.iter_mut().zip(a.macs.iter()) {
                ms.push(m.clone());
            }
            for (ks, k) in keys.iter_mut().zip(a.keys.iter()) {
                ks.push(k.clone())
            }
        }

        Abits { bits, macs, keys }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreshAbits<F> {
    // the number of actual bits encoded
    // which may be less than shares.len() * F::BIT_SIZE
    // where the final element of shares would be under-packed
    pub nbits: usize,
    shares: Vec<F>,
    // #parties x #shares*F::BIT_SIZE
    macs: Vec<Vec<F>>,
    keys: Vec<Vec<F>>,
}

impl<F> ThreshAbits<F> {
    pub fn new() -> Self {
        ThreshAbits {
            nbits: 0,
            shares: Vec::new(),
            macs: Vec::new(),
            keys: Vec::new(),
        }
    }
}

impl<F: Field + CircuitElement + Copy> ThreshAbits<F> {
    /*
    pub fn add_assign(&mut self, other: &Self) {
        self.share += other.share;

        // technically this implementation is only correct for fields of characteristic 2
        self.macs
            .iter_mut()
            .zip(other.macs.iter())
            .for_each(|(m, o)| *m += o);

        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(m, o)| *m += o);
    }
    */

    pub fn add_consts(&mut self, cs: &[bool], delta: F) {
        assert!(self.nbits == cs.len());
        let num_bits = F::BIT_SIZE;

        cs.chunks_exact(num_bits)
            .zip(self.shares.iter_mut())
            .for_each(|(c, s)| {
                let t = F::from_bits(c);
                *s += t;
            });

        // if there is a remainder add to the last share
        let r = cs.chunks_exact(num_bits).remainder();
        if r.len() > 0 {
            let mut c_bits = vec![false; num_bits];
            c_bits[..r.len()].copy_from_slice(r);
            let x = F::from_bits(&c_bits);
            *self.shares.last_mut().unwrap() += x;
        }

        // implicitly the remaining bits are 0 so we won't need to many any modification to the remaining keys
        self.keys.iter_mut().for_each(|ks| {
            ks.iter_mut().zip(cs.iter()).for_each(|(k, c)| {
                if *c {
                    *k += delta;
                }
            });
        });

        // no changes to the MACs
    }

    pub fn from_abits(nbits: usize, shares: Vec<F>, abits: Abits<F>) -> Self {
        ThreshAbits {
            nbits: nbits,
            shares: shares,
            macs: abits.macs,
            keys: abits.keys,
        }
    }

    pub fn convert(
        &self,
        my_point: &F,
        all_points: &[F],
        sub_points: &[F],
        bit_idx: &[usize],
    ) -> Abits<F> {
        assert!(bit_idx.iter().all(|&i| i < self.nbits));
        let n = all_points.len();
        assert!(self.macs.len() == (n - 1));
        assert!(self.keys.len() == (n - 1));

        let my_idx = sub_points.iter().position(|x| x == my_point).unwrap();

        let bsize = F::BIT_SIZE;

        let (sub_idx, lp_i, lp_bits) = Self::convert_pre(my_point, all_points, sub_points);

        let mut add_share_bits = vec![false; bsize * self.shares.len()];

        add_share_bits
            .chunks_exact_mut(bsize)
            .zip(self.shares.iter())
            .for_each(|(c, s)| {
                let ls = lp_i.clone() * s;
                ls.to_bits(c);
            });

        let bits = bit_idx.iter().map(|&bi| add_share_bits[bi]).collect();
        let mut macs = Vec::with_capacity(sub_idx.len());
        let mut keys = Vec::with_capacity(sub_idx.len());

        let fis = |xs: &[F], i: usize, k: usize| {
            let mut x = F::zero();

            for j in 0..bsize {
                if lp_bits[i][j][k] {
                    x += xs[j];
                }
            }
            x
        };

        for &(j, si) in sub_idx.iter() {
            // apply lp_i to each of the macs
            macs.push(
                bit_idx
                    .iter()
                    .map(|&bi| {
                        let el = bi / bsize;
                        let idx = bi % bsize;
                        fis(&self.macs[si][el * bsize..(el + 1) * bsize], my_idx, idx)
                    })
                    .collect(),
            );

            // apply lp_j to each of the keys for j != p
            keys.push(
                bit_idx
                    .iter()
                    .map(|&bi| {
                        let el = bi / bsize;
                        let idx = bi % bsize;
                        fis(&self.keys[si][el * bsize..(el + 1) * bsize], j, idx)
                    })
                    .collect(),
            );
        }

        Abits { bits, macs, keys }
    }

    fn convert_pre(
        my_point: &F,
        all_points: &[F],
        sub_points: &[F],
    ) -> (Vec<(usize, usize)>, F, Vec<Vec<Vec<bool>>>) {
        let my_idx = sub_points.iter().position(|x| x == my_point).unwrap();
        let mut sub_idx: Vec<(usize, usize)> = Vec::new();
        //assert!("all_points, sub_points in same order")
        let mut j = 0;
        for i in 0..sub_points.len() {
            while all_points[j] != sub_points[i] {
                j += 1;
            }
            // skip our point
            if i != my_idx {
                // adjust indices after ours
                let idx = if i > my_idx { j - 1 } else { j };
                sub_idx.push((i, idx));
            }
        }

        // eval at 0
        let lps: Vec<_> = sub_points
            .iter()
            .map(|p| lagrange_poly(sub_points, p, |x| *x))
            .collect();

        let lp_i = lps[my_idx].clone();

        let bsize = F::BIT_SIZE;

        // for FF_{2^k} calculate X^0 .. X^{k-1}
        let mut x = F::one();
        let two: F = F::gen();
        let mut pows = Vec::with_capacity(bsize);
        pows.push(x);

        for _ in 1..(bsize) {
            x *= two;
            pows.push(x);
        }

        // for each coefficient, for each input bit calculate the effect on inclusion in output bit
        let lp_bits: Vec<Vec<Vec<bool>>> = lps
            .iter()
            .map(|l| {
                pows.iter()
                    .map(|p| {
                        let lp = l.clone() * p;
                        let mut bits = vec![false; bsize];
                        lp.to_bits(&mut bits);
                        bits
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        (sub_idx, lp_i, lp_bits)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        ff2_128::FF2_128,
        field::RandElement,
        polynomial::{FixedPolynomial, Polynomial},
    };

    #[test]
    fn test_tabit_conversion() {
        let mut rng = rand::thread_rng();
        let d1 = FF2_128::rand(&mut rng);
        let d2 = FF2_128::rand(&mut rng);

        let poly = FixedPolynomial::rand_polynomial(&mut rng, 1);

        let p1 = FF2_128::rand(&mut rng);
        let p2 = FF2_128::rand(&mut rng);

        let s1 = poly.evaluate(&p1);
        let s2 = poly.evaluate(&p2);

        let mut s0_bits = vec![false; 128];
        let mut s1_bits = vec![false; 128];
        let mut s2_bits = vec![false; 128];

        poly.eval_zero().to_bits(&mut s0_bits);
        s1.to_bits(&mut s1_bits);
        s2.to_bits(&mut s2_bits);

        let ks_1: Vec<_> = (0..128).map(|_| FF2_128::rand(&mut rng)).collect();
        let ks_2: Vec<_> = (0..128).map(|_| FF2_128::rand(&mut rng)).collect();

        let ms_1: Vec<_> = ks_2
            .iter()
            .zip(s1_bits.iter())
            .map(|(k, b)| {
                let mut m = *k;
                if *b {
                    m += d2;
                }
                m
            })
            .collect();

        let ms_2: Vec<_> = ks_1
            .iter()
            .zip(s2_bits.iter())
            .map(|(k, b)| {
                let mut m = *k;
                if *b {
                    m += d1;
                }
                m
            })
            .collect();

        let tbits_1 = ThreshAbits {
            nbits: 128,
            shares: vec![s1.clone()],
            macs: vec![ms_1],
            keys: vec![ks_1],
        };

        let tbits_2 = ThreshAbits {
            nbits: 128,
            shares: vec![s2.clone()],
            macs: vec![ms_2],
            keys: vec![ks_2],
        };

        let bit_idxs: Vec<_> = (0..128).collect();
        let points = [p1.clone(), p2.clone()];
        let as_1 = tbits_1.convert(&p1, &points, &points, &bit_idxs);
        let as_2 = tbits_2.convert(&p2, &points, &points, &bit_idxs);

        for k in 0..128 {
            assert_eq!(s0_bits[k], as_1.bits[k] ^ as_2.bits[k]);
        }
    }
}
