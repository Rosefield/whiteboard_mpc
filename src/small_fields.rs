
use std::ops::{Neg, Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    field_macros::{impl_all_ring, impl_module},
    field::{
        ToFromBytes, ConstInt, RandElement, 
        Extension, Module,
    },
    linalg::Vector
};
use rand::Rng;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FF2
{
    i: bool
}

impl FF2
{
    pub fn new(e: bool) -> Self {
        Self {
            i: e
        }
    }

    pub fn neg(mut self) -> Self {
        self.i ^= true;
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.i ^= other.i;
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.i ^= other.i;
    }

    pub fn mul_assign(&mut self, other: &Self) {
        self.i &= other.i;
    }
}
impl_all_ring!(FF2);

impl From<u64> for FF2
{
    fn from(other: u64) -> Self {
        Self::new(other != 0)
    }
}

impl ToFromBytes for FF2 {
    const BYTES: usize = 1;

    fn num_bytes(&self) -> usize {
        return Self::BYTES;
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= Self::BYTES);
        b[0] = self.i as u8;
        return Self::BYTES;
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= Self::BYTES);
        Self {
            i: b[0] == 1
        }
    }
}

impl ConstInt for FF2
{
    fn zero() -> Self {
        return Self::new(false);
    }
    fn one() -> Self {
        return Self::new(true);
    }
    fn is_zero(&self) -> bool {
        !self.i
    }
}

impl RandElement for FF2 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let b = rng.gen();

        return Self::new(b);
    }
}


// Use vector internal storage to deal with annoying [; N.div_ceil()] bounds
// and also future stack size issues
#[derive(Clone, Debug)]
pub struct Rr2Ell<const N: usize>
{
    i: Vec<u8>
}

/*
#[derive(Copy, Clone, Debug)]
pub struct Rr2Ell<const N: usize>
    where [u8; N.div_ceil(8)]: Sized
{
    i: [u8; N.div_ceil(8)],
}
*/

impl<const N: usize> AsRef<[u8]> for Rr2Ell<N>
{
    fn as_ref(&self) -> &[u8] {
        &self.i
    }
}

impl<const N: usize> Rr2Ell<N>
{

    pub fn as_vector(&self) -> Vector<FF2, N> {
        // TODO: can the Vector struct use this as the internal storage instead of FWrap?
        std::array::from_fn(|i| {
            let word = i / 8;
            let idx = i % 8;

            let mask = 0x01u8 << idx;

            FF2::new((self.i[word] & mask) == mask)
        }).into()
    }

    pub fn broadcast(bit: bool) -> Self {
        let val = if bit { 0b11111111 } else { 0 };
        Self {
            i: vec![val; N.div_ceil(8)]
        }
    }

    pub fn neg(mut self) -> Self {
        self.i.iter_mut().for_each(|x| {*x ^= 0xFF;});
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| *s ^= o);
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| *s ^= o);
    }

    pub fn mul_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| *s &= o);
    }
}

impl<const N: usize> From<Rr2Ell<N>> for Vector<FF2, N> {
    fn from(e: Rr2Ell<N>) -> Self {
        e.as_vector()
    }
}

// \FF_2[X] \ <x^2 + x + 1>
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FF4
{
    i: u8
}

/// Calculates the multiplication of (four) FF_4 elements componentwise in each of s,o
fn ff4_mul(s: u8, o: u8) -> u8 {
    // x^2 = x + 1
    // 00 x __ -> 00
    // 01 x ab -> ab
    // 10 x ab -> (a+b)|a
    // 11 x ab -> b|(a+b)
    // cd x ab -> (cb + da + ca)|(db + ca)
    // s & o = ca|db

    // calculate a+b
    let mask = 0b10101010;

    // bottom bit is left nonsense
    /*
    let sum_top = |x: u8| {
        x ^ (x << 1)
    };
    let ab = sum_top(o);
    let cd = sum_top(s);
    */
    let swap = |x:u8| {
        ((x & mask) >> 1) ^ ((x << 1) & mask)
    };
    // definitely not the most optimal but I've been mentally trying to optimize for
    // too long 
    let ca_db = s & o;
    let cb_da = s & swap(o);
    // ca | ca+db ^ cb + da | 0
    ((ca_db&mask) >> 1 ^ ca_db) ^ ((cb_da ^ cb_da << 1) & mask)
}

impl FF4
{
    pub fn new(e: u8) -> Self {
        assert!(e < 4);
        Self {
            i: e
        }
    }

    pub fn neg(mut self) -> Self {
        self.i ^= 0xFF;
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.i ^= other.i;
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.i ^= other.i;
    }

    pub fn mul_assign(&mut self, other: &Self) {
        self.i = ff4_mul(self.i, other.i);
    }
}

impl_all_ring!(FF4);

impl From<u64> for FF4
{
    fn from(other: u64) -> Self {
        Self::new(other as u8)
    }
}

impl ToFromBytes for FF4 {
    const BYTES: usize = 1;

    fn num_bytes(&self) -> usize {
        return Self::BYTES;
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= Self::BYTES);
        b[0] = self.i;
        return Self::BYTES;
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= Self::BYTES);
        Self {
            i: b[0]
        }
    }
}

impl ConstInt for FF4
{
    fn zero() -> Self {
        return Self::new(0);
    }
    fn one() -> Self {
        return Self::new(1);
    }
    fn is_zero(&self) -> bool {
        self.i == 0
    }
}

impl RandElement for FF4 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut r = [0u8; 1];
        rng.fill(&mut r);

        return Self::new(r[0] & 0x03);
    }
}

impl Extension<FF2> for FF4
{
    fn embed(e: &FF2) -> Self {
        Self {
            i: e.i as u8
        }
    }
}

// \FF_4^\ell
#[derive(Clone, Debug)]
pub struct Rr4Ell<const N: usize>
{
    i: Vec<u8>
}
/*
#[derive(Copy, Clone, Debug)]
pub struct Rr4Ell<const N: usize>
    where [u8; N.div_ceil(4)]: Sized
{
    i: [u8; N.div_ceil(4)],
}
*/

impl<const N: usize> AsRef<[u8]> for Rr4Ell<N>
//    where [u8; N.div_ceil(4)]: Sized
{
    fn as_ref(&self) -> &[u8] {
        &self.i
    }
}

impl<const N: usize> Rr4Ell<N>
//    where [u8; N.div_ceil(4)]: Sized
{
    pub fn as_vector(&self) -> Vector<FF4, N> {
        // TODO: can the Vector struct use this as the internal storage instead of FWrap?
        std::array::from_fn(|i| {
            let word = i / 4;
            let idx = i % 4;

            let mask = 0x03u8;

            FF4::new((self.i[word] >> (2*idx)) & mask)
        }).into()
    }

    fn broadcast_1(e: FF4) -> u8{
        let e = e.i;
        let val = e << 6 | e << 4 | e << 2 | e;
        val
    }
    pub fn broadcast(e: u8) -> Self {
        let val = Self::broadcast_1(FF4::new(e));
        Self {
            i: vec![val; N.div_ceil(4)]
        }

    }

    pub fn neg(mut self) -> Self {
        self.i.iter_mut().for_each(|x| {*x ^= 0xFF;});
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| *s ^= o);
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| *s ^= o);
    }

    pub fn mul_assign(&mut self, other: &Self) {
        self.i.iter_mut().zip(other.i.iter())
            .for_each(|(s, o)| {
                *s = ff4_mul(*s, *o);
            });
    }

    pub fn add_assign_ff4(&mut self, other: &FF4) {
        let o = Self::broadcast_1(*other);
        self.i.iter_mut()
            .for_each(|s| *s ^= o);
    }

    pub fn sub_assign_ff4(&mut self, other: &FF4) {
        let o = Self::broadcast_1(*other);
        self.i.iter_mut()
            .for_each(|s| *s ^= o);
    }

    pub fn mul_assign_ff4(&mut self, other: &FF4) {
        let o = Self::broadcast_1(*other);
        self.i.iter_mut()
            .for_each(|s| {
                *s = ff4_mul(*s, o);
            });
    }
}


impl<const N: usize> From<Rr4Ell<N>> for Vector<FF4, N> {
    fn from(e: Rr4Ell<N>) -> Self {
        e.as_vector()
    }
}


impl<const N: usize> Extension<Rr2Ell<N>> for Rr4Ell<N>
//    where [u8; N.div_ceil(8)]: Sized,
//          [u8; N.div_ceil(4)]: Sized
{
    fn embed(e: &Rr2Ell<N>) -> Self {
        let spread = |x: u8| {
            // 0b____abcd -> 0b0a0b0c0d
            (x & 0b00000001) ^ ((x << 1) & 0b00000100) ^ ((x << 2) & 0b00010000) ^ ((x << 3) & 0b01000000)
        };

        let mut new = vec![0u8; N.div_ceil(4)];

        assert_eq!(0b01010101, spread(0b00001111));

        for i in 0..N.div_ceil(4) {
            //odd indices of the new vector use the top half of the old element
            let o = if (i & 1) == 1 { e.i[i/2] >> 4 } else { e.i[i/2] };
            new[i] = spread(o);
        }

        Self {
            i: new
        }
    }
}

macro_rules! impl_all_sized {
    ($type:ty, $constraint:expr) => {

        impl<const N: usize> PartialEq for $type
        //    where [u8; $constraint]: Sized 
        {
            fn eq(&self, other: &Self) -> bool {
                self.i == other.i
            }
        }

        //impl_all_ring!($type, {const N: usize}; {[u8; $constraint]: Sized});
        impl_all_ring!($type, {const N: usize}; {(): });

        impl<const N: usize> RandElement for $type
        //    where [u8; $constraint]: Sized
        {
            fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
                let mut bytes = vec![0u8; $constraint];
                rng.fill(&mut bytes[..]);

                Self {
                    i: bytes
                }
            }
        }
    };
}

impl_all_sized!(Rr2Ell<N>, N.div_ceil(8));


impl<const N: usize> From<u64> for Rr2Ell<N> 
//    where [u8; N.div_ceil(8)]: Sized 
{
    fn from(other: u64) -> Self {
        Self::broadcast(other == 1)
    }
}

impl<const N: usize> ToFromBytes for Rr2Ell<N>
//    where [u8; N.div_ceil(8)]: Sized 
{
    const BYTES: usize = usize::div_ceil(N, 8);

    fn num_bytes(&self) -> usize {
        return Self::BYTES;
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= Self::BYTES);

        b[..Self::BYTES].copy_from_slice(&self.i);
        return Self::BYTES;
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= Self::BYTES);
        let mut bytes = vec![0; N.div_ceil(8)];
        bytes.copy_from_slice(&b[..Self::BYTES]);
        Self {
            i: bytes
        }
    }
}

impl<const N: usize> ConstInt for Rr2Ell<N>
//    where [u8; N.div_ceil(8)]: Sized
{
    fn zero() -> Self {
        return Self::broadcast(false);
    }
    fn one() -> Self {
        return Self::broadcast(true);
    }
    fn is_zero(&self) -> bool {
        self.i.iter().all(|x| *x == 0)
    }
}

impl_all_sized!(Rr4Ell<N>, N.div_ceil(4));
//impl_module!(Rr4Ell<N>, FF4, ff4, N, {const N: usize}; {[u8; N.div_ceil(4)]: Sized});
impl_module!(Rr4Ell<N>, FF4, ff4, N, {const N: usize}; {(): });

impl<const N: usize> From<u64> for Rr4Ell<N> 
//    where [u8; N.div_ceil(4)]: Sized 
{
    fn from(other: u64) -> Self {
        Self::broadcast(other as u8)
    }
}
impl<const N: usize> ToFromBytes for Rr4Ell<N>
//    where [u8; N.div_ceil(4)]: Sized
{
    const BYTES: usize = usize::div_ceil(N, 4);

    fn num_bytes(&self) -> usize {
        return Self::BYTES;
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= Self::BYTES);

        b[..Self::BYTES].copy_from_slice(&self.i);
        return Self::BYTES;
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= Self::BYTES);
        let mut bytes = vec![0; Self::BYTES];
        bytes.copy_from_slice(&b[..Self::BYTES]);
        Self {
            i: bytes
        }
    }
}

impl<const N: usize> ConstInt for Rr4Ell<N>
//    where [u8; N.div_ceil(4)]: Sized
{
    fn zero() -> Self {
        return Self::broadcast(0);
    }
    fn one() -> Self {
        return Self::broadcast(1);
    }
    fn is_zero(&self) -> bool {
        self.i.iter().all(|x| *x == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rr2_arith() {
        let zeros = Rr2Ell::<8>::zero();
        let ones = Rr2Ell::<8>::one();
        assert_eq!(ones.num_bytes(), 1);
        assert_eq!(ones.clone()+&ones, zeros);
        assert_eq!(ones.clone()*&ones, ones);

    }

    #[test]
    fn test_rr4_arith() {
        let zeros = Rr4Ell::<8>::zero();
        let ones = Rr4Ell::<8>::one();
        let twos = Rr4Ell::broadcast(2);
        let threes = Rr4Ell::broadcast(3);
        assert_eq!(ones.num_bytes(), 2);
        assert_eq!(ones.clone()+&ones, zeros);
        assert_eq!(ones.clone()*&ones, ones);
        assert_eq!(ones.clone()*&twos, twos);
        assert_eq!(twos.clone()*&twos, threes);
        assert_eq!(twos.clone()*&threes, ones);
    }

    #[test]
    fn test_rr4_extension() {
        // test conversion of FF2 vec to FF4 vec
        let r2ones = Rr2Ell::<16>::one();
        let r4ones = Rr4Ell::embed(&r2ones);
        assert_eq!(r4ones, Rr4Ell::one());

        // test conversion to non-compact vector
        let r4ones_vec = r4ones.as_vector();
        assert!(r4ones_vec.into_iter().all(|x| x == FF4::one()));
    }


}
