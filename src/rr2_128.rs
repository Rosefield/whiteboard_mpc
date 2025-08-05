use core::arch::x86_64::*;

use std::ops::{Neg, Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    field_macros::impl_all_ring,
    field::{
    ToFromBytes, ConstInt, RandElement, 
    }
};

use rand::Rng;

/// An element in the ring {F_2}^128
#[derive(Copy, Clone, Debug)]
pub struct RR2_128 {
    i: __m128i,
}

impl PartialEq for RR2_128 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let t = _mm_xor_si128(self.i, other.i);
            let r = _mm_test_all_zeros(t, t);
            return r == 1;
        }
    }
}

impl RR2_128 {
    pub fn new(high: u64, low: u64) -> Self {
        unsafe {
            return Self {
                i: _mm_set_epi64x(high as i64, low as i64),
            };
        }
    }

    pub fn neg(mut self) -> Self {
        self += Self::one();
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        unsafe { self.i = _mm_xor_si128(self.i, other.i) }
    }

    pub fn sub_assign(&mut self, other: &Self) {
        unsafe { self.i = _mm_xor_si128(self.i, other.i) }
    }

    pub fn mul_assign(&mut self, other: &Self) {
        unsafe {
            self.i = _mm_and_si128(self.i, other.i);
        }
    }
}

impl_all_ring!(RR2_128);

impl From<u64> for RR2_128 {
    fn from(other: u64) -> RR2_128 {
        return RR2_128::new(0, other);
    }
}

impl ToFromBytes for RR2_128 {
    const BYTES: usize = 16;
    fn num_bytes(&self) -> usize {
        return 16;
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= 16);

        let bytes: &[u8; 16] = unsafe { std::mem::transmute(self) };
        b[..16].copy_from_slice(bytes);

        return 16;
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= 16);
        let hi = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let lo = u64::from_le_bytes(b[8..16].try_into().unwrap());

        Self::new(hi, lo)
    }
}

impl ConstInt for RR2_128 {

    fn zero() -> Self {
        return Self::new(0, 0);
    }
    fn one() -> Self {
        return Self::new(u64::MAX, u64::MAX);
    }
    fn is_zero(&self) -> bool {
        unsafe {
            return _mm_test_all_zeros(self.i, self.i) == 1;
        }
    }
}

impl RandElement for RR2_128 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut r = [0u64; 2];
        rng.fill(&mut r);

        return Self::new(r[0], r[1]);
    }
}

#[cfg(test)]
mod tests {}
