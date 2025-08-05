use core::arch::x86_64::*;

use std::ops::{Neg, Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    field_macros::impl_all_ring,
    field::{ToFromBytes, ConstInt, Field, ExpGroup, RandElement, exp_sliding_window},
};
use rand::Rng;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An element in the field F_2[x] / (x^128 + x^7 + x^2 + x + 1)
#[derive(Copy, Clone)]
pub struct FF2_128 {
    i: __m128i,
}

impl Serialize for FF2_128 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = [0; 16];
        self.to_bytes(&mut bytes);
        bytes.serialize(serializer)
    }
}

struct FF2Visitor;

impl<'d> Visitor<'d> for FF2Visitor {
    type Value = FF2_128;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a 16-byte array")
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        if v.len() == 16 {
            Ok(FF2_128::from_bytes(v))
        } else {
            Err(E::custom(format!("byte array not len 16: {}", v.len())))
        }
    }

    fn visit_seq<A: de::SeqAccess<'d>>(self, mut v: A) -> Result<Self::Value, A::Error> {
        let mut bytes = [0; 16];
        for i in 0..16 {
            if let Some(x) = v.next_element()? {
                bytes[i] = x;
            } else {
                return Err(<A::Error as de::Error>::custom(format!(
                    "expected to find {}'th element in array",
                    i
                )));
            }
        }
        Ok(FF2_128::from_bytes(&bytes))
    }
}

impl<'d> Deserialize<'d> for FF2_128 {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(FF2Visitor)
    }
}

impl std::fmt::Debug for FF2_128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limbs: &[u64; 2] = unsafe { std::mem::transmute(self) };

        write!(f, "FF2_128({:#x}, {:#x})", limbs[0], limbs[1])
    }
}

impl PartialEq for FF2_128 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let t = _mm_xor_si128(self.i, other.i);
            let r = _mm_test_all_zeros(t, t);
            return r == 1;
        }
    }
}

impl FF2_128 {
    pub fn new(high: u64, low: u64) -> Self {
        unsafe {
            return Self {
                i: _mm_set_epi64x(high as i64, low as i64),
            };
        }
    }

    pub fn neg(self) -> Self {
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        unsafe { self.i = _mm_xor_si128(self.i, other.i) }
    }

    pub fn sub_assign(&mut self, other: &Self) {
        unsafe { self.i = _mm_xor_si128(self.i, other.i) }
    }

    pub fn mul_assign(&mut self, other: &Self) {
        // karatsuba-style mult
        // This is me attempting to use AVX-like instructions, but of course I haven't
        // benchmarked anything nor have much experience so this is probably not optimal.
        unsafe {
            let n2 = _mm_clmulepi64_si128(self.i, other.i, 0x11);
            let n0 = _mm_clmulepi64_si128(self.i, other.i, 0x00);

            /*
            let n11 = _mm_clmulepi64_si128(self.i, other.i, 0x10);
            let n12 = _mm_clmulepi64_si128(self.i, other.i, 0x01);
            let n1 = _mm_xor_si128(n11, n12);
            */

            // Karatsuba style
            // Tentative benchmarks has this ~30% faster than the naive version above
            let mut n1 = n2.clone();
            // Want to create a register that is (self[1] ^ self[0]) || (other[1] ^ other[0])
            let l = _mm_unpackhi_epi64(other.i, self.i);
            let r = _mm_unpacklo_epi64(other.i, self.i);
            let m = _mm_xor_si128(l, r);
            // (self[1] ^ self[0]) * (other[1] ^ other[0])
            let mid = _mm_clmulepi64_si128(m, m, 0x10);
            n1 = _mm_xor_si128(n1, mid);
            n1 = _mm_xor_si128(n1, n0);

            // need to calculate n2 * 2^128 + n1*2^64 + n0
            let t1 = _mm_xor_si128(n2, _mm_srli_si128(n1, 8));
            let t0 = _mm_xor_si128(n0, _mm_slli_si128(n1, 8));

            // reduce [t1 t0] by the polynomial
            // It would be too easy if you could just shift the whole vector by a number of bits
            let t1h = _mm_extract_epi64(t1, 1);
            let t1l = _mm_extract_epi64(t1, 0);
            fn mul_high(y: i64) -> i64 {
                let x = y as u64;
                let z = (x >> 57) ^ (x >> 62) ^ (x >> 63);
                z as i64
            }
            fn mul_low(y: i64) -> i64 {
                let x = y as u64;
                let z = x ^ (x << 1) ^ (x << 2) ^ (x << 7);
                z as i64
            }
            let t1l = t1l ^ mul_high(t1h);
            let mh = mul_low(t1h) ^ mul_high(t1l);
            let ml = mul_low(t1l);
            let mask = _mm_set_epi64x(mh, ml);

            self.i = _mm_xor_si128(t0, mask);
        }
    }

    pub fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // Calculate a^{-1} = a^{2^128 - 2}
        // or exponentiate with the bit pattern 1111...1110
        // this is done in 159 multiplications in squaring windows of size 5
        // using size-4 windows requires 161, and size-6, 158.
        // it is unclear if the code-size difference is worth the 2/-1 multiplications
        // That all being said, there is probably a better algorithm that calculates
        // the inverse directly.
        let a = self.clone();
        let a2_1 = a * a;
        let a2_2 = a2_1 * a2_1;
        let a2_3 = a2_2 * a2_2;
        let a2_4 = a2_3 * a2_3;
        let a3 = a2_1 * a;
        let a7 = a2_2 * a3;
        let a15 = a2_3 * a7;
        let a31 = a2_4 * a15;

        // 11
        let mut i = a3;
        for _ in 0..25 {
            // square 5 times to get i || 00000, then multiply by 11111 to get i || 11111
            i *= i;
            i *= i;
            i *= i;
            i *= i;
            i *= i;
            i *= a31;
        }

        // i is now a^{2^{127} -1}
        // then i^{2} = a^{2^128 - 2}
        // as desired
        i *= i;

        Some(i)
    }
}
impl_all_ring!(FF2_128);

impl From<u64> for FF2_128 {
    fn from(other: u64) -> FF2_128 {
        return FF2_128::new(0, other);
    }
}

impl ToFromBytes for FF2_128 {
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
        let lo = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let hi = u64::from_le_bytes(b[8..16].try_into().unwrap());

        Self::new(hi, lo)
    }
}

impl ConstInt for FF2_128 {
    fn zero() -> Self {
        return Self::new(0, 0);
    }

    fn one() -> Self {
        return Self::new(0, 1);
    }

    fn is_zero(&self) -> bool {
        unsafe {
            return _mm_test_all_zeros(self.i, self.i) == 1;
        }
    }
}

impl Field for FF2_128 {
    fn gen() -> Self {
        return Self::new(0, 2);
    }

    fn inv(&self) -> Option<Self> {
        return FF2_128::inv(self);
    }
}

impl RandElement for FF2_128 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut r = [0u64; 2];
        rng.fill(&mut r);

        return Self::new(r[0], r[1]);
    }
}

impl ExpGroup<u128> for FF2_128 {
    fn exp(&self, e: &u128) -> Self {
        if *e == 0 {
            return Self::one();
        }
        let bytes = e.to_be_bytes();
        exp_sliding_window(self, &bytes)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use rand;
    use test::Bencher;

    #[bench]
    fn bench_mul(b: &mut Bencher) {
        let x = FF2_128::new(8405355093950566387, 13433973052803411392);
        let y = FF2_128::new(10655597574255505399, 8855082111849768422);
        b.iter(|| {
            let n = test::black_box(10000);
            (0..n).fold(FF2_128::zero(), |old, _| old + (x * y))
        });
    }

    #[bench]
    fn bench_inv(b: &mut Bencher) {
        let x = FF2_128::new(8405355093950566387, 13433973052803411392);
        b.iter(|| {
            let n = test::black_box(100);
            (0..n).fold(FF2_128::zero(), |old, _| old + x.inv().unwrap())
        });
    }

    #[test]
    fn test_mul() {
        // couple of random multiplications straight out of SAGE
        {
            let x = FF2_128::new(8405355093950566387, 13433973052803411392);
            let y = FF2_128::new(10655597574255505399, 8855082111849768422);
            let z = FF2_128::new(6112988964148991676, 11292369341253676568);
            dbg!(x * y, z);
            assert!(x * y == z);
        }

        {
            let x = FF2_128::new(13582814881232980082, 13533280003910603092);
            let y = FF2_128::new(6249718029340932425, 965851581480529057);
            let z = FF2_128::new(14444192429104062601, 15332326950068355486);
            dbg!(x * y, z);
            assert!(x * y == z);
        }

        {
            let x = FF2_128::new(8127387034772781855, 2184651014883997490);
            let y = FF2_128::new(10471407400604350249, 1236872039594208442);
            let z = FF2_128::new(16833883055683103281, 6805129979863820364);
            dbg!(x * y, z);
            assert!(x * y == z);
        }

        {
            let x = FF2_128::new(4247814983155021246, 8938114806903946159);
            let y = FF2_128::new(17254603535503132576, 6701304688585168118);
            let z = FF2_128::new(15706757134225084095, 3431301149681691759);
            assert!(x * y == z);
        }

        {
            let x = FF2_128::new(12379867753862226045, 11890716767991229637);
            let y = FF2_128::new(17211587100699079571, 15335621025362235604);
            let z = FF2_128::new(13400512513552687770, 6076968003818681967);
            assert!(x * y == z);
        }

        {
            let x = FF2_128::new(8065553106679331374, 9796440155182912654);
            let y = FF2_128::new(1733266973438645675, 15142776612367171862);
            let z = FF2_128::new(14295794314133301526, 3440670992175780843);
            assert!(x * y == z);
        }
    }

    #[test]
    fn test_inv() {
        {
            let x = FF2_128::new(16542412965618483126, 10523052148348452803);
            let y = FF2_128::new(5743251910337479519, 12037164058816094527);
            assert!(x.inv() == Some(y));
        }

        {
            let x = FF2_128::new(15344043119869342270, 13536107431080300635);
            let y = FF2_128::new(9463324440232948964, 16488160259435988851);
            assert!(x.inv() == Some(y));
        }

        {
            let x = FF2_128::new(4142684366113569777, 12177262448850725360);
            let y = FF2_128::new(405072602358188721, 3809638000420065597);
            assert!(x.inv() == Some(y));
        }
    }

    #[test]
    fn test_serialize_round_trip() {
        let mut bytes = [0; 16];
        let mut rng = rand::thread_rng();
        let val = FF2_128::rand(&mut rng);

        val.to_bytes(&mut bytes);

        let new_val = FF2_128::from_bytes(&bytes);
        assert_eq!(val, new_val);
    }
}
