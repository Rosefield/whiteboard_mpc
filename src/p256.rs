use std::ops::{Neg, Add, AddAssign, Mul, Sub, SubAssign};


use crate::{
    ecgroup::{EcGroup, ReduceFromInteger},
    field::{ToFromBytes, RandElement, ConstInt, Field, ExpGroup},
    field_macros::{impl_all_group},
};

use p256::{FieldBytes, Scalar, ProjectivePoint, FieldElement};
use elliptic_curve::{
    group::{Group, GroupEncoding, ff::{Field as ECField, PrimeField}},
    point::AffineCoordinates,
    bigint::{NonZero, U256},
    scalar::FromUintUnchecked,
    };
use rand::Rng;

pub type P256Scalar = Scalar;
pub type P256Field = FieldElement;
pub type P256Point = ProjectivePoint;

#[derive(Debug, Clone, PartialEq)]
pub struct P256(P256Point);

impl P256 {
    pub fn neg(self) -> Self {
        Self(self.0.neg())
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.0 += other.0;
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.0 -= other.0;
    }
}


impl ToFromBytes for P256 {
    const BYTES: usize = 33;

    fn num_bytes(&self) -> usize {
        Self::BYTES
    }

    fn from_bytes(b: &[u8]) -> Self {
        let p = P256Point::from_bytes(b[..33].into()).unwrap();
        Self(p)
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        let bs = self.0.to_bytes();
        let bs = bs.as_slice();
        b[..bs.len()].copy_from_slice(bs);
        bs.len()
    }
}

impl From<u64> for P256 {
    fn from(_e: u64) -> Self {
        unimplemented!()
    }
}

impl ConstInt for P256 {
    fn zero() -> Self {
        Self(P256Point::IDENTITY)
    }
    fn one() -> Self {
        Self(P256Point::GENERATOR)
    }
    fn is_zero(&self) -> bool {
        self.0 == P256Point::IDENTITY
    }
}

impl_all_group!(P256);

impl RandElement for P256 {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let p = P256Point::random(rng);
        Self(p)
    }
}

impl ExpGroup<P256Scalar> for P256 {
    fn exp(&self, e: &P256Scalar) -> Self {
        Self(self.0.mul(e))
    }
}

impl ToFromBytes for P256Scalar {
    const BYTES: usize = 32;

    fn num_bytes(&self) -> usize {
        Self::BYTES
    }

    fn from_bytes(b: &[u8]) -> Self {
        let array: &FieldBytes = b[..32].into();
        Self::from_repr(*array).unwrap()
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        let bs = self.to_bytes();
        let bs = bs.as_slice();
        b[..bs.len()].copy_from_slice(bs);
        bs.len()
    }
}


impl ConstInt for P256Scalar {
    fn zero() -> Self {
        Self::ZERO
    }
    fn one() -> Self {
        Self::ONE
    }
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

}


impl Field for P256Scalar {
    fn gen() -> Self {
        Self::MULTIPLICATIVE_GENERATOR
    }

    fn inv(&self) -> Option<Self> {
        self.invert().into_option()
    }

}

impl RandElement for P256Scalar {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        <P256Scalar as ECField>::random(rng)
    }
}

impl ReduceFromInteger for P256Scalar {
    fn reduce(bs: &[u8]) -> Self {
        assert_eq!(bs.len(), 32);
        let int = U256::from_be_slice(&bs[..32]);

        const ORDER: U256 = U256::from_be_hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
        let nz = NonZero::new(ORDER).unwrap();

        let reduced = int.rem(&nz);

        P256Scalar::from_uint_unchecked(reduced)
    }
}


impl ToFromBytes for P256Field {
    const BYTES: usize = 32;

    fn num_bytes(&self) -> usize {
        Self::BYTES
    }

    fn from_bytes(b: &[u8]) -> Self {
        let array: &FieldBytes = b[..32].into();
        Self::from_repr(*array).unwrap()
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        let bs = self.to_repr();
        let bs = bs.as_slice();
        b[..bs.len()].copy_from_slice(bs);
        bs.len()
    }
}


impl ConstInt for P256Field {
    fn zero() -> Self {
        Self::ZERO
    }
    fn one() -> Self {
        Self::ONE
    }
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

}

impl Field for P256Field {
    fn gen() -> Self {
        Self::MULTIPLICATIVE_GENERATOR
    }

    fn inv(&self) -> Option<Self> {
        self.invert().into_option()
    }

}

impl RandElement for P256Field {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        P256Field::random(rng)
    }
}

impl EcGroup for P256 {
    type Ford = P256Scalar;
    type Fq = P256Field;

    fn x_reduced(&self) -> Self::Ford {

        let affine = self.0.to_affine();
        let xbytes = affine.x();

        P256Scalar::reduce(&xbytes)
    }


    fn x(&self) -> Self::Fq {
        let affine = self.0.to_affine();
        // stupid private internal fields
        Self::Fq::from_repr(affine.x()).unwrap()
    }

    fn y(&self) -> Self::Fq {
        let _affine = self.0.to_affine();
        // stupid private internal fields, and this version of the library doesn't even expose the
        // y() method
        unimplemented!() //Self::Fq::from_repr(affine.y()).unwrap()
    }
}





