use crate::field::{ExpGroup, Field, Group};

pub trait ReduceFromInteger {
    fn reduce(bs: &[u8]) -> Self;
}

pub trait EcGroup: Group + ExpGroup<Self::Ford> {
    type Ford: Field + ReduceFromInteger;
    type Fq: Field;

    fn x_reduced(&self) -> Self::Ford;
    fn x(&self) -> Self::Fq;
    fn y(&self) -> Self::Fq;
}
