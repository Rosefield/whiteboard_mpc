use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Neg, Add, AddAssign, Mul, MulAssign, Sub, SubAssign, BitXor, BitXorAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign},
};

use crate::field_macros::{impl_arith, impl_unary, impl_sum_prod, impl_module};

use rand::Rng;

/// Allows sampling an element in the set
pub trait RandElement: Clone {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

impl<const N:usize> RandElement for [u8; N] {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut s = [0; N];
        rng.fill(s.as_mut_slice());
        s
    }
}

impl<T: RandElement, const N: usize> RandElement for [T; N] {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        std::array::from_fn(|_| T::rand(rng))
    }
}

// My own trait for now, probbly want to find a good standard one to use instead
// in the future
// maybe use std::io::{Read, Write}?
pub trait ToFromBytes {
    const BYTES: usize;
    fn num_bytes(&self) -> usize;
    fn to_bytes(&self, b: &mut [u8]) -> usize;
    fn from_bytes(_: &[u8]) -> Self;

    /*
    fn to_reader<'a>(&'a self) -> impl Read;
    fn from_writer<W: Write(w: W) -> Option<Self>;
    */

}

impl<const N:usize> ToFromBytes for [u8; N] {
    const BYTES: usize = N;
    fn num_bytes(&self) -> usize { N }
    fn to_bytes(&self, b: &mut [u8]) -> usize {
        b[..N].copy_from_slice(self);
        N
    }
    fn from_bytes(b: &[u8]) -> Self {
        b[..N].try_into().unwrap()
    }
}

impl<T: ToFromBytes, const N:usize> ToFromBytes for [T; N] {
    const BYTES: usize = T::BYTES*N;
    fn num_bytes(&self) -> usize { T::BYTES*N }
    fn to_bytes(&self, b: &mut [u8]) -> usize {
        let size = T::BYTES;
        for i in 0..N {
            self[i].to_bytes(&mut b[size*i..size*(i+1)]);
        }
        Self::BYTES
    }
    fn from_bytes(b: &[u8]) -> Self {
        let size = T::BYTES;
        std::array::from_fn(|i| T::from_bytes(&b[i*size..]))
    }
}

impl ToFromBytes for u128 {
    const BYTES: usize = 16;
    fn num_bytes(&self) -> usize { Self::BYTES }
    fn to_bytes(&self, b: &mut [u8]) -> usize {
        b[..16].copy_from_slice(&self.to_le_bytes());
        Self::BYTES
    }
    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() > 16);
        let bytes = b[..16].try_into().unwrap();
        u128::from_le_bytes(bytes)
    }
}

/// Base trait for an integer
pub trait ConstInt: ToFromBytes + PartialEq + Clone + Send + Debug + Sized + From<u64> {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    //    fn as_usize(&self) -> usize;
    //    fn pow2(_: usize) -> Self;
}

impl ConstInt for u128 {
    fn zero() -> u128 {
        0
    }

    fn one() -> u128 {
        1
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }
}


pub trait ModInt {
    fn mod_add(&self, right: &Self, m: &Self) -> Self;
    fn mod_sub(&self, right: &Self, m: &Self) -> Self;
    fn mod_mul(&self, right: &Self, m: &Self) -> Self;
    fn mod_pow(&self, exp: &Self, m: &Self) -> Self;
    fn mod_inv(&self, m: &Self) -> Option<Self>
    where
        Self: Sized;
}

// It would be nice to have an Add<&Self, Output=Self> for &Self bound as well,
// but I can't seem to figure it out
pub trait Group: ConstInt + 
     Add<Output = Self> + AddAssign + Sum +
    for<'a> Add<&'a Self, Output = Self> + 
    for<'a> AddAssign<&'a Self> + 
    for<'a> Sum<&'a Self> +
    Sub<Output = Self> + SubAssign +
    for<'a> Sub<&'a Self, Output = Self> +
    for<'a> SubAssign<&'a Self> +
    Neg<Output = Self>
{}

impl<T> Group for T
    where T: ConstInt + 
        Add<Output = Self> + AddAssign + Sum +
        for<'a> Add<&'a Self, Output = Self> + 
        for<'a> AddAssign<&'a Self> + 
        for<'a> Sum<&'a Self> +
        Sub<Output = Self> + SubAssign +
        for<'a> Sub<&'a Self, Output = Self> +
        for<'a> SubAssign<&'a Self> +
        Neg<Output = Self>
{}


pub trait ExpGroup<Ford: ConstInt>: Sized {
    fn exp(&self, e: &Ford) -> Self;
}

pub fn exp_sliding_window<R: Ring>(b: &R, be_bytes: &[u8]) -> R {

        assert!(be_bytes.len() > 0);
        // TODO: really this should be variable window as the name states
        // but I'm using the lazy implementation so I don't need
        // to handle variable-width bit reading of the exponent.

        let mut bases: [R; 16] = std::array::from_fn(|_| R::one());
        for i in 1..16 {
            bases[i] = bases[i-1].clone() * b;
        }

        let round = |i: &mut R, e: u8| {
            let mut j = i.clone();
            j = j.clone()*j;
            j = j.clone()*j;
            j = j.clone()*j;
            j = j.clone()*j;
            j *= &bases[(e & 0x0F) as usize];
            *i = j;
        };

        let b0 = be_bytes[0];
        let zs = b0.leading_zeros();

        let mut i = if zs < 4 {
            // b0 >= 16
            let top = b0 >> 4;
            let mut i = bases[(top & 0x0F) as usize].clone();
            let bot = b0 & 0x0F;
            round(&mut i, bot);
            i
        } else {
            let bot = b0 & 0x0F;
            bases[(bot & 0x0F) as usize].clone()
        };

        for b in be_bytes[1..].into_iter() {
            let top = b >> 4;
            round(&mut i, top);

            let bot = b & 0x0F;
            round(&mut i, bot);
        }

        i
}

pub trait Ring: Group +
    Mul<Output = Self> + MulAssign + Product +
    for<'a> Mul<&'a Self, Output = Self> +
    for<'a> MulAssign<&'a Self> +
    for<'a> Product<&'a Self>
{}

impl<T> Ring for T
    where T: Group +
        Mul<Output = Self> + MulAssign + Product +
        for<'a> Mul<&'a Self, Output = Self> +
        for<'a> MulAssign<&'a Self> +
        for<'a> Product<&'a Self>
{}

pub trait Field: Ring {
    fn gen() -> Self;
    fn inv(&self) -> Option<Self>;
}

pub trait Extension<R: Group> {
    fn embed(el: &R) -> Self;
}

pub trait RingExtension<R: Ring>: Extension<R> + Ring {}

impl<R1: Ring, R2: Ring + Extension<R1>> RingExtension<R1> for R2 {}

pub trait Module<R: Ring>: Group + Add<R, Output=Self> + AddAssign<R> + for<'a> AddAssign<&'a R> +
    Sub<R, Output=Self> + SubAssign<R> + for<'a> SubAssign<&'a R> +
    Mul<R, Output=Self> + MulAssign<R> + for<'a> MulAssign<&'a R> {

    const DEGREE: usize;
}

pub trait VectorSpace<F: Field>: Module<F> {}

macro_rules! expr {
    ($x:expr) => {
        $x
    };
}
macro_rules! idx {
    ($t:expr, $idx:tt) => {
        expr!($t.$idx)
    };
}

// Can't impl the arithmetics on tuples/arrays ourselves because of coherence rules
// so just wrap in a struct for convenience
#[repr(transparent)]
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct FWrap<T>(pub T);

macro_rules! impl_unary_tuple {
    (,$($is:tt),+; $($types:ident),+,; $trait:ident, $tf:ident) => {
        impl<$($types: $trait<Output=$types>,)+> $trait for FWrap<($($types,)+)> {
            type Output = Self;
            fn $tf(self) -> Self::Output {
                FWrap(($(
                idx!(self.0, $is).$tf(),)+
                ))
            }
        }
    };
}

macro_rules! impl_arith_assign_tuple {
    (,$($is:tt),+; $($types:ident),+,; $trait:ident, $tf:ident, $traitassign:ident, $taf:ident) => {
        impl<$($types: $traitassign,)+> $traitassign for FWrap<($($types,)+)> {
            fn $taf(&mut self, other: Self) {
                $(
                idx!(self.0, $is).$taf(idx!(other.0, $is));
                )+
            }
        }

        impl<'a, $($types: $traitassign<&'a $types>,)+> $traitassign<&'a Self> for FWrap<($($types,)+)> {
            fn $taf(&mut self, other: &'a Self) {
                $(
                idx!(self.0, $is).$taf(&idx!(other.0, $is));
                )+
            }
        }

        impl_arith!(FWrap<($($types,)+)>, FWrap<($($types,)+)>, $trait, $tf, $traitassign, $taf, {$($types: $traitassign + for<'c> $traitassign<&'c$types>,)+}; {FWrap<($($types,)+)>: $traitassign + Clone + for<'c> $traitassign<&'c Self> });
    };
}

macro_rules! impl_sum_prod_tuple {
    ($($types:ident),+,) => {
        impl_sum_prod!(FWrap<($($types,)+)>, {$($types,)+};
            {FWrap<($($types,)+)>: ConstInt + AddAssign + for<'c> AddAssign<&'c Self> };
            {FWrap<($($types,)+)>: ConstInt + MulAssign + for<'c> MulAssign<&'c Self> });
    };
}

macro_rules! impl_const_int_tuple {
    (,$($is: tt),+; $($types:ident),+,) => {
        impl<$($types: From<u64>,)+> From<u64> for FWrap<($($types,)+)> {
            fn from(other: u64) -> Self {
                return FWrap(($($types::from(other),)+));
            }
        }

        impl<$($types: ToFromBytes,)+> ToFromBytes for FWrap<($($types,)+)> {
            const BYTES: usize = {$($types::BYTES +)+ 0};
            fn num_bytes(&self) -> usize {
                Self::BYTES
            }
            fn to_bytes(&self, b: &mut [u8]) -> usize {
                assert!(b.len() >= Self::BYTES);

                let mut start = 0;

                $(
                    idx!(self.0, $is).to_bytes(&mut b[start..]);
                    start += $types::BYTES;
                )+

                start
            }
            fn from_bytes(b: &[u8]) -> Self {
                let mut starts = Vec::new();
                starts.push(0);
                $(
                starts.push(starts.last().unwrap() + $types::BYTES);
                )+

                FWrap(($(
                    $types::from_bytes(&b[starts[$is]..]),
                )+))
            }
        }


        impl<$($types: ConstInt,)+> ConstInt for FWrap<($($types,)+)> {
            fn zero() -> Self {
                FWrap(($($types::zero(),)+))
            }
            fn one() -> Self {
                FWrap(($($types::one(),)+))
            }
            fn is_zero(&self) -> bool {
                $(idx!(self.0, $is).is_zero() &&)+ true
            }
        }
    }
}


macro_rules! impl_tuple_arith {
    ($($is: tt),+; $($types:ident),+) => {
        impl_const_int_tuple!($(,$is)+; $($types,)+);
        impl_unary_tuple!($(,$is)+; $($types,)+; Neg, neg);
        impl_arith_assign_tuple!($(,$is)+; $($types,)+; Add, add, AddAssign, add_assign);
        impl_arith_assign_tuple!($(,$is)+; $($types,)+; Sub, sub, SubAssign, sub_assign);
        impl_arith_assign_tuple!($(,$is)+; $($types,)+; Mul, mul, MulAssign, mul_assign);
        impl_sum_prod_tuple!($($types,)+);

        impl<E: ConstInt, $($types: ExpGroup<E>,)+> ExpGroup<E> for FWrap<($($types,)+)> {
            fn exp(&self, e: &E) -> Self {
                FWrap(($( idx!(self.0, $is).exp(e), )+))
            }
        }

        impl<$($types: RandElement,)+> RandElement for FWrap<($($types,)+)> {
            fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
                FWrap(($(
                    $types::rand(rng),
                )+))
            }
        }
    }
}

impl_tuple_arith!(0; T0);
impl_tuple_arith!(0, 1; T0, T1);
impl_tuple_arith!(0, 1, 2; T0, T1, T2);
impl_tuple_arith!(0, 1, 2, 3; T0, T1, T2, T3);




impl<T: From<u64>, const N: usize> From<u64> for FWrap<[T; N]> {
    fn from(other: u64) -> Self {
        return FWrap(std::array::from_fn(|_| T::from(other)));
    }
}

impl<T: ToFromBytes, const N: usize> ToFromBytes for FWrap<[T; N]> {
    const BYTES: usize = {T::BYTES * N};

    fn num_bytes(&self) -> usize {
        Self::BYTES
    }

    fn to_bytes(&self, b: &mut [u8]) -> usize {
        assert!(b.len() >= Self::BYTES);

        let mut start = 0;

        for i in 0..N {
            self.0[i].to_bytes(&mut b[start..]);
            start += T::BYTES;
        }

        start
    }
    fn from_bytes(b: &[u8]) -> Self {
        assert!(b.len() >= Self::BYTES);

        let s = T::BYTES;
        let a = std::array::from_fn(|i| T::from_bytes(&b[i*s..]));

        FWrap(a)
    }
}

impl<T: ConstInt, const N: usize> ConstInt for FWrap<[T; N]> {
    fn zero() -> Self {
        FWrap(std::array::from_fn(|_| T::zero()))
    }
    fn one() -> Self {
        FWrap(std::array::from_fn(|_| T::one()))
    }
    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

macro_rules! impl_arith_assign_array {
    ($trait:ident, $tf:ident, $traitassign:ident, $taf:ident) => {
        impl<T: $traitassign, const N: usize> $traitassign for FWrap<[T; N]> {
            fn $taf(&mut self, other: Self) {
                self.0.iter_mut().zip(other.0.into_iter())
                    .for_each(|(s, o)| s.$taf(o));
            }
        }

        impl<'a, T: $traitassign<&'a T>, const N: usize> $traitassign<&'a Self> for FWrap<[T; N]> {
            fn $taf(&mut self, other: &'a Self) {
                for i in 0..N {
                    self.0[i].$taf(&other.0[i]);
                }
            }
        }

        impl_arith!(FWrap<[T; N]>, FWrap<[T; N]>, $trait, $tf, $traitassign, $taf, {const N: usize, T: Clone + $traitassign + for<'c> $traitassign<&'c T>}; {(): });
    };
}

impl<T: Neg<Output=T>+Clone, const N: usize> FWrap<[T; N]> {

    pub fn negt(self) -> Self {
        // TODO: should be able to do this in place
        Self(self.0.map(|x| x.neg()))
    }

}

impl_unary!(FWrap<[T; N]>, Neg, neg, negt, {T: Neg<Output=T>+Clone, const N: usize}; {():});

impl<T, const N: usize> From<[T; N]> for FWrap<[T; N]> {
    fn from(e: [T; N]) -> Self {
        Self(e)
    }
}

impl<T: Clone, const N: usize> FWrap<[T; N]> {
    pub fn broadcast(v: T) -> Self {
        Self(std::array::from_fn(|_| v.clone()))
    }
}

impl<T: Ring, const N: usize> FWrap<[T; N]> {

    fn add_assign_T(&mut self, other: &T) {
        for i in 0..N {
            self.0[i] += other;
        }
    }
    fn sub_assign_T(&mut self, other: &T) {
        for i in 0..N {
            self.0[i] *= other;
        }
    }
    fn mul_assign_T(&mut self, other: &T) {
        for i in 0..N {
            self.0[i] *= other;
        }
    }
}

impl_arith_assign_array!(Add, add, AddAssign, add_assign);
impl_arith_assign_array!(Sub, sub, SubAssign, sub_assign);
impl_arith_assign_array!(Mul, mul, MulAssign, mul_assign);

// extras for working with byte arrays
impl_arith_assign_array!(BitXor, bitxor, BitXorAssign, bitxor_assign);
impl_arith_assign_array!(BitAnd, bitand, BitAndAssign, bitand_assign);
impl_arith_assign_array!(BitOr, bitor, BitOrAssign, bitor_assign);

impl_sum_prod!(FWrap<[T; N]>, {T, const N: usize};
    {FWrap<[T; N]>: ConstInt + AddAssign + for<'c> AddAssign<&'c Self> };
    {FWrap<[T; N]>: ConstInt + MulAssign + for<'c> MulAssign<&'c Self> });

impl<E: ConstInt, T: ExpGroup<E>, const N: usize> ExpGroup<E> for FWrap<[T; N]> {
    fn exp(&self, e: &E) -> Self {
        FWrap(std::array::from_fn(|i| self.0[i].exp(e)))
    }
}

impl_module!(FWrap<[T; N]>, T, T, N, {T: Ring, const N: usize}; {():});

impl<T, const N: usize> IntoIterator for FWrap<[T; N]> {
    type Item = T;
    type IntoIter = <[T;N] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: RandElement, const N: usize> RandElement for FWrap<[T; N]> {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        FWrap(std::array::from_fn(|_| T::rand(rng)))
    }
}
