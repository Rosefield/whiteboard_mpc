
/*
use std::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, BitXor, BitXorAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign},
};

use crate::field::{
    ToFromBytes, ConstInt, Group, ExpGroup, Ring, Field, Module, VectorSpace,
};
*/



macro_rules! impl_arith {
    ($type:ty, $type2:ty, $trait:ident, $tf:ident, $traitassign:ident, $taf:ident, {$($params:tt)*}; {$($constraint:tt)*}) => {
        impl<$($params)*> $trait<$type2> for $type
            where $($constraint)*
        {
            type Output = Self;
            fn $tf(self, other: $type2) -> Self {
                let mut c = self;
                <Self as $traitassign<$type2>>::$taf(&mut c, other);
                c
            }
        }

        impl<$($params)*> $trait<&$type2> for $type
            where $($constraint)*
        {
            type Output = Self;
            fn $tf(self, other: &$type2) -> Self {
                let mut c = self;
                <Self as $traitassign<&$type2>>::$taf(&mut c, other);
                c
            }
        }

        impl<'a, $($params)*> $trait<$type2> for &'a $type
            where $($constraint)*
        {
            type Output = $type;
            fn $tf(self, other: $type2) -> $type {
                let mut c = self.clone();
                <$type as $traitassign<$type2>>::$taf(&mut c, other);
                c
            }
        }

        impl<'a, 'b, $($params)*> $trait<&'b $type2> for &'a $type
            where $($constraint)*
        {
            type Output = $type;
            fn $tf(self, other: &'b $type2) -> $type {
                let mut c = self.clone();
                <$type as $traitassign<&'b $type2>>::$taf(&mut c, other);
                c
            }
        }
    };
}

macro_rules! impl_arith_assign {
    ($type:ty, $type2:ty, $trait:ident, $tf:ident, $traitassign:ident, $taf:ident, $impl:ident, {$($params:tt)*}; {$($constraint:tt)*}) => {
        impl<$($params)*> $traitassign<$type2> for $type
            where $($constraint)*
        {
            fn $taf(&mut self, other: $type2) {
                self.$impl(&other);
            }
        }

        impl<$($params)*> $traitassign<&$type2> for $type
            where $($constraint)*
        {
            fn $taf(&mut self, other: &$type2) {
                self.$impl(other);
            }
        }

        $crate::field_macros::impl_arith!($type, $type2, $trait, $tf, $traitassign, $taf, {$($params)*}; {$($constraint)*});
    };
}

macro_rules! impl_sum {
    ($type:ty, {$($params:tt)*}; {$($csum:tt)*}) => {
        impl<$($params)*> std::iter::Sum for $type
            where $($csum)*
        {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                let mut acc = Self::zero();
                for i in iter {
                    acc += i;
                }
                acc
            }
        }
        impl<'a, $($params)*> std::iter::Sum<&'a Self> for $type
            where $($csum)*
        {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                let mut acc = Self::zero();
                for i in iter {
                    acc += i;
                }
                acc
            }
        }
    };
}

macro_rules! impl_prod {
    ($type:ty, {$($params:tt)*}; {$($cprod:tt)*}) => {

        impl<$($params)*> std::iter::Product for $type
            where $($cprod)*
        {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                let mut acc = Self::one();
                for i in iter {
                    acc *= i;
                }
                acc
            }
        }

        impl<'a, $($params)*> std::iter::Product<&'a Self> for $type
            where $($cprod)*
        {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                let mut acc = Self::one();
                for i in iter {
                    acc *= i;
                }
                acc
            }
        }
    };
}

macro_rules! impl_sum_prod {
    ($type:ty, {$($params:tt)*}; {$($constraint:tt)*}) => {
        $crate::field_macros::impl_sum_prod!($type, {$($params)*}; {$($constraint)*}; {$($constraint)*});
    };
    ($type:ty, {$($params:tt)*}; {$($csum:tt)*}; {$($cprod:tt)*}) => {
        $crate::field_macros::impl_sum!($type, {$($params)*}; {$($csum)*});
        $crate::field_macros::impl_prod!($type, {$($params)*}; {$($cprod)*});
    };
}

macro_rules! impl_unary {
    ($type:ty, $trait:ident, $tf:ident, $impl:ident, {$($params:tt)*}; {$($constraint:tt)*}) => {
        impl<$($params)*> $trait for $type
            where $($constraint)*
        {
            type Output = $type;
            
            fn $tf(self) -> Self::Output {
                self.$impl()
            }
        }

        impl<$($params)*> $trait for &$type
            where $($constraint)*
        {
            type Output = $type;

            fn $tf(self) -> Self::Output {
                self.clone().$impl()
            }
        }
    }
}

macro_rules! impl_all_group {
    ($type:ty) => {
        $crate::field_macros::impl_all_group!($type, { }; {(): });
    };

    ($type:ty, {$($params:tt)*}; {$($constraint:tt)*}) => {
        $crate::field_macros::impl_unary!($type, Neg, neg, neg, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_arith_assign!($type, $type, Add, add, AddAssign, add_assign, add_assign, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_arith_assign!($type, $type, Sub, sub, SubAssign, sub_assign, sub_assign, {$($params)*}; {$($constraint)*});

        $crate::field_macros::impl_sum!($type, {$($params)*}; {$($constraint)*});
    };
}
macro_rules! impl_just_ring {
    ($type:ty) => {
        $crate::field_macros::impl_just_ring!($type, { }; {(): });
    };

    ($type:ty, {$($params:tt)*}; {$($constraint:tt)*}) => {
        $crate::field_macros::impl_arith_assign!($type, $type, Mul, mul, MulAssign, mul_assign, mul_assign, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_prod!($type, {$($params)*}; {$($constraint)*});
    };
}

macro_rules! impl_all_ring {
    ($type:ty) => {
        $crate::field_macros::impl_all_ring!($type, { }; {(): });
    };

    ($type:ty, {$($params:tt)*}; {$($constraint:tt)*}) => {
        $crate::field_macros::impl_all_group!($type, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_just_ring!($type, {$($params)*}; {$($constraint)*});
    };
}

macro_rules! impl_module {
    ($type:ty, $type2:ty, $suffix:ident, $ext:expr) => {
        impl_module!($type, $type2, $suffix, $ext, {}; {(): });
    };

    ($type:ty, $type2:ty, $suffix:ident, $ext:expr, {$($params:tt)*}; {$($constraint:tt)*}) => {
        $crate::field_macros::impl_arith_assign!($type, $type2, Add, add, AddAssign, add_assign, ${ concat(add_assign_, $suffix) }, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_arith_assign!($type, $type2, Sub, sub, SubAssign, sub_assign, ${ concat(sub_assign_, $suffix) }, {$($params)*}; {$($constraint)*});
        $crate::field_macros::impl_arith_assign!($type, $type2, Mul, mul, MulAssign, mul_assign, ${ concat(mul_assign_, $suffix) }, {$($params)*}; {$($constraint)*});

        impl<$($params)*> Module<$type2> for $type
            where $($constraint)*
        {
            const DEGREE:usize = $ext;
        }
    };
}

pub(crate) use impl_arith;
pub(crate) use impl_unary;
pub(crate) use impl_arith_assign;
pub(crate) use impl_sum;
pub(crate) use impl_prod;
pub(crate) use impl_sum_prod;
pub(crate) use impl_all_group;
pub(crate) use impl_just_ring;
pub(crate) use impl_all_ring;
pub(crate) use impl_module;
