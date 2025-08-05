use std::{
    ops::{Neg, Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
    marker::PhantomData,
    fmt::Debug,
};

use crate::{
    field_macros::{impl_all_group, impl_all_ring, impl_just_ring, impl_module},
    field::{ToFromBytes, RandElement, ConstInt, Group, Ring, Module, FWrap, Extension, RingExtension},
};

use rand::Rng;

/// TODO: Matrix views/slices/strides


pub trait VectorStorage<R, const N: usize> {

    type TypeCon<const M: usize>: VectorStorage<R, M>;

    fn as_array(self) -> [R; N];
    fn get(&self, i: usize) -> R;
    fn set(&mut self, i: usize, val: R);
    fn broadcast(val: R) -> Self;
    fn concat<const M: usize>(&self, other: &Self::TypeCon<M>) -> Self::TypeCon<{N+M}>;
}

impl<R: Clone, const N: usize> VectorStorage<R, N> for FWrap<[R; N]> {
    type TypeCon<const M: usize> = FWrap<[R; M]>; 

    fn as_array(self) -> [R; N] {
        self.0
    }

    fn get(&self, i: usize) -> R {
        assert!(i < N);
        self.0[i].clone()
    }

    fn set(&mut self, i: usize, val: R) {
        assert!(i < N);
        self.0[i] = val;
    }
    fn broadcast(val: R) -> Self {
        FWrap::broadcast(val)
    }

    fn concat<const M: usize>(&self, other: &Self::TypeCon<M>) -> FWrap<[R; N+M]> {
        // using Vec to avoid unsafety with maybeuninit since there doesn't seem to be a
        // slice.concat that works with different sized arrays
        //let new: [R; {N+M}] = [self.0, other.0].concat();
        let mut new = Vec::with_capacity(N+M);
        new.extend_from_slice(&self.0);
        new.extend_from_slice(&other.0);

        let a: [R; N+M] = new.try_into().map_err(|_| ()).unwrap();

        FWrap(a)
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct Vector<R, const N: usize, S = FWrap<[R; N]>> {
    pub i: S,
    _r: PhantomData<R>
}

impl<R, const N: usize, S: VectorStorage<R, N>> IntoIterator for Vector<R,N, S> 
    where S: IntoIterator<Item=R>
{
    type Item = R;
    type IntoIter = <S as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.i.into_iter()
    }
}

impl<R, const N: usize, S: VectorStorage<R, N>> From<[R; N]> for Vector<R, N, S> 
    where S: From<[R; N]>
{
    fn from(other: [R; N]) -> Self {
        Self {
            i: S::from(other),
            _r: PhantomData
        }
    }
}

impl<R: Clone, const N: usize> TryFrom<&[R]> for Vector<R, N> {
    type Error = ();
    fn try_from(other: &[R]) -> Result<Self, Self::Error> {
        let c = other.first_chunk().ok_or(())?;
        let a: [R; N] = c.clone();
        Ok(Self {
            i: FWrap(a),
            _r: PhantomData
        })
    }
}

impl<R, const N: usize> TryFrom<Vec<R>> for Vector<R, N> {
    type Error = Vec<R>;
    fn try_from(other: Vec<R>) -> Result<Self, Self::Error> {
        let a: [R; N] = other.try_into()?;
        Ok(Self {
            i: FWrap(a),
            _r: PhantomData
        })
    }
}

impl<R: Clone, const N: usize, S: VectorStorage<R, N>> Vector<R, N, S> 
    where S: From<[R; N]>
{

    pub fn as_array(self) -> [R; N] {
        self.i.as_array()
    }

    pub fn broadcast(v: R) -> Self {
        Self {
            i: S::broadcast(v),
            _r: PhantomData
        }
    }

    pub fn concat<const M: usize>(&self, other: &Vector<R, M, S::TypeCon<M>>) -> Vector<R, {N+M}, S::TypeCon<{N+M}>> {
        let new = self.i.concat(&other.i);

        Vector {
            i: new,
            _r: PhantomData
        }
    }

    pub fn as_column(self) -> Matrix<R, N, 1> {
        let a = std::array::from_fn(|i| {
            [self.i.get(i)].into()
        });

        Matrix {
            rows: Box::new(a.into())
        }
    }

    pub fn as_row(self) -> Matrix<R, 1, N> {

        let vals = self.i.as_array();
        let vec = vals.into();
        Matrix {
            rows: Box::new([vec].into())
        }
    }

}

impl<R: Group, const N: usize, S: VectorStorage<R, N>> Vector<R, N, S> 
    where S: Group
{
    pub fn neg(self) -> Self {
        Self {
            i: -self.i,
            _r: PhantomData
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.i += &other.i;
    }

    pub fn sub_assign(&mut self, other: &Self) {
        self.i -= &other.i;
    }
}

impl<R: Ring, const N: usize, S: VectorStorage<R, N>> Vector<R, N, S> 
    where S: Ring + IntoIterator<Item=R>
{
    pub fn mul_assign(&mut self, other: &Self) {
        self.i *= &other.i;
    }
    pub fn dot(&self, other: &Self) -> R {
        let mut c = self.i.clone();
        c *= &other.i;
        c.into_iter().sum()
    }
}

impl<R: Ring, const N: usize> Vector<R, N> 
{
    pub fn vm<const M: usize>(&self, other: &Matrix<R, N, M>) -> Vector<R, M> {
        let ot = other.transpose();
    
        std::array::from_fn(|i| self.dot(&ot.rows.i.get(i))).into()
    }
}

impl<R: Ring, const N: usize, S: VectorStorage<R, N>> Vector<R, N, S> 
    where S: Ring + Module<R> 
{
    pub fn add_assign_R(&mut self, other: &R) {
        self.i += other;
    }

    pub fn sub_assign_R(&mut self, other: &R) {
        self.i -= other;
    }
    pub fn mul_assign_R(&mut self, other: &R) {
        self.i *= other;
    }
}



impl<R: ToFromBytes, const N: usize> ToFromBytes for Vector<R, N> {
    const BYTES: usize = FWrap::<[R; N]>::BYTES;
    fn num_bytes(&self) -> usize {
        Self::BYTES
    }
    fn to_bytes(&self, b: &mut [u8]) -> usize {
        self.i.to_bytes(b)
    }
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            i: ToFromBytes::from_bytes(b),
            _r: PhantomData
        }
    }
}

impl<R: From<u64>, const N: usize> From<u64> for Vector<R, N> {
    fn from(other: u64) -> Self {
        Self {
            i: From::from(other),
            _r: PhantomData
        }
    }
}

impl<R: ConstInt, const N: usize> ConstInt for Vector<R, N> {
    fn zero() -> Self {
        Self {
            i: ConstInt::zero(),
            _r: PhantomData

        }
    }
    fn one() -> Self {
        Self {
            i: ConstInt::one(),
            _r: PhantomData
        }

    }
    fn is_zero(&self) -> bool {
        self.i.is_zero()
    }
}

impl<R: RandElement, const N: usize> RandElement for Vector<R, N> {
    fn rand<RNG: Rng + ?Sized>(rng: &mut RNG) -> Self {
        Self {
            i: FWrap::rand(rng),
            _r: PhantomData

        }
    }
}

impl_all_ring!(Vector<R, N>, {R: Ring, const N: usize}; {():});
impl_module!(Vector<R, N>, R, R, N, {R: Ring, const N: usize}; {():});



#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<R, const N: usize, const M: usize> {
    pub rows: Box<Vector<Vector<R, M>, N>>
}

/*
pub struct MatrixView<'a, R, const N: usize, const M: usize> {
    cols: Range,
    rows: Range,
    mat: &'a Matrix<R, N, M>,
}
*/

impl<R: Clone, const N: usize, const M: usize> Matrix<R, N, M> {

    // TOOD: hack around compiler limitations
    pub fn split_col<const S: usize, const T: usize>(&self) -> (Matrix<R, N, S>, Matrix<R, N, T>)
        //where [(); M-S]: 
    {
        assert!(S+T == M);
        
        let (a1, a2) : (Vec<_>, Vec<_>) = self.rows()
            .map(|r| {
                let a1 = &r.i.0[..S];
                let a2 = &r.i.0[S..M];
                (a1.try_into().unwrap(), a2.try_into().unwrap())
            }).unzip();

        // this should never err
        if let (Ok(m1), Ok(m2)) = (a1.try_into(), a2.try_into()) {
            (m1, m2)
        } else {
            unreachable!()
        }
    }

    pub fn split_row<const S: usize>(&self) -> (Matrix<R, S, M>, Matrix<R, {N-S}, M>)
        where [(); N-S]: {

        let a1 = std::array::from_fn(|i| self.rows.i.0[i].clone());
        let a2 = std::array::from_fn(|i| self.rows.i.0[S+i].clone());

        (Matrix { rows: Box::new(a1.into())}, Matrix { rows: Box::new(a2.into()) } )
    }

    pub fn get_row(&self, i: usize) -> &Vector<R, M> {
        &self.rows.i.0[i]
    }

    pub fn rows(&self) -> impl Iterator<Item=&Vector<R, M>> {
        self.rows.i.0.iter()
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item=&mut Vector<R, M>> {
        self.rows.i.0.iter_mut()
    }

    pub fn transpose(&self) -> Matrix<R, M, N> {
        // TODO: allocations, efficiency
        let a = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                self.rows.i.0[j].i.0[i].clone()
            }).into()
        });

        Matrix {
            rows: Box::new(a.into())
        }
    }
}

impl<R: Ring, const N: usize, const M: usize> Matrix<R, N, M> {

    pub fn broadcast(v: R) -> Self {
        let vec: Vector<R, M> = Vector::broadcast(v);
        Self {
            rows: Box::new(Vector::broadcast(vec))
        }
    }

    pub fn diag(v: &Vector<R, N>) -> Self {
        let mut z = Self::zero();
        for i in 0..N.min(M) {
            z.rows.i.0[i].i.0[i] = v.i.0[i].clone();
        }
        z
    }

    pub fn neg(mut self) -> Self {
        
        *self.rows = self.rows.neg();

        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        *self.rows += &*other.rows;
    }

    pub fn sub_assign(&mut self, other: &Self) {
        *self.rows -= &*other.rows;
    }

    pub fn add_assign_R(&mut self, other: &R) {
        self.rows.i.0.iter_mut().for_each(|j| *j += other);
    }

    pub fn sub_assign_R(&mut self, other: &R) {
        self.rows.i.0.iter_mut().for_each(|j| *j -= other);
    }

    pub fn mul_assign_R(&mut self, other: &R) {
        self.rows.i.0.iter_mut().for_each(|j| *j *= other);
    }

    pub fn component_mul(&mut self, other: &Self) {
        *self.rows *= &*other.rows;
    }

    pub fn mv(&self, other: &Vector<R, M>) -> Vector<R, N> {
        let a = std::array::from_fn(|i| self.rows.i.0[i].dot(other));

        a.into()
    }

    pub fn mm<const O: usize>(&self, other: &Matrix<R, M, O>) -> Matrix<R, N, O> {
        // TODO: allocations
        let other = other.transpose();
        let a = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                self.rows.i.0[i].dot(&other.rows.i.0[j])
            }).into()
        });

        Matrix {
            rows: Box::new(a.into())
        }
    }
}

impl<R: Ring, const N: usize> Matrix<R, N, N> {
    pub fn mul_assign(&mut self, other: &Self) {
        *self = self.mm(other);
    }
}


impl<R: ToFromBytes, const N: usize, const M: usize> ToFromBytes for Matrix<R, N, M> {
    const BYTES: usize = Vector::<R,N>::BYTES * M;
    fn num_bytes(&self) -> usize {
        Self::BYTES
    }
    fn to_bytes(&self, b: &mut [u8]) -> usize {
        self.rows.to_bytes(b)
    }
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            rows: Box::new(ToFromBytes::from_bytes(b))
        }
    }
}

impl<R: From<u64>, const N: usize, const M: usize> From<u64> for Matrix<R, N, M> {
    fn from(other: u64) -> Self {
        Self {
            rows: Box::new(From::from(other))
        }
    }
}

impl<R, const N: usize, const M: usize> From<Vector<Vector<R, M>, N>> for Matrix<R, N, M> {
    fn from(other: Vector<Vector<R, M>, N>) -> Self {
        Self {
            rows: Box::new(other)
        }
    }
}

impl<R, const N: usize, const M: usize> TryFrom<Vec<Vector<R, M>>> for Matrix<R, N, M> {
    type Error = Vec<Vector<R, M>>;
    fn try_from(other: Vec<Vector<R, M>>) -> Result<Self, Self::Error> {
        let a: Vector<Vector<R, M>, N> = other.try_into()?;
        Ok(Self {
            rows: Box::new(a)
        })
    }
}

impl<R: ConstInt, const N: usize, const M: usize> ConstInt for Matrix<R, N, M> {
    fn zero() -> Self {
        Self {
            rows: Box::new(ConstInt::zero())
        }
    }

    fn one() -> Self {
        let mut s = Self::zero();
        for i in 0..N.min(M) {
            s.rows.i.0[i].i.0[i] = R::one();
        }

        s
    }

    fn is_zero(&self) -> bool {
        self.rows.is_zero()
    }
}
impl<R: RandElement, const N: usize, const M: usize> RandElement for Matrix<R, N, M> {
    fn rand<RNG: Rng + ?Sized>(rng: &mut RNG) -> Self {
        Self {
            rows: Box::new(Vector::rand(rng))
        }
    }
}

impl<R1: Ring, R2: Ring + RingExtension<R1>, const N: usize, const M: usize> Extension<Matrix<R1, N, M>> for Matrix<R2, N, M> {
    fn embed(el: &Matrix<R1, N, M>) -> Self {
        let mut out = Self::zero();

        for i in 0..N {
            for j in 0..M {
                out.rows.i.0[i].i.0[j] = R2::embed(&el.rows.i.0[i].i.0[j]);
            }
        }

        out
    }
}


impl_all_group!(Matrix<R, N, M>, {R: Ring, const N: usize, const M: usize}; {(): });
impl_just_ring!(Matrix<R, N, N>, {R: Ring, const N: usize}; {(): });
impl_module!(Matrix<R, N, M>, R, R, N*M, {R: Ring, const N: usize, const M: usize}; {(): });
