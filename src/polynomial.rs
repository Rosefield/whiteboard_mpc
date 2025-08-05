use crate::field::{ConstInt, Field, RandElement, Ring};
use rand::Rng;

pub trait Polynomial<T: Ring> {
    fn evaluate(&self, p: &T) -> T;
    fn eval_zero(&self) -> T;
}

#[derive(Clone)]
pub struct FixedPolynomial<T> {
    coefficients: Vec<T>,
}

impl<T: Ring> FixedPolynomial<T> {
    pub fn new(coef: Vec<T>) -> Self {
        FixedPolynomial { coefficients: coef }
    }
}

impl<T: RandElement> FixedPolynomial<T> {
    pub fn rand_polynomial<R: Rng + ?Sized>(rng: &mut R, degree: usize) -> Self {
        let v: Vec<T> = (0..degree + 1).map(|_| T::rand(rng)).collect();

        FixedPolynomial { coefficients: v }
    }
}

impl<T: Ring> Polynomial<T> for FixedPolynomial<T> {
    fn evaluate(&self, p: &T) -> T {
        let mut x: T = ConstInt::one();
        self.coefficients
            .iter()
            .map(|c| {
                let a = c.clone() * &x;
                x *= p;
                return a;
            })
            .fold(ConstInt::zero(), |acc, n| acc + n)
    }

    fn eval_zero(&self) -> T {
        return self.coefficients[0].clone();
    }
}

#[derive(Clone)]
pub struct InterpolationPolynomial<T: Ring> {
    pub points: Vec<T>,
    pub vals: Vec<T>,
}

impl<T: Ring> InterpolationPolynomial<T> {
    pub fn new(points: &[T], vals: &[T]) -> Result<Self, &'static str> {
        if points.len() != vals.len() {
            return Err("vectors must have the same length");
        }

        Ok(InterpolationPolynomial {
            points: points.to_vec(),
            vals: vals.to_vec(),
        })
    }
}

impl<T: Ring + RandElement> InterpolationPolynomial<T> {
    pub fn rand_share<R: Rng>(rng: &mut R, t: usize, points: &[T]) -> Self {
        assert!(points.len() >= t.try_into().unwrap());

        let p = FixedPolynomial::rand_polynomial(rng, t - 1);

        let vals = points.iter().map(|x| p.evaluate(x)).collect();

        InterpolationPolynomial {
            points: points.to_vec(),
            vals: vals,
        }
    }

    pub fn secret_share<R: Rng>(rng: &mut R, val: T, t: usize, points: &[T]) -> Self {
        assert!(points.len() >= t.try_into().unwrap());

        let mut p = FixedPolynomial::rand_polynomial(rng, t - 1);
        p.coefficients[0] = val;

        let vals = points.iter().map(|x| p.evaluate(x)).collect();

        InterpolationPolynomial {
            points: points.to_vec(),
            vals: vals,
        }
    }
}

pub fn lagrange_poly<T: Field, F: Fn(&T) -> T>(points: &[T], xi: &T, num_func: F) -> T {
    let (n, d) = points
        .iter()
        .filter(|xj| xi != *xj)
        .map(|xj| (num_func(xj), (xi.clone() - xj)))
        .fold((T::one(), T::one()), |acc: (T, T), r: (T, T)| {
            (acc.0 * r.0, acc.1 * r.1)
        });

    n * d.inv().unwrap()
}

impl<T: Field> Polynomial<T> for InterpolationPolynomial<T> {
    fn evaluate(&self, x: &T) -> T {
        if *x == ConstInt::zero() {
            return self.eval_zero();
        }
        self.points
            .iter()
            .zip(self.vals.iter())
            .map(|(xi, yi)| {
                let l = lagrange_poly(&self.points, xi, |xj| x.clone() - xj);
                return l * yi;
            })
            .sum()
    }

    fn eval_zero(&self) -> T {
        self.points
            .iter()
            .zip(self.vals.iter())
            .map(|(xi, yi)| {
                let l = lagrange_poly(&self.points, xi, |x| -x.clone());
                return l * yi;
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ff2_128::FF2_128, field::ConstInt};

    use rand;

    #[test]
    fn test_poly_to_add() {
        let mut rng = rand::thread_rng();

        let val = FF2_128::rand(&mut rng);
        let points: Vec<_> = (1..10).map(|p| FF2_128::from(p)).collect();

        let t = 5;

        let shares = InterpolationPolynomial::secret_share(&mut rng, val.clone(), t, &points);

        for i in 1..10 {
            let p = InterpolationPolynomial::new(&shares.points[..i], &shares.vals[..i]).unwrap();

            let yi = p.eval_zero();
            if i < t {
                assert!(yi != val);
            } else {
                assert_eq!(yi, val);
            }
        }

        let l = 2;
        let zero = FF2_128::zero();
        let shares_zeros: Vec<_> = (0..2 * l)
            .map(|_| InterpolationPolynomial::secret_share(&mut rng, zero.clone(), t, &points))
            .map(|p| p.vals)
            .collect();
        let sum_vals = shares_zeros
            .into_iter()
            .reduce(|left, right| {
                left.into_iter()
                    .zip(right.iter())
                    .map(|(x, y)| x + y)
                    .collect()
            })
            .unwrap();

        let p2 = InterpolationPolynomial::new(&points, &sum_vals).unwrap();
        assert!(p2.eval_zero() == zero);
    }
}
