//! Tangent spaces and tangent bundles.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A tangent vector at a point on a manifold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TangentVector {
    /// The base point (coordinates in some chart).
    pub base_point: DVector<f64>,
    /// The tangent vector components in the same chart.
    pub components: DVector<f64>,
}

impl TangentVector {
    pub fn new(base_point: DVector<f64>, components: DVector<f64>) -> Self {
        assert_eq!(base_point.len(), components.len());
        Self { base_point, components }
    }

    /// Dimension of the tangent space.
    pub fn dimension(&self) -> usize {
        self.components.len()
    }

    /// Add two tangent vectors at the same base point.
    pub fn add(&self, other: &TangentVector) -> TangentVector {
        assert_eq!(self.base_point, other.base_point);
        TangentVector::new(self.base_point.clone(), &self.components + &other.components)
    }

    /// Scale a tangent vector.
    pub fn scale(&self, scalar: f64) -> TangentVector {
        TangentVector::new(self.base_point.clone(), &self.components * scalar)
    }

    /// Inner product of tangent vectors (using the standard metric).
    pub fn inner_product(&self, other: &TangentVector) -> f64 {
        self.components.dot(&other.components)
    }

    /// Norm of the tangent vector.
    pub fn norm(&self) -> f64 {
        self.components.norm()
    }
}

/// A tangent space at a point on a manifold of dimension n.
/// Represented as R^n with the standard inner product.
#[derive(Clone, Debug)]
pub struct TangentSpace {
    pub base_point: DVector<f64>,
    pub dimension: usize,
    /// Optional metric tensor (symmetric positive definite n×n matrix).
    pub metric: Option<DMatrix<f64>>,
}

impl TangentSpace {
    pub fn new(base_point: DVector<f64>) -> Self {
        let dim = base_point.len();
        Self { base_point, dimension: dim, metric: None }
    }

    pub fn with_metric(mut self, metric: DMatrix<f64>) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Inner product of two tangent vectors using the metric.
    pub fn inner_product(&self, v: &TangentVector, w: &TangentVector) -> f64 {
        match &self.metric {
            Some(g) => {
                let gv = g * &v.components;
                gv.dot(&w.components)
            }
            None => v.components.dot(&w.components),
        }
    }

    /// The zero tangent vector.
    pub fn zero(&self) -> TangentVector {
        TangentVector::new(self.base_point.clone(), DVector::zeros(self.dimension))
    }
}

/// The tangent bundle TM of a manifold M.
/// Represented as pairs (point, tangent_vector) with dimension 2n.
#[derive(Clone, Debug)]
pub struct TangentBundle {
    /// Dimension of the base manifold.
    pub base_dimension: usize,
    /// Total dimension of the bundle (2 * base_dimension).
    pub total_dimension: usize,
}

impl TangentBundle {
    pub fn new(base_dimension: usize) -> Self {
        Self { base_dimension, total_dimension: 2 * base_dimension }
    }

    /// Create a tangent bundle element (point, vector).
    pub fn element(&self, point: DVector<f64>, vector: DVector<f64>) -> TangentVector {
        assert_eq!(point.len(), self.base_dimension);
        assert_eq!(vector.len(), self.base_dimension);
        TangentVector::new(point, vector)
    }

    /// Project onto the base manifold.
    pub fn projection(&self, tv: &TangentVector) -> DVector<f64> {
        tv.base_point.clone()
    }

    /// The fiber over a point (the tangent space at that point).
    pub fn fiber(&self, point: DVector<f64>) -> TangentSpace {
        TangentSpace::new(point)
    }
}

/// The cotangent space at a point: dual to the tangent space.
/// Elements are covectors (linear functionals on tangent vectors).
#[derive(Clone, Debug)]
pub struct CotangentVector {
    pub base_point: DVector<f64>,
    pub components: DVector<f64>,
}

impl CotangentVector {
    pub fn new(base_point: DVector<f64>, components: DVector<f64>) -> Self {
        Self { base_point, components }
    }

    /// Apply the covector to a tangent vector.
    pub fn apply(&self, v: &TangentVector) -> f64 {
        self.components.dot(&v.components)
    }
}

/// The differential (pushforward) of a smooth map between manifolds.
pub struct Differential {
    /// Jacobian matrix of the map at a point.
    pub jacobian: DMatrix<f64>,
    pub source_dim: usize,
    pub target_dim: usize,
}

impl Differential {
    pub fn new(jacobian: DMatrix<f64>) -> Self {
        let source_dim = jacobian.ncols();
        let target_dim = jacobian.nrows();
        Self { jacobian, source_dim, target_dim }
    }

    /// Push forward a tangent vector.
    pub fn push_forward(&self, v: &TangentVector) -> TangentVector {
        let new_components = &self.jacobian * &v.components;
        TangentVector::new(v.base_point.clone(), new_components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tangent_vector_add() {
        let p = DVector::from_vec(vec![1.0, 0.0]);
        let v = TangentVector::new(p.clone(), DVector::from_vec(vec![1.0, 0.0]));
        let w = TangentVector::new(p.clone(), DVector::from_vec(vec![0.0, 1.0]));
        let sum = v.add(&w);
        assert!((sum.components[0] - 1.0).abs() < 1e-10);
        assert!((sum.components[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_vector_norm() {
        let v = TangentVector::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![3.0, 4.0]));
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_space_with_metric() {
        let p = DVector::from_vec(vec![0.0, 0.0]);
        let ts = TangentSpace::new(p).with_metric(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]));
        let v = TangentVector::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![1.0, 0.0]));
        let w = TangentVector::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![1.0, 0.0]));
        assert!((ts.inner_product(&v, &w) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_bundle() {
        let tb = TangentBundle::new(2);
        assert_eq!(tb.total_dimension, 4);
        let elem = tb.element(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let base = tb.projection(&elem);
        assert!((base[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_differential_pushforward() {
        let jac = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let d = Differential::new(jac);
        let v = TangentVector::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![1.0, 2.0]));
        let pushed = d.push_forward(&v);
        assert!((pushed.components[0] - 1.0).abs() < 1e-10);
        assert!((pushed.components[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cotangent_vector() {
        let cov = CotangentVector::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![3.0]));
        let v = TangentVector::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![5.0]));
        assert!((cov.apply(&v) - 15.0).abs() < 1e-10);
    }
}
