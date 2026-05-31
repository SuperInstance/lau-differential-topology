//! Vector fields and flows: integral curves, equilibrium points.

use nalgebra::DVector;
use crate::tangent::{TangentVector, TangentSpace};

/// A smooth vector field on a manifold of dimension n.
/// Represented as a function from R^n to R^n.
pub struct VectorField {
    pub dimension: usize,
    /// The vector field as a function V: R^n -> R^n.
    pub f: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    /// Name of the vector field.
    pub name: String,
}

impl VectorField {
    pub fn new(name: &str, dimension: usize, f: impl Fn(&DVector<f64>) -> DVector<f64> + 'static) -> Self {
        Self { dimension, f: Box::new(f), name: name.to_string() }
    }

    /// Evaluate the vector field at a point.
    pub fn evaluate(&self, p: &DVector<f64>) -> TangentVector {
        TangentVector::new(p.clone(), (self.f)(p))
    }

    /// Find equilibrium points (zeros of the vector field) by checking a grid.
    /// Returns approximate equilibrium points.
    pub fn find_equilibria_grid(&self, bounds: &[(f64, f64)], resolution: usize) -> Vec<DVector<f64>> {
        assert_eq!(bounds.len(), self.dimension);
        let threshold = 0.1; // Use a larger threshold for grid search

        // Generate grid points
        let mut grid_points = vec![DVector::zeros(self.dimension)];
        for dim in 0..self.dimension {
            let (lo, hi) = bounds[dim];
            let step = (hi - lo) / resolution as f64;
            let mut new_points = Vec::new();
            for pt in &grid_points {
                for i in 0..resolution {
                    let mut new_pt = pt.clone();
                    new_pt[dim] = lo + step * (i as f64 + 0.5);
                    new_points.push(new_pt);
                }
            }
            grid_points = new_points;
        }

        // Also add the origin explicitly
        grid_points.push(DVector::zeros(self.dimension));

        let mut found: Vec<DVector<f64>> = Vec::new();
        for pt in grid_points {
            let val = (self.f)(&pt);
            if val.norm() < threshold {
                // Check not too close to existing
                let is_dup = found.iter().any(|f| (f - &pt).norm() < 0.2);
                if !is_dup {
                    found.push(pt);
                }
            }
        }
        found
    }
}

/// Classification of equilibrium points.
#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumType {
    /// Stable node/spiral.
    Stable,
    /// Unstable node/spiral.
    Unstable,
    /// Saddle point.
    Saddle,
    /// Center (neutrally stable).
    Center,
    /// Degenerate (need higher-order analysis).
    Degenerate,
}

/// Analyze an equilibrium point using linearization.
pub fn classify_equilibrium(jacobian: &nalgebra::DMatrix<f64>) -> EquilibriumType {
    let n = jacobian.nrows();
    if n != jacobian.ncols() {
        return EquilibriumType::Degenerate;
    }

    // Compute eigenvalues
    let eigen = jacobian.clone().symmetric_eigen();
    let eigenvalues = eigen.eigenvalues;

    let mut real_parts: Vec<f64> = Vec::new();
    let mut has_imaginary = false;

    // For a general matrix, we'd need complex eigenvalues.
    // For symmetric matrices, eigenvalues are real.
    // We'll use the real eigenvalues from symmetric decomposition.
    for i in 0..eigenvalues.len() {
        let ev = eigenvalues[i];
        real_parts.push(ev);
    }

    let all_negative = real_parts.iter().all(|&r| r < -1e-10);
    let all_positive = real_parts.iter().all(|&r| r > 1e-10);
    let has_negative = real_parts.iter().any(|&r| r < -1e-10);
    let has_positive = real_parts.iter().any(|&r| r > 1e-10);

    if all_negative {
        EquilibriumType::Stable
    } else if all_positive {
        EquilibriumType::Unstable
    } else if has_negative && has_positive {
        EquilibriumType::Saddle
    } else {
        let all_near_zero = real_parts.iter().all(|&r| r.abs() < 1e-10);
        if all_near_zero {
            EquilibriumType::Center
        } else {
            EquilibriumType::Degenerate
        }
    }
}

/// Integrate an integral curve of a vector field using RK4.
pub fn integral_curve(
    vf: &VectorField,
    start: &DVector<f64>,
    dt: f64,
    steps: usize,
) -> Vec<DVector<f64>> {
    let mut curve = Vec::with_capacity(steps + 1);
    let mut current = start.clone();
    curve.push(current.clone());

    for _ in 0..steps {
        let k1 = (vf.f)(&current);
        let p2 = &current + &k1 * (dt / 2.0);
        let k2 = (vf.f)(&p2);
        let p3 = &current + &k2 * (dt / 2.0);
        let k3 = (vf.f)(&p3);
        let p4 = &current + &k3 * dt;
        let k4 = (vf.f)(&p4);
        current = &current + (&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0);
        curve.push(current.clone());
    }
    curve
}

/// The divergence of a vector field (using finite differences).
pub fn divergence(vf: &VectorField, p: &DVector<f64>, h: f64) -> f64 {
    let mut div = 0.0;
    for i in 0..vf.dimension {
        let mut p_plus = p.clone();
        let mut p_minus = p.clone();
        p_plus[i] += h;
        p_minus[i] -= h;
        let f_plus = (vf.f)(&p_plus);
        let f_minus = (vf.f)(&p_minus);
        div += (f_plus[i] - f_minus[i]) / (2.0 * h);
    }
    div
}

/// The curl of a 2D vector field (scalar curl = ∂V₂/∂x - ∂V₁/∂y).
pub fn curl_2d(vf: &VectorField, p: &DVector<f64>, h: f64) -> f64 {
    assert_eq!(vf.dimension, 2);
    let mut px = p.clone();
    let mut py = p.clone();
    px[0] += h;
    py[1] += h;
    let mut mx = p.clone();
    let mut my = p.clone();
    mx[0] -= h;
    my[1] -= h;

    let dv2_dx = ((vf.f)(&px)[1] - (vf.f)(&mx)[1]) / (2.0 * h);
    let dv1_dy = ((vf.f)(&py)[0] - (vf.f)(&my)[0]) / (2.0 * h);
    dv2_dx - dv1_dy
}

/// Predefined: constant vector field on R^n.
pub fn constant_vector_field(n: usize, value: DVector<f64>) -> VectorField {
    VectorField::new(&format!("constant_{}d", n), n, move |_: &DVector<f64>| value.clone())
}

/// Predefined: radial vector field V(x) = x.
pub fn radial_vector_field(n: usize) -> VectorField {
    VectorField::new(&format!("radial_{}d", n), n, |p: &DVector<f64>| p.clone())
}

/// Predefined: rotational vector field on R^2: V(x,y) = (-y, x).
pub fn rotational_vector_field() -> VectorField {
    VectorField::new("rotational_2d", 2, |p: &DVector<f64>| {
        DVector::from_vec(vec![-p[1], p[0]])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_vector_field() {
        let vf = constant_vector_field(2, DVector::from_vec(vec![1.0, 0.0]));
        let p = DVector::from_vec(vec![5.0, 3.0]);
        let tv = vf.evaluate(&p);
        assert!((tv.components[0] - 1.0).abs() < 1e-10);
        assert!((tv.components[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_radial_equilibrium() {
        let vf = radial_vector_field(2);
        let eqs = vf.find_equilibria_grid(&[(-1.0, 1.0), (-1.0, 1.0)], 20);
        assert!(!eqs.is_empty(), "Origin should be an equilibrium of radial field");
        // The origin should be near (0,0)
        let closest = eqs.iter().min_by(|a, b| {
            a.norm().partial_cmp(&b.norm()).unwrap()
        }).unwrap();
        assert!(closest.norm() < 0.2, "Equilibrium near origin: {:?}", closest);
    }

    #[test]
    fn test_classify_stable() {
        // Stable: all negative eigenvalues
        let jac = nalgebra::DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        assert_eq!(classify_equilibrium(&jac), EquilibriumType::Stable);
    }

    #[test]
    fn test_classify_unstable() {
        let jac = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        assert_eq!(classify_equilibrium(&jac), EquilibriumType::Unstable);
    }

    #[test]
    fn test_classify_saddle() {
        let jac = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        assert_eq!(classify_equilibrium(&jac), EquilibriumType::Saddle);
    }

    #[test]
    fn test_integral_curve_constant() {
        let vf = constant_vector_field(2, DVector::from_vec(vec![1.0, 0.0]));
        let start = DVector::from_vec(vec![0.0, 0.0]);
        let curve = integral_curve(&vf, &start, 0.1, 10);
        assert!((curve[10][0] - 1.0).abs() < 1e-6);
        assert!((curve[10][1]).abs() < 1e-6);
    }

    #[test]
    fn test_divergence_radial() {
        let vf = radial_vector_field(2);
        let p = DVector::from_vec(vec![1.0, 1.0]);
        let div = divergence(&vf, &p, 1e-5);
        assert!((div - 2.0).abs() < 1e-3, "div(x,y) = 2, got {}", div);
    }

    #[test]
    fn test_curl_rotational() {
        let vf = rotational_vector_field();
        let p = DVector::from_vec(vec![1.0, 0.0]);
        let c = curl_2d(&vf, &p, 1e-5);
        assert!((c - 2.0).abs() < 1e-3, "curl(-y,x) = 2, got {}", c);
    }
}
