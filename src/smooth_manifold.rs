//! Smooth manifolds: charts, atlases, transition maps.

use nalgebra::{DVector, DMatrix, Dynamic, VecStorage};
use serde::{Serialize, Deserialize};

/// A coordinate chart mapping an open subset of R^n to/from the manifold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chart {
    /// Dimension of the ambient Euclidean space (chart domain).
    pub dimension: usize,
    /// Human-readable label.
    pub name: String,
}

impl Chart {
    pub fn new(dimension: usize, name: &str) -> Self {
        Self { dimension, name: name.to_string() }
    }
}

/// A smooth transition map φ_ij = φ_j ∘ φ_i^{-1} between two charts.
pub struct TransitionMap {
    pub from_chart: usize,
    pub to_chart: usize,
    /// The Jacobian matrix of the transition at a point.
    pub jacobian_fn: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
}

impl Clone for TransitionMap {
    fn clone(&self) -> Self {
        // Transition maps can't easily be cloned; provide a placeholder
        panic!("TransitionMap cannot be cloned; use references instead");
    }
}

/// A smooth atlas: a collection of compatible charts with transition maps.
pub struct Atlas {
    pub charts: Vec<Chart>,
    pub transitions: Vec<TransitionMap>,
    pub dimension: usize,
}

impl Atlas {
    pub fn new(dimension: usize) -> Self {
        Self { charts: Vec::new(), transitions: Vec::new(), dimension }
    }

    pub fn add_chart(&mut self, name: &str) -> usize {
        let idx = self.charts.len();
        self.charts.push(Chart::new(self.dimension, name));
        idx
    }

    /// Add a transition map between chart `i` and chart `j`.
    pub fn add_transition(&mut self, i: usize, j: usize, jacobian: impl Fn(&DVector<f64>) -> DMatrix<f64> + 'static) {
        self.transitions.push(TransitionMap {
            from_chart: i,
            to_chart: j,
            jacobian_fn: Box::new(jacobian),
        });
    }

    /// Get the Jacobian of the transition map from chart `i` to chart `j` at point `p`.
    pub fn transition_jacobian(&self, i: usize, j: usize, p: &DVector<f64>) -> Option<DMatrix<f64>> {
        for t in &self.transitions {
            if t.from_chart == i && t.to_chart == j {
                return Some((t.jacobian_fn)(p));
            }
        }
        // Check inverse
        for t in &self.transitions {
            if t.from_chart == j && t.to_chart == i {
                let jac = (t.jacobian_fn)(p);
                if let Some(inv) = jac.try_inverse() {
                    return Some(inv);
                }
            }
        }
        None
    }
}

/// A smooth manifold equipped with an atlas.
pub struct SmoothManifold {
    pub atlas: Atlas,
    pub name: String,
}

impl SmoothManifold {
    pub fn new(name: &str, dimension: usize) -> Self {
        Self { atlas: Atlas::new(dimension), name: name.to_string() }
    }

    pub fn dimension(&self) -> usize {
        self.atlas.dimension
    }
}

// Helper: check determinant is nonzero (used in transition_jacobian)
trait DeterminantCheck {
    fn determinant_is_nonzero(&self) -> bool;
}

impl DeterminantCheck for DMatrix<f64> {
    fn determinant_is_nonzero(&self) -> bool {
        if self.nrows() != self.ncols() { return false; }
        self.clone().try_inverse().is_some()
    }
}

/// Create the circle S^1 as a smooth 1-manifold with two-chart atlas.
pub fn circle_manifold() -> SmoothManifold {
    let mut m = SmoothManifold::new("S^1", 1);
    let c0 = m.atlas.add_chart("theta_0");
    let c1 = m.atlas.add_chart("theta_1");
    // Transition is identity on the overlap
    m.atlas.add_transition(c0, c1, |_: &DVector<f64>| DMatrix::identity(1, 1));
    m.atlas.add_transition(c1, c0, |_: &DVector<f64>| DMatrix::identity(1, 1));
    m
}

/// Create the 2-sphere S^2 as a smooth 2-manifold with stereographic charts.
pub fn sphere_manifold_2d() -> SmoothManifold {
    let mut m = SmoothManifold::new("S^2", 2);
    let north = m.atlas.add_chart("stereographic_north");
    let south = m.atlas.add_chart("stereographic_south");
    // Stereographic transition Jacobian: for (x,y) -> (x/(r²), y/(r²)) where r²=x²+y²
    // Jacobian of the transition
    m.atlas.add_transition(north, south, |p: &DVector<f64>| {
        let x = p[0]; let y = p[1];
        let r2 = x*x + y*y;
        let r4 = r2 * r2;
        if r4 < 1e-30 { return DMatrix::zeros(2,2); }
        // d/dx (x/r²) = (r² - 2x²)/r⁴, d/dy (x/r²) = -2xy/r⁴
        // d/dx (y/r²) = -2xy/r⁴, d/dy (y/r²) = (r² - 2y²)/r⁴
        DMatrix::from_row_slice(2, 2, &[
            (r2 - 2.0*x*x)/r4, -2.0*x*y/r4,
            -2.0*x*y/r4, (r2 - 2.0*y*y)/r4,
        ])
    });
    m.atlas.add_transition(south, north, |p: &DVector<f64>| {
        let x = p[0]; let y = p[1];
        let r2 = x*x + y*y;
        let r4 = r2 * r2;
        if r4 < 1e-30 { return DMatrix::zeros(2,2); }
        DMatrix::from_row_slice(2, 2, &[
            (r2 - 2.0*x*x)/r4, -2.0*x*y/r4,
            -2.0*x*y/r4, (r2 - 2.0*y*y)/r4,
        ])
    });
    m
}

/// Create the 2-torus T^2 as a smooth 2-manifold.
pub fn torus_manifold() -> SmoothManifold {
    let mut m = SmoothManifold::new("T^2", 2);
    let c0 = m.atlas.add_chart("angle_0");
    let c1 = m.atlas.add_chart("angle_1");
    // Transition: identity (flat torus)
    m.atlas.add_transition(c0, c1, |_: &DVector<f64>| DMatrix::identity(2, 2));
    m.atlas.add_transition(c1, c0, |_: &DVector<f64>| DMatrix::identity(2, 2));
    m
}

/// Create the n-sphere S^n as a smooth manifold (using two stereographic charts).
pub fn n_sphere_manifold(n: usize) -> SmoothManifold {
    let mut m = SmoothManifold::new(&format!("S^{}", n), n);
    let c0 = m.atlas.add_chart("stereographic_north");
    let c1 = m.atlas.add_chart("stereographic_south");
    // Use a generic stereographic Jacobian function
    // We store the dimension in the closure but need fn pointer
    // So we create the closures inline
    match n {
        1 => {
            m.atlas.add_transition(c0, c1, |p: &DVector<f64>| {
                DMatrix::from_row_slice(1, 1, &[1.0])
            });
            m.atlas.add_transition(c1, c0, |p: &DVector<f64>| {
                DMatrix::from_row_slice(1, 1, &[1.0])
            });
        }
        2 => {
            m.atlas.add_transition(c0, c1, |p: &DVector<f64>| {
                let x = p[0]; let y = p[1];
                let r2 = x*x + y*y;
                let r4 = r2 * r2;
                if r4 < 1e-30 { return DMatrix::zeros(2,2); }
                DMatrix::from_row_slice(2, 2, &[
                    (r2 - 2.0*x*x)/r4, -2.0*x*y/r4,
                    -2.0*x*y/r4, (r2 - 2.0*y*y)/r4,
                ])
            });
            m.atlas.add_transition(c1, c0, |p: &DVector<f64>| {
                let x = p[0]; let y = p[1];
                let r2 = x*x + y*y;
                let r4 = r2 * r2;
                if r4 < 1e-30 { return DMatrix::zeros(2,2); }
                DMatrix::from_row_slice(2, 2, &[
                    (r2 - 2.0*x*x)/r4, -2.0*x*y/r4,
                    -2.0*x*y/r4, (r2 - 2.0*y*y)/r4,
                ])
            });
        }
        3 => {
            m.atlas.add_transition(c0, c1, |p: &DVector<f64>| {
                let r2 = p.dot(p);
                let r4 = r2 * r2;
                if r4 < 1e-30 { return DMatrix::zeros(3,3); }
                DMatrix::from_row_slice(3, 3, &[
                    (r2-2.0*p[0]*p[0])/r4, -2.0*p[0]*p[1]/r4, -2.0*p[0]*p[2]/r4,
                    -2.0*p[1]*p[0]/r4, (r2-2.0*p[1]*p[1])/r4, -2.0*p[1]*p[2]/r4,
                    -2.0*p[2]*p[0]/r4, -2.0*p[2]*p[1]/r4, (r2-2.0*p[2]*p[2])/r4,
                ])
            });
            m.atlas.add_transition(c1, c0, |p: &DVector<f64>| {
                let r2 = p.dot(p);
                let r4 = r2 * r2;
                if r4 < 1e-30 { return DMatrix::zeros(3,3); }
                DMatrix::from_row_slice(3, 3, &[
                    (r2-2.0*p[0]*p[0])/r4, -2.0*p[0]*p[1]/r4, -2.0*p[0]*p[2]/r4,
                    -2.0*p[1]*p[0]/r4, (r2-2.0*p[1]*p[1])/r4, -2.0*p[1]*p[2]/r4,
                    -2.0*p[2]*p[0]/r4, -2.0*p[2]*p[1]/r4, (r2-2.0*p[2]*p[2])/r4,
                ])
            });
        }
        _ => {
            // For other dimensions, use identity transition (simplified)
            let dim = n;
            m.atlas.add_transition(c0, c1, move |_: &DVector<f64>| DMatrix::identity(dim, dim));
            let dim = n;
            m.atlas.add_transition(c1, c0, move |_: &DVector<f64>| DMatrix::identity(dim, dim));
        }
    }
    m
}

/// Create R^n as a trivial smooth manifold (single chart).
pub fn euclidean_manifold(n: usize) -> SmoothManifold {
    let mut m = SmoothManifold::new(&format!("R^{}", n), n);
    m.atlas.add_chart("identity");
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_circle_manifold() {
        let m = circle_manifold();
        assert_eq!(m.dimension(), 1);
        assert_eq!(m.atlas.charts.len(), 2);
    }

    #[test]
    fn test_sphere_2d_manifold() {
        let m = sphere_manifold_2d();
        assert_eq!(m.dimension(), 2);
        assert_eq!(m.atlas.charts.len(), 2);
    }

    #[test]
    fn test_torus_manifold() {
        let m = torus_manifold();
        assert_eq!(m.dimension(), 2);
    }

    #[test]
    fn test_stereographic_transition() {
        let m = sphere_manifold_2d();
        let p = DVector::from_vec(vec![1.0, 0.0]);
        let jac = m.atlas.transition_jacobian(0, 1, &p).unwrap();
        // At (1,0): r²=1, transition Jacobian diag((1-2)/1, 1/1) = diag(-1, 1)
        assert!((jac[(0,0)] + 1.0).abs() < 1e-10);
        assert!((jac[(1,1)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_manifold() {
        let m = euclidean_manifold(3);
        assert_eq!(m.dimension(), 3);
        assert_eq!(m.atlas.charts.len(), 1);
    }

    #[test]
    fn test_n_sphere_3() {
        let m = n_sphere_manifold(3);
        assert_eq!(m.dimension(), 3);
        assert_eq!(m.atlas.charts.len(), 2);
    }
}
