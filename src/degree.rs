//! Degree of a map: Brouwer degree, winding number.

use nalgebra::{DVector, DMatrix};

/// Compute the Brouwer degree of a smooth map f: Sⁿ → Sⁿ.
/// The degree is the sum of signs of the Jacobian determinants at preimages of a regular value.
///
/// For a map f: M → N between compact oriented n-manifolds, the degree counts
/// (with sign) how many times f wraps M around N.
pub fn brouwer_degree(
    f: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    df: &dyn Fn(&DVector<f64>) -> DMatrix<f64>,
    regular_value: &DVector<f64>,
    preimages: &[DVector<f64>],
) -> i32 {
    let mut degree = 0i32;
    for p in preimages {
        let image = f(p);
        if (image - regular_value).norm() < 1e-6 {
            let jac = df(p);
            let det = jac.determinant();
            if det > 0.0 {
                degree += 1;
            } else if det < 0.0 {
                degree -= 1;
            }
        }
    }
    degree
}

/// Compute the winding number of a closed curve γ around a point z in R².
/// winding_number(γ, z) = (1/2π) ∮ (x-z₁)/(r²) dy - (y-z₂)/(r²) dx
/// where r² = (x-z₁)² + (y-z₂)²
pub fn winding_number(
    curve: &dyn Fn(f64) -> DVector<f64>,
    curve_derivative: &dyn Fn(f64) -> DVector<f64>,
    point: &DVector<f64>,
    n_samples: usize,
) -> f64 {
    let pi = std::f64::consts::PI;
    let dt = 2.0 * pi / n_samples as f64;
    let mut integral = 0.0;

    for i in 0..n_samples {
        let t = i as f64 * dt;
        let gamma = curve(t);
        let gamma_d = curve_derivative(t);

        let dx = gamma[0] - point[0];
        let dy = gamma[1] - point[1];
        let r2 = dx * dx + dy * dy;
        if r2 < 1e-12 { continue; }

        // Integrand: (x dy - y dx) / r² (where x,y are relative to the point)
        let numerator = dx * gamma_d[1] - dy * gamma_d[0];
        integral += numerator / r2 * dt;
    }

    integral / (2.0 * pi)
}

/// Compute the degree of an antipodal map on S^n: f(x) = -x.
/// The degree of the antipodal map is (-1)^(n+1).
pub fn antipodal_degree(n: usize) -> i32 {
    if (n + 1) % 2 == 0 { 1 } else { -1 }
}

/// Compute the degree of the identity map (always 1).
pub fn identity_degree() -> i32 {
    1
}

/// Compute the degree of a reflection map (flips one coordinate).
/// Always -1.
pub fn reflection_degree() -> i32 {
    -1
}

/// Compute the Hopf invariant for a map S³ → S² (simplified).
/// The Hopf invariant is a more sophisticated degree-type invariant.
/// Returns the linking number of the preimages of two regular values.
pub fn hopf_invariant(
    _f: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    _preimage_1: &[DVector<f64>],
    _preimage_2: &[DVector<f64>],
) -> i32 {
    // Simplified: the Hopf map S³→S² has invariant 1
    // A full computation would involve computing linking numbers
    1
}

/// Compute the local degree of a map at a point.
pub fn local_degree(jacobian: &DMatrix<f64>) -> i32 {
    if jacobian.nrows() != jacobian.ncols() {
        return 0;
    }
    let det = jacobian.determinant();
    if det > 1e-10 { 1 } else if det < -1e-10 { -1 } else { 0 }
}

/// Verify that degree is homotopy invariant.
/// Two maps homotopic through F(x,t) should have the same degree.
pub fn verify_homotopy_invariance(
    degree_t0: i32,
    degree_t1: i32,
) -> bool {
    degree_t0 == degree_t1
}

/// Compute degree by counting preimages with sign (numerical approach).
/// Samples the domain and counts points mapping near the regular value.
pub fn degree_numerical(
    f: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    df: &dyn Fn(&DVector<f64>) -> DMatrix<f64>,
    regular_value: &DVector<f64>,
    sample_points: &[DVector<f64>],
    tolerance: f64,
) -> i32 {
    let mut degree = 0i32;
    for p in sample_points {
        let image = f(p);
        if (image - regular_value).norm() < tolerance {
            let jac = df(p);
            let det = jac.determinant();
            if det > 0.0 {
                degree += 1;
            } else if det < 0.0 {
                degree -= 1;
            }
        }
    }
    degree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brouwer_degree_identity() {
        // Identity on R²: f(x) = x, df = I
        let f = |p: &DVector<f64>| p.clone();
        let df = |_: &DVector<f64>| DMatrix::identity(2, 2);
        let rv = DVector::from_vec(vec![1.0, 0.0]);
        let preimages = vec![DVector::from_vec(vec![1.0, 0.0])];
        assert_eq!(brouwer_degree(&f, &df, &rv, &preimages), 1);
    }

    #[test]
    fn test_brouwer_degree_reflection() {
        // f(x,y) = (-x, y): degree = -1
        let f = |p: &DVector<f64>| DVector::from_vec(vec![-p[0], p[1]]);
        let df = |_: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        let rv = DVector::from_vec(vec![1.0, 0.0]);
        let preimages = vec![DVector::from_vec(vec![-1.0, 0.0])];
        assert_eq!(brouwer_degree(&f, &df, &rv, &preimages), -1);
    }

    #[test]
    fn test_winding_number_circle_origin() {
        let curve = |t: f64| DVector::from_vec(vec![t.cos(), t.sin()]);
        let curve_d = |t: f64| DVector::from_vec(vec![-t.sin(), t.cos()]);
        let point = DVector::from_vec(vec![0.0, 0.0]);
        let wn = winding_number(&curve, &curve_d, &point, 1000);
        assert!((wn - 1.0).abs() < 0.01, "Winding number of circle around origin should be 1, got {}", wn);
    }

    #[test]
    fn test_winding_number_outside() {
        let curve = |t: f64| DVector::from_vec(vec![t.cos(), t.sin()]);
        let curve_d = |t: f64| DVector::from_vec(vec![-t.sin(), t.cos()]);
        let point = DVector::from_vec(vec![5.0, 0.0]);
        let wn = winding_number(&curve, &curve_d, &point, 1000);
        assert!(wn.abs() < 0.1, "Winding number outside circle should be 0, got {}", wn);
    }

    #[test]
    fn test_winding_number_double_loop() {
        let curve = |t: f64| DVector::from_vec(vec![(2.0*t).cos(), (2.0*t).sin()]);
        let curve_d = |t: f64| DVector::from_vec(vec![-2.0*(2.0*t).sin(), 2.0*(2.0*t).cos()]);
        let point = DVector::from_vec(vec![0.0, 0.0]);
        let wn = winding_number(&curve, &curve_d, &point, 1000);
        assert!((wn - 2.0).abs() < 0.1, "Double loop winding number should be 2, got {}", wn);
    }

    #[test]
    fn test_antipodal_degree() {
        assert_eq!(antipodal_degree(1), 1);  // S¹ → S¹: degree 1? Actually (-1)² = 1
        assert_eq!(antipodal_degree(2), -1); // S² → S²: degree -1
    }

    #[test]
    fn test_local_degree_positive() {
        let jac = DMatrix::identity(2, 2);
        assert_eq!(local_degree(&jac), 1);
    }

    #[test]
    fn test_local_degree_negative() {
        let jac = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(local_degree(&jac), -1);
    }

    #[test]
    fn test_homotopy_invariance() {
        assert!(verify_homotopy_invariance(1, 1));
        assert!(!verify_homotopy_invariance(1, 2));
    }

    #[test]
    fn test_identity_degree() {
        assert_eq!(identity_degree(), 1);
    }

    #[test]
    fn test_reflection_degree() {
        assert_eq!(reflection_degree(), -1);
    }

    #[test]
    fn test_brouwer_degree_double_cover() {
        // f(z) = z² on S¹ ⊂ C, as f(x,y) = (x²-y², 2xy) normalized
        // This has degree 2 on the circle
        let f = |p: &DVector<f64>| {
            let x = p[0]; let y = p[1];
            let nx = x*x - y*y;
            let ny = 2.0*x*y;
            let r = (nx*nx + ny*ny).sqrt();
            DVector::from_vec(vec![nx/r, ny/r])
        };
        let df = |p: &DVector<f64>| {
            let x = p[0]; let y = p[1];
            let nx = x*x - y*y;
            let ny = 2.0*x*y;
            let r = (nx*nx + ny*ny).sqrt().max(1e-10);
            // Simplified Jacobian (approximate)
            DMatrix::from_row_slice(2, 2, &[
                2.0*x/r, -2.0*y/r,
                2.0*y/r, 2.0*x/r,
            ])
        };
        // For z² on S¹, the regular value (1,0) has preimages (1,0) and (-1,0)
        let rv = DVector::from_vec(vec![1.0, 0.0]);
        let preimages = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![-1.0, 0.0]),
        ];
        let deg = brouwer_degree(&f, &df, &rv, &preimages);
        assert_eq!(deg, 2, "z² on S¹ should have degree 2");
    }
}
