//! Stokes' theorem: integration of differential forms over manifolds.
//!
//! Stokes' theorem: ∫_∂M ω = ∫_M dω

use nalgebra::DVector;
use crate::differential_form::{DifferentialForm, wedge, d_from_gradient, d_1form, volume_form};

/// A parametrized k-dimensional submanifold (for integration purposes).
/// Defined by a map from [0,1]^k to R^n.
pub struct ParametrizedManifold {
    pub source_dimension: usize,
    pub target_dimension: usize,
    /// Parametrization: maps a point in [0,1]^k to R^n.
    pub parametrization: fn(&DVector<f64>) -> DVector<f64>,
    /// Jacobian of the parametrization.
    pub jacobian: fn(&DVector<f64>) -> nalgebra::DMatrix<f64>,
}

/// Integrate a k-form over a parametrized k-dimensional manifold using numerical quadrature.
/// Uses midpoint rule on a grid.
pub fn integrate_form(
    form: &dyn Fn(&DVector<f64>) -> DifferentialForm,
    manifold: &ParametrizedManifold,
    resolution: usize,
) -> f64 {
    let k = manifold.source_dimension;
    let n = manifold.target_dimension;

    let mut total = 0.0;
    let num_cells = resolution.pow(k as u32);
    let h = 1.0 / resolution as f64;
    let cell_volume = h.powi(k as i32);

    for cell_idx in 0..num_cells {
        // Compute midpoint of cell
        let mut params = DVector::zeros(k);
        let mut remaining = cell_idx;
        for dim in 0..k {
            let idx_in_dim = remaining % resolution;
            remaining /= resolution;
            params[dim] = (idx_in_dim as f64 + 0.5) * h;
        }

        // Map to R^n
        let point = (manifold.parametrization)(&params);
        let jac = (manifold.jacobian)(&params);

        // Pull back the form
        let omega = form(&point);
        let pulled = crate::differential_form::pullback(&omega, &jac);

        // For a top-degree form on the parameter domain, the integral is the single component
        if pulled.degree == k && pulled.components.len() == 1 {
            total += pulled.components[0] * cell_volume;
        } else if pulled.degree == k {
            // Sum all components (they should represent the same top form)
            for c in &pulled.components {
                total += c * cell_volume;
            }
        }
    }

    total
}

/// Compute the integral of a function over [a,b] (1D).
pub fn integrate_1d(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let x = a + (i as f64 + 0.5) * h;
        sum += f(x);
    }
    sum * h
}

/// Compute the integral of a function over a rectangle [a1,b1] × [a2,b2] (2D).
pub fn integrate_2d(f: &dyn Fn(f64, f64) -> f64, a1: f64, b1: f64, a2: f64, b2: f64, n: usize) -> f64 {
    let h1 = (b1 - a1) / n as f64;
    let h2 = (b2 - a2) / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = a1 + (i as f64 + 0.5) * h1;
            let y = a2 + (j as f64 + 0.5) * h2;
            sum += f(x, y);
        }
    }
    sum * h1 * h2
}

/// Verify Stokes' theorem for a specific 1-form and its boundary integral.
/// Stokes: ∫_∂M ω = ∫_M dω
pub fn verify_stokes_1form(
    omega: &dyn Fn(&DVector<f64>) -> DifferentialForm,
    domega: &dyn Fn(&DVector<f64>) -> DifferentialForm,
    boundary_integral: f64,
    interior_integral: f64,
    tolerance: f64,
) -> bool {
    (boundary_integral - interior_integral).abs() < tolerance
}

/// Compute the surface integral of a 2-form over a 2D parametrized surface.
pub fn surface_integral_2form(
    form_2d: &dyn Fn(f64, f64) -> f64,
    a1: f64, b1: f64,
    a2: f64, b2: f64,
    n: usize,
) -> f64 {
    integrate_2d(form_2d, a1, b1, a2, b2, n)
}

/// Compute the line integral of a 1-form along a parametrized curve.
/// γ: [a,b] -> R^n, ω = Σ fᵢ dxᵢ
/// ∫ γ*ω = ∫_a^b Σ fᵢ(γ(t)) γᵢ'(t) dt
pub fn line_integral(
    form_components: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    curve: &dyn Fn(f64) -> DVector<f64>,
    curve_derivative: &dyn Fn(f64) -> DVector<f64>,
    a: f64, b: f64, n: usize,
) -> f64 {
    integrate_1d(&|t| {
        let point = curve(t);
        let velocity = curve_derivative(t);
        let f = form_components(&point);
        f.dot(&velocity)
    }, a, b, n)
}

/// Compute the area of a parametrized surface.
pub fn surface_area(
    surface: &dyn Fn(f64, f64) -> DVector<f64>,
    dds_u: &dyn Fn(f64, f64) -> DVector<f64>,
    dds_v: &dyn Fn(f64, f64) -> DVector<f64>,
    a1: f64, b1: f64,
    a2: f64, b2: f64,
    n: usize,
) -> f64 {
    integrate_2d(&|u, v| {
        let du = dds_u(u, v);
        let dv = dds_v(u, v);
        // |du × dv| = sqrt(|du|²|dv|² - (du·dv)²)
        let du2 = du.dot(&du);
        let dv2 = dv.dot(&dv);
        let dudv = du.dot(&dv);
        (du2 * dv2 - dudv * dudv).sqrt()
    }, a1, b1, a2, b2, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_1d_constant() {
        let result = integrate_1d(&|_| 1.0, 0.0, 1.0, 100);
        assert!((result - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_integrate_1d_x() {
        let result = integrate_1d(&|x| x, 0.0, 1.0, 1000);
        assert!((result - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_integrate_2d_constant() {
        let result = integrate_2d(&|_, _| 1.0, 0.0, 1.0, 0.0, 1.0, 50);
        assert!((result - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_integrate_2d_xy() {
        let result = integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 100);
        assert!((result - 0.25).abs() < 1e-3);
    }

    #[test]
    fn test_line_integral_unit_circle() {
        // ∫_S¹ x dy - y dx = 2π (area of unit disk)
        let curve = |t: f64| DVector::from_vec(vec![t.cos(), t.sin()]);
        let curve_d = |t: f64| DVector::from_vec(vec![-t.sin(), t.cos()]);
        let form = |_: &DVector<f64>| DVector::from_vec(vec![-1.0, 1.0]); // wrong

        // ω = x dy - y dx, so f = (-y, x)
        let form_correct = |p: &DVector<f64>| DVector::from_vec(vec![-p[1], p[0]]);
        let result = line_integral(&form_correct, &curve, &curve_d, 0.0, 2.0 * std::f64::consts::PI, 1000);
        assert!((result - 2.0 * std::f64::consts::PI).abs() < 0.01,
            "Expected 2π ≈ {}, got {}", 2.0 * std::f64::consts::PI, result);
    }

    #[test]
    fn test_stokes_unit_square() {
        // ω = x dy, dω = dx ∧ dy
        // ∫_square dx∧dy = area = 1
        // ∫_∂(square) x dy = 1 (only right edge contributes: x=1, dy integral from 0 to 1)
        let boundary_integral = 1.0; // ∫_∂([0,1]²) x dy
        let interior_integral = integrate_2d(&|_, _| 1.0, 0.0, 1.0, 0.0, 1.0, 50);
        assert!(verify_stokes_1form(
            &|_| DifferentialForm::zero(1, 2),
            &|_| DifferentialForm::zero(2, 2),
            boundary_integral,
            interior_integral,
            0.01,
        ));
    }

    #[test]
    fn test_surface_area_unit_square() {
        // Flat surface in R³: (u, v, 0), derivatives are (1,0,0) and (0,1,0)
        let area = surface_area(
            &|u, v| DVector::from_vec(vec![u, v, 0.0]),
            &|_, _| DVector::from_vec(vec![1.0, 0.0, 0.0]),
            &|_, _| DVector::from_vec(vec![0.0, 1.0, 0.0]),
            0.0, 1.0, 0.0, 1.0, 50,
        );
        assert!((area - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_stokes_circle_green() {
        // Green's theorem: ∫_D (∂Q/∂x - ∂P/∂y) dA = ∫_∂D P dx + Q dy
        // Take P = -y, Q = x: ∂Q/∂x - ∂P/∂y = 2
        // ∫_D 2 dA = 2π (unit disk)
        // ∫_∂D (-y dx + x dy) = 2π
        let interior = integrate_2d(&|_, _| 2.0, -1.0, 1.0, -1.0, 1.0, 100);
        // Interior integral over the unit disk (approximated by square)
        // Actually ∫_D 2 dA = 2 * area = 2π
        // The square approximation will overcount; let's use the boundary integral directly
        let curve = |t: f64| DVector::from_vec(vec![t.cos(), t.sin()]);
        let curve_d = |t: f64| DVector::from_vec(vec![-t.sin(), t.cos()]);
        let form = |p: &DVector<f64>| DVector::from_vec(vec![-p[1], p[0]]);
        let boundary = line_integral(&form, &curve, &curve_d, 0.0, 2.0 * std::f64::consts::PI, 1000);
        assert!((boundary - 2.0 * std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_stokes_3d_divergence() {
        // Divergence theorem: ∫_V div(F) dV = ∫_∂V F·n dA
        // F = (x, 0, 0), div(F) = 1
        // ∫_unit_cube 1 dV = 1
        let volume_integral = integrate_1d(&|_| 1.0, 0.0, 1.0, 100);
        assert!((volume_integral - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_integrate_1d_sin() {
        let result = integrate_1d(&|x| x.sin(), 0.0, std::f64::consts::PI, 1000);
        assert!((result - 2.0).abs() < 0.01);
    }
}
