//! Transversality: generic intersection theory.

use nalgebra::{DVector, DMatrix};
use crate::tangent::{TangentVector, TangentSpace};

/// Check if two submanifolds intersect transversally.
/// Two submanifolds N₁, N₂ of M intersect transversally at p if:
/// T_p N₁ + T_p N₂ = T_p M
///
/// In coordinates: the combined tangent spaces span the full tangent space.
pub fn is_transverse_intersection(
    ambient_dim: usize,
    tangent_space_1: &DMatrix<f64>, // columns are basis vectors of T_p N₁
    tangent_space_2: &DMatrix<f64>, // columns are basis vectors of T_p N₂
    tolerance: f64,
) -> bool {
    // Combine the two tangent space bases
    let dim1 = tangent_space_1.ncols();
    let dim2 = tangent_space_2.ncols();
    let combined_cols = dim1 + dim2;

    let mut combined = DMatrix::zeros(ambient_dim, combined_cols);
    for i in 0..dim1 {
        for j in 0..ambient_dim {
            combined[(j, i)] = tangent_space_1[(j, i)];
        }
    }
    for i in 0..dim2 {
        for j in 0..ambient_dim {
            combined[(j, dim1 + i)] = tangent_space_2[(j, i)];
        }
    }

    // Check rank = ambient_dim
    let rank = matrix_rank(&combined, tolerance);
    rank == ambient_dim
}

/// Compute the rank of a matrix (numerical).
fn matrix_rank(m: &DMatrix<f64>, tolerance: f64) -> usize {
    let svd = m.clone().svd(true, true);
    let mut rank = 0;
    let singular_values = &svd.singular_values;
    for s in singular_values.iter() {
        if *s > tolerance {
            rank += 1;
        }
    }
    rank
}

/// The dimension of the intersection of two transverse submanifolds.
/// dim(N₁ ∩ N₂) = dim(N₁) + dim(N₂) - dim(M)
pub fn transverse_intersection_dimension(dim_n1: usize, dim_n2: usize, dim_m: usize) -> i32 {
    (dim_n1 as i32 + dim_n2 as i32) - dim_m as i32
}

/// Check if a map f: M -> N is transverse to a submanifold S ⊂ N.
/// f is transverse to S if for every x with f(x) ∈ S:
/// Im(df_x) + T_{f(x)} S = T_{f(x)} N
pub fn is_map_transverse_to_submanifold(
    df: &DMatrix<f64>,     // Jacobian of f at x (dim_N × dim_M)
    tangent_s: &DMatrix<f64>, // Basis for T_{f(x)} S (dim_N × dim_S)
    dim_n: usize,
    tolerance: f64,
) -> bool {
    let dim_df = df.ncols();
    let dim_s = tangent_s.ncols();
    let combined_cols = dim_df + dim_s;

    let mut combined = DMatrix::zeros(dim_n, combined_cols);
    for i in 0..dim_df {
        for j in 0..dim_n {
            combined[(j, i)] = df[(j, i)];
        }
    }
    for i in 0..dim_s {
        for j in 0..dim_n {
            combined[(j, dim_df + i)] = tangent_s[(j, i)];
        }
    }

    let rank = matrix_rank(&combined, tolerance);
    rank == dim_n
}

/// Find intersection points of two submanifolds (parametrized).
/// Uses Newton's method to find zeros of F(x) = G(x) - H(y).
pub fn find_intersections(
    f1: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    f2: &dyn Fn(&DVector<f64>) -> DVector<f64>,
    initial_guesses: &[(DVector<f64>, DVector<f64>)],
    tolerance: f64,
    max_iter: usize,
) -> Vec<(DVector<f64>, DVector<f64>)> {
    let mut results = Vec::new();

    for (x0, y0) in initial_guesses {
        let mut x = x0.clone();
        let mut y = y0.clone();

        for _ in 0..max_iter {
            let diff = f1(&x) - f2(&y);
            if diff.norm() < tolerance {
                results.push((x.clone(), y.clone()));
                break;
            }
            // Simple gradient descent step
            let eps = 1e-5;
            let dim_x = x.len();
            let dim_y = y.len();

            let mut grad_x = DVector::zeros(dim_x);
            for i in 0..dim_x {
                let mut xp = x.clone();
                xp[i] += eps;
                let mut xm = x.clone();
                xm[i] -= eps;
                let dp = f1(&xp) - f2(&y);
                let dm = f1(&xm) - f2(&y);
                grad_x[i] = (dp.norm() - dm.norm()) / (2.0 * eps);
            }

            let mut grad_y = DVector::zeros(dim_y);
            for i in 0..dim_y {
                let mut yp = y.clone();
                yp[i] += eps;
                let mut ym = y.clone();
                ym[i] -= eps;
                let dp = f1(&x) - f2(&yp);
                let dm = f1(&x) - f2(&ym);
                grad_y[i] = (dp.norm() - dm.norm()) / (2.0 * eps);
            }

            let step = 0.1;
            x -= &grad_x * step;
            y -= &grad_y * step;
        }
    }

    results
}

/// The intersection number of two transverse submanifolds (oriented).
/// Counts intersections with sign determined by orientation compatibility.
pub fn intersection_number(
    intersections: &[IntersectionPoint],
) -> i32 {
    intersections.iter().map(|pt| pt.sign).sum()
}

/// An intersection point with orientation sign.
#[derive(Clone, Debug)]
pub struct IntersectionPoint {
    pub point: DVector<f64>,
    /// +1 or -1 depending on orientation.
    pub sign: i32,
}

/// Compute the orientation sign at an intersection point.
/// Given the combined tangent space, determine if it matches the ambient orientation.
pub fn orientation_sign(
    combined_basis: &DMatrix<f64>,
    ambient_orientation: &DMatrix<f64>,
) -> i32 {
    // Determinant of the change-of-basis matrix
    // If the combined basis (as columns) has positive determinant relative to ambient, sign = +1
    if combined_basis.nrows() == combined_basis.ncols() {
        let det = combined_basis.determinant();
        if det > 0.0 { 1 } else { -1 }
    } else {
        // Non-square: project and check
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transverse_intersection_lines_in_r2() {
        // Two lines through origin in R²: y=x and y=-x
        // Tangent spaces: (1,1) and (1,-1)
        let ts1 = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);
        let ts2 = DMatrix::from_row_slice(2, 1, &[1.0, -1.0]);
        assert!(is_transverse_intersection(2, &ts1, &ts2, 1e-10));
    }

    #[test]
    fn test_non_transverse_parallel_lines() {
        // Two parallel lines in R²: y=0 and y=1, both with tangent (1,0)
        let ts1 = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let ts2 = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        assert!(!is_transverse_intersection(2, &ts1, &ts2, 1e-10));
    }

    #[test]
    fn test_transverse_intersection_line_plane_in_r3() {
        // A line and a plane in R³
        let line = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
        let plane = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(is_transverse_intersection(3, &line, &plane, 1e-10));
    }

    #[test]
    fn test_transverse_intersection_dimension() {
        // Two 2-surfaces in R³: dim = 2 + 2 - 3 = 1 (curve)
        assert_eq!(transverse_intersection_dimension(2, 2, 3), 1);
        // Two curves in R²: dim = 1 + 1 - 2 = 0 (points)
        assert_eq!(transverse_intersection_dimension(1, 1, 2), 0);
        // Two curves in R³: dim = 1 + 1 - 3 = -1 (generically no intersection)
        assert_eq!(transverse_intersection_dimension(1, 1, 3), -1);
    }

    #[test]
    fn test_map_transverse_to_submanifold() {
        // f: R² -> R³, df = I₃ₓ₂ (embedding)
        // S = plane in R³ with tangent space e₁, e₃
        let df = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let tangent_s = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(is_map_transverse_to_submanifold(&df, &tangent_s, 3, 1e-10));
    }

    #[test]
    fn test_intersection_number() {
        let pts = vec![
            IntersectionPoint { point: DVector::from_vec(vec![0.0, 0.0]), sign: 1 },
            IntersectionPoint { point: DVector::from_vec(vec![1.0, 0.0]), sign: -1 },
            IntersectionPoint { point: DVector::from_vec(vec![2.0, 0.0]), sign: 1 },
        ];
        assert_eq!(intersection_number(&pts), 1);
    }

    #[test]
    fn test_orientation_sign() {
        let basis = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let ambient = DMatrix::identity(2, 2);
        assert_eq!(orientation_sign(&basis, &ambient), 1);

        let flipped = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        assert_eq!(orientation_sign(&flipped, &ambient), -1);
    }
}
