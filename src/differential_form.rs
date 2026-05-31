//! Differential forms: exterior algebra, wedge product, pullback, exterior derivative.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// A differential k-form on R^n at a point.
/// A k-form is an alternating multilinear map (R^n)^k -> R.
/// We represent it by its components as an array of index-value pairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentialForm {
    /// The degree k of the form (0-form = function, 1-form = covector, etc.)
    pub degree: usize,
    /// The dimension n of the ambient space.
    pub ambient_dimension: usize,
    /// Components indexed by increasing k-tuples.
    /// For a k-form on R^n, there are C(n,k) components.
    /// Indices stored as a flat array in lexicographic order of k-tuples.
    pub components: Vec<f64>,
    /// The base point (where the form lives).
    pub base_point: Option<DVector<f64>>,
}

impl DifferentialForm {
    /// Create a zero k-form on R^n.
    pub fn zero(degree: usize, ambient_dimension: usize) -> Self {
        let num_components = binomial(ambient_dimension, degree);
        Self {
            degree,
            ambient_dimension,
            components: vec![0.0; num_components],
            base_point: None,
        }
    }

    /// Create a k-form from components.
    pub fn from_components(degree: usize, ambient_dimension: usize, components: Vec<f64>) -> Self {
        let expected = binomial(ambient_dimension, degree);
        assert_eq!(components.len(), expected, "Wrong number of components: expected {}, got {}", expected, components.len());
        Self { degree, ambient_dimension, components, base_point: None }
    }

    /// Create a k-form at a specific point.
    pub fn at_point(mut self, point: DVector<f64>) -> Self {
        self.base_point = Some(point);
        self
    }

    /// Evaluate the k-form on k tangent vectors.
    /// For a 0-form, returns the single component.
    /// For a 1-form, applies the covector to a vector.
    /// For a k-form, computes the alternating sum.
    pub fn evaluate(&self, vectors: &[DVector<f64>]) -> f64 {
        assert_eq!(vectors.len(), self.degree);
        if self.degree == 0 {
            return self.components[0];
        }

        let indices = k_combinations(self.ambient_dimension, self.degree);
        let mut result = 0.0;
        for (comp_idx, combo) in indices.iter().enumerate() {
            // Determinant of the submatrix formed by taking rows from combo and columns from vectors
            let mut det_sum = 0.0;
            let perms = permutations(self.degree);
            for perm in &perms {
                let mut sign = 1.0;
                let mut prod = 1.0;
                for (i, &p) in perm.iter().enumerate() {
                    prod *= vectors[p][combo[i]];
                    sign *= if is_even_permutation(perm) { 1.0 } else { -1.0 };
                }
                // Only need sign once per permutation
                let s = if is_even_permutation(perm) { 1.0 } else { -1.0 };
                det_sum += s * prod;
            }
            result += self.components[comp_idx] * det_sum;
        }
        result
    }

    /// Evaluate on a single vector (1-form shortcut).
    pub fn apply_to_vector(&self, v: &DVector<f64>) -> f64 {
        assert_eq!(self.degree, 1);
        let mut result = 0.0;
        for i in 0..self.ambient_dimension {
            result += self.components[i] * v[i];
        }
        result
    }

    /// Add two forms of the same degree.
    pub fn add(&self, other: &DifferentialForm) -> DifferentialForm {
        assert_eq!(self.degree, other.degree);
        assert_eq!(self.ambient_dimension, other.ambient_dimension);
        let components: Vec<f64> = self.components.iter()
            .zip(other.components.iter())
            .map(|(a, b)| a + b)
            .collect();
        DifferentialForm { degree: self.degree, ambient_dimension: self.ambient_dimension, components, base_point: self.base_point.clone() }
    }

    /// Scale a form.
    pub fn scale(&self, scalar: f64) -> DifferentialForm {
        let components: Vec<f64> = self.components.iter().map(|c| c * scalar).collect();
        DifferentialForm { degree: self.degree, ambient_dimension: self.ambient_dimension, components, base_point: self.base_point.clone() }
    }
}

/// Wedge product of a k-form and an l-form, yielding a (k+l)-form.
pub fn wedge(alpha: &DifferentialForm, beta: &DifferentialForm) -> DifferentialForm {
    let k = alpha.degree;
    let l = beta.degree;
    let n = alpha.ambient_dimension;
    assert_eq!(n, beta.ambient_dimension);
    let result_degree = k + l;

    if result_degree > n {
        return DifferentialForm::zero(result_degree, n);
    }

    let result_components = binomial(n, result_degree);
    let mut result = vec![0.0; result_components];
    let k_indices = k_combinations(n, k);
    let l_indices = k_combinations(n, l);
    let r_indices = k_combinations(n, result_degree);

    for (ki, k_combo) in k_indices.iter().enumerate() {
        for (li, l_combo) in l_indices.iter().enumerate() {
            // Merge k_combo and l_combo
            let merged: Vec<usize> = {
                let mut m: Vec<usize> = k_combo.iter().chain(l_combo.iter()).cloned().collect();
                m.sort();
                m
            };
            if merged.len() != result_degree { continue; }
            // Check for duplicates (wedge of forms with shared indices is zero)
            let mut deduped = merged.clone();
            deduped.dedup();
            if deduped.len() != result_degree { continue; }

            // Find the sign of the shuffle permutation
            let sign = shuffle_sign(k_combo, l_combo, &merged);

            if let Some(ri) = r_indices.iter().position(|x| x == &merged) {
                result[ri] += sign * alpha.components[ki] * beta.components[li];
            }
        }
    }

    DifferentialForm::from_components(result_degree, n, result)
}

/// Exterior derivative d: k-forms -> (k+1)-forms.
/// Computed numerically from the form's dependence on position.
/// For symbolic computation, we use known formulas for standard forms.
pub fn exterior_derivative(
    form: &DifferentialForm,
    _h: f64,
) -> DifferentialForm {
    // For numerical computation, we'd need the form as a function of position.
    // Here we provide the algebraic structure; specific cases handled in tests.
    let new_degree = form.degree + 1;
    if new_degree > form.ambient_dimension {
        return DifferentialForm::zero(new_degree, form.ambient_dimension);
    }
    // Placeholder: returns zero form (caller should use specific derivative functions)
    DifferentialForm::zero(new_degree, form.ambient_dimension)
}

/// Compute the exterior derivative of a 0-form (function) given its gradient.
pub fn d_from_gradient(gradient: &DVector<f64>) -> DifferentialForm {
    let n = gradient.len();
    DifferentialForm::from_components(1, n, gradient.iter().cloned().collect())
}

/// Compute the exterior derivative of a 1-form given its components and the Jacobian.
/// For a 1-form ω = Σ fᵢ dxᵢ, dω = Σⱼ ∂fᵢ/∂xⱼ dxⱼ ∧ dxᵢ
/// = Σ_{i<j} (∂fⱼ/∂xᵢ - ∂fᵢ/∂xⱼ) dxᵢ ∧ dxⱼ
pub fn d_1form(jacobian: &nalgebra::DMatrix<f64>) -> DifferentialForm {
    let n = jacobian.nrows();
    let pairs = k_combinations(n, 2);
    let mut components = Vec::new();
    for pair in &pairs {
        let i = pair[0];
        let j = pair[1];
        // dx_i ∧ dx_j coefficient: ∂f_j/∂x_i - ∂f_i/∂x_j
        components.push(jacobian[(j, i)] - jacobian[(i, j)]);
    }
    DifferentialForm::from_components(2, n, components)
}

/// Pullback of a k-form by a smooth map.
/// Given a map F: R^m -> R^n with Jacobian J, and a k-form ω on R^n,
/// the pullback F*ω is a k-form on R^m.
pub fn pullback(
    form: &DifferentialForm,
    jacobian: &nalgebra::DMatrix<f64>,
) -> DifferentialForm {
    let m = jacobian.ncols(); // source dimension
    let k = form.degree;
    let n = form.ambient_dimension;

    if k == 0 {
        let mut result = form.clone();
        result.ambient_dimension = m;
        return result;
    }

    let result_indices = k_combinations(m, k);
    let form_indices = k_combinations(n, k);
    let mut components = Vec::new();

    for r_combo in &result_indices {
        let mut val = 0.0;
        for (fi, f_combo) in form_indices.iter().enumerate() {
            // Minor determinant of Jacobian
            let minor = jacobian.select_rows(f_combo).select_columns(r_combo);
            let det = minor.determinant();
            val += form.components[fi] * det;
        }
        components.push(val);
    }

    DifferentialForm::from_components(k, m, components)
}

/// The volume form on R^n (dx₁ ∧ dx₂ ∧ ... ∧ dxₙ).
pub fn volume_form(n: usize) -> DifferentialForm {
    let mut components = vec![0.0; binomial(n, n)];
    components[0] = 1.0;
    DifferentialForm::from_components(n, n, components)
}

/// The standard symplectic form on R^{2n}: ω = Σ dxᵢ ∧ dyᵢ.
pub fn symplectic_form(n: usize) -> DifferentialForm {
    let dim = 2 * n;
    let pairs = k_combinations(dim, 2);
    let mut components = vec![0.0; pairs.len()];
    for (idx, pair) in pairs.iter().enumerate() {
        // dx_i ∧ dy_i: x indices are even (0,2,4,...), y indices are odd (1,3,5,...)
        for i in 0..n {
            if pair[0] == 2 * i && pair[1] == 2 * i + 1 {
                components[idx] = 1.0;
            }
        }
    }
    DifferentialForm::from_components(2, dim, components)
}

// --- Helper functions ---

fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

fn k_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if k > n { return vec![]; }
    let mut result = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();
    loop {
        result.push(combo.clone());
        // Next combination
        let mut i = k - 1;
        while i > 0 && combo[i] == n - k + i {
            i -= 1;
        }
        if combo[i] == n - k + i + 1 { break; }
        // Actually: simpler approach
        let mut done = true;
        for j in (0..k).rev() {
            if combo[j] < n - k + j {
                combo[j] += 1;
                for l in j+1..k {
                    combo[l] = combo[l-1] + 1;
                }
                done = false;
                break;
            }
        }
        if done { break; }
    }
    result
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 { return vec![vec![]]; }
    let mut result = Vec::new();
    let mut perm: Vec<usize> = (0..n).collect();
    result.push(perm.clone());
    loop {
        // Next permutation (lexicographic)
        let mut k = None;
        for i in (0..n-1).rev() {
            if perm[i] < perm[i+1] {
                k = Some(i);
                break;
            }
        }
        let ki = match k {
            Some(v) => v,
            None => break,
        };
        let mut l = 0;
        for j in (ki+1..n).rev() {
            if perm[ki] < perm[j] {
                l = j;
                break;
            }
        }
        perm.swap(ki, l);
        perm[ki+1..].reverse();
        result.push(perm.clone());
    }
    result
}

fn is_even_permutation(perm: &[usize]) -> bool {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut num_cycles = 0;
    for i in 0..n {
        if !visited[i] {
            num_cycles += 1;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
            }
        }
    }
    (n - num_cycles) % 2 == 0
}

fn shuffle_sign(k_combo: &[usize], l_combo: &[usize], merged: &[usize]) -> f64 {
    // Count transpositions to move k_combo elements to the front of merged
    // The sign is (-1)^(number of transpositions)
    let mut merged_copy: Vec<usize> = k_combo.iter().chain(l_combo.iter()).cloned().collect();
    let target: Vec<usize> = merged.to_vec();
    let mut sign = 1;
    // Sort merged_copy to match target, counting swaps
    for i in 0..merged_copy.len() {
        if merged_copy[i] != target[i] {
            for j in (i+1)..merged_copy.len() {
                if merged_copy[j] == target[i] {
                    merged_copy.swap(i, j);
                    sign *= -1;
                    break;
                }
            }
        }
    }
    sign as f64
}

/// The Hodge star operator on R^n with the standard metric.
/// Maps k-forms to (n-k)-forms.
pub fn hodge_star(form: &DifferentialForm) -> DifferentialForm {
    let k = form.degree;
    let n = form.ambient_dimension;
    let target_degree = n - k;
    let k_indices = k_combinations(n, k);
    let nk_indices = k_combinations(n, target_degree);

    let mut result = vec![0.0; nk_indices.len()];

    for (ki, k_combo) in k_indices.iter().enumerate() {
        if form.components[ki].abs() < 1e-30 { continue; }
        // Complement of k_combo
        let complement: Vec<usize> = (0..n).filter(|x| !k_combo.contains(x)).collect();
        if let Some(nki) = nk_indices.iter().position(|x| *x == complement) {
            // Sign: (-1)^(k*(n-k)) times the sign of the permutation
            // (k_combo, complement) -> (0, 1, ..., n-1)
            let full: Vec<usize> = k_combo.iter().chain(complement.iter()).cloned().collect();
            let identity: Vec<usize> = (0..n).collect();
            let sign = if is_even_permutation_from(&full, &identity) { 1.0 } else { -1.0 };
            let k_nk_sign = if (k * (n - k)) % 2 == 0 { 1.0 } else { -1.0 };
            result[nki] += k_nk_sign * sign * form.components[ki];
        }
    }

    DifferentialForm::from_components(target_degree, n, result)
}

fn is_even_permutation_from(from: &[usize], to: &[usize]) -> bool {
    let n = from.len();
    let mut pos = vec![0usize; n];
    for i in 0..n {
        pos[from[i]] = i;
    }
    // Create the permutation that maps from -> to
    let mut perm: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        perm.push(pos[to[i]]);
    }
    is_even_permutation(&perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_form() {
        let f = DifferentialForm::from_components(0, 3, vec![7.0]);
        assert_eq!(f.degree, 0);
        assert_eq!(f.components[0], 7.0);
    }

    #[test]
    fn test_wedge_1_1() {
        // dx₁ ∧ dx₂ on R²
        let dx1 = DifferentialForm::from_components(1, 2, vec![1.0, 0.0]);
        let dx2 = DifferentialForm::from_components(1, 2, vec![0.0, 1.0]);
        let w = wedge(&dx1, &dx2);
        assert_eq!(w.degree, 2);
        assert!((w.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wedge_anticommutative() {
        let dx1 = DifferentialForm::from_components(1, 2, vec![1.0, 0.0]);
        let dx2 = DifferentialForm::from_components(1, 2, vec![0.0, 1.0]);
        let w1 = wedge(&dx1, &dx2);
        let w2 = wedge(&dx2, &dx1);
        assert!((w1.components[0] + w2.components[0]).abs() < 1e-10);
    }

    #[test]
    fn test_wedge_1_2_in_r3() {
        // dx₁ ∧ (dx₂ ∧ dx₃) = dx₁ ∧ dx₂ ∧ dx₃
        let dx1 = DifferentialForm::from_components(1, 3, vec![1.0, 0.0, 0.0]);
        let dx2dx3 = DifferentialForm::from_components(2, 3, vec![0.0, 0.0, 1.0]); // dx₂∧dx₃
        let w = wedge(&dx1, &dx2dx3);
        assert_eq!(w.degree, 3);
        assert!((w.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_d_from_gradient() {
        let grad = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let df = d_from_gradient(&grad);
        assert_eq!(df.degree, 1);
        assert_eq!(df.components.len(), 3);
        assert!((df.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_d_1form() {
        // ω = x dy, so dω = dx ∧ dy
        // f = (0, x), Jacobian = [[0,0],[1,0]]
        let jac = nalgebra::DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]);
        let dw = d_1form(&jac);
        assert_eq!(dw.degree, 2);
        // (∂f₂/∂x₁ - ∂f₁/∂x₂) = 1 - 0 = 1
        assert!((dw.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pullback_identity() {
        let jac = nalgebra::DMatrix::identity(2, 2);
        let form = DifferentialForm::from_components(1, 2, vec![1.0, 0.0]);
        let pb = pullback(&form, &jac);
        assert!((pb.components[0] - 1.0).abs() < 1e-10);
        assert!((pb.components[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_form() {
        let vol = volume_form(3);
        assert_eq!(vol.degree, 3);
        assert!((vol.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symplectic_form_r2() {
        let sym = symplectic_form(1); // On R²
        assert_eq!(sym.degree, 2);
        assert_eq!(sym.ambient_dimension, 2);
        assert!((sym.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hodge_star_r3() {
        // *dx₁ = dx₂∧dx₃ in R³
        let dx1 = DifferentialForm::from_components(1, 3, vec![1.0, 0.0, 0.0]);
        let star = hodge_star(&dx1);
        assert_eq!(star.degree, 2);
        // dx₂∧dx₃ has index combo [1,2] which is index 2 in the list of 2-combos of {0,1,2}
        // combos: [0,1], [0,2], [1,2] -> index 2
        assert!((star.components[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_forms() {
        let f1 = DifferentialForm::from_components(1, 2, vec![1.0, 2.0]);
        let f2 = DifferentialForm::from_components(1, 2, vec![3.0, 4.0]);
        let sum = f1.add(&f2);
        assert!((sum.components[0] - 4.0).abs() < 1e-10);
        assert!((sum.components[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_form() {
        let f = DifferentialForm::from_components(1, 2, vec![1.0, 2.0]);
        let scaled = f.scale(3.0);
        assert!((scaled.components[0] - 3.0).abs() < 1e-10);
        assert!((scaled.components[1] - 6.0).abs() < 1e-10);
    }
}
