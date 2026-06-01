# lau-differential-topology

A Rust library for **differential topology**: smooth manifolds, tangent bundles, differential forms, Stokes' theorem, transversality, degree theory, and the Euler characteristic via Poincaré–Hopf.

Built on `nalgebra` for linear algebra and `serde` for serialization.

---

## What This Does

This crate provides the building blocks of differential topology as composable Rust types:

| Module | What you get |
|---|---|
| `smooth_manifold` | Charts, atlases, transition maps, and pre-built manifolds (Sⁿ, T², Rⁿ) |
| `tangent` | Tangent vectors, tangent/cotangent spaces, tangent bundles, pushforwards |
| `vector_field` | Smooth vector fields, integral curves (RK4), equilibrium classification, divergence, curl |
| `differential_form` | k-forms, wedge product, exterior derivative, pullback, Hodge star, volume/symplectic forms |
| `stokes` | Integration of forms over parametrized manifolds, line integrals, surface integrals, Stokes/Green/divergence theorem verification |
| `transversality` | Transverse intersection checking, intersection dimension, intersection number with orientation signs |
| `degree` | Brouwer degree, winding number, antipodal/reflection degree, local degree, homotopy invariance |
| `euler` | Euler characteristic (Poincaré–Hopf, triangulation, CW-complex, Morse theory), Gauss–Bonnet verification |
| `agent_dynamics` | Multi-agent systems on manifolds, flow analysis, Poincaré return maps, Lyapunov exponents |

---

## Key Idea

Differential topology studies the global structure of smooth manifolds through smooth maps, vector fields, and differential forms. The central results—Stokes' theorem, the Poincaré–Hopf theorem, degree theory—connect local differential data to global topological invariants.

This library makes those objects and theorems **computable**: you construct manifolds, define vector fields and forms as Rust closures, and then integrate, classify, or verify results numerically.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-differential-topology = { git = "https://github.com/SuperInstance/lau-differential-topology" }
```

Or via crates.io (if published):

```toml
[dependencies]
lau-differential-topology = "0.1"
```

Requires Rust 2021 edition.

---

## Quick Start

### Build a manifold and inspect its atlas

```rust
use lau_differential_topology::sphere_manifold_2d;

let s2 = sphere_manifold_2d(); // S² with two stereographic charts
assert_eq!(s2.dimension(), 2);
assert_eq!(s2.atlas.charts.len(), 2);
```

### Define a vector field, find equilibria, classify them

```rust
use lau_differential_topology::{VectorField, classify_equilibrium};
use nalgebra::DVector;

// V(x,y) = (x, -y) — saddle at origin
let vf = VectorField::new("saddle", 2, |p: &DVector<f64>| {
    DVector::from_vec(vec![p[0], -p[1]])
});

let eqs = vf.find_equilibria_grid(&[(-2.0, 2.0), (-2.0, 2.0)], 30);
for eq in &eqs {
    let jac = /* numerical jacobian or analytic */;
    println!("{:?} at {:?}", classify_equilibrium(&jac), eq);
}
```

### Compute a line integral (verify Green's theorem)

```rust
use lau_differential_topology::line_integral;
use nalgebra::DVector;

// ∫_∂D (−y dx + x dy) around the unit circle = 2π
let result = line_integral(
    &|p: &DVector<f64>| DVector::from_vec(vec![-p[1], p[0]]),
    &|t: f64| DVector::from_vec(vec![t.cos(), t.sin()]),
    &|t: f64| DVector::from_vec(vec![-t.sin(), t.cos()]),
    0.0, 2.0 * std::f64::consts::PI, 1000,
);
assert!((result - 2.0 * std::f64::consts::PI).abs() < 0.01);
```

### Compute winding number and Brouwer degree

```rust
use lau_differential_topology::{winding_number, brouwer_degree};
use nalgebra::{DVector, DMatrix};

let wn = winding_number(
    &|t| DVector::from_vec(vec![t.cos(), t.sin()]),
    &|t| DVector::from_vec(vec![-t.sin(), t.cos()]),
    &DVector::from_vec(vec![0.0, 0.0]),
    1000,
);
assert!((wn - 1.0).abs() < 0.01);

// f(z) = z² on S¹ has degree 2
let deg = brouwer_degree(
    &|p| { /* z² normalized */ },
    &|p| { /* Jacobian */ },
    &DVector::from_vec(vec![1.0, 0.0]),
    &vec![DVector::from_vec(vec![1.0, 0.0]), DVector::from_vec(vec![-1.0, 0.0])],
);
assert_eq!(deg, 2);
```

### Euler characteristic via Poincaré–Hopf

```rust
use lau_differential_topology::{euler_characteristic_poincare_hopf, euler_characteristic_sphere};
use nalgebra::DMatrix;

// S²: two zeros of the height field, each with index +1
let chi = euler_characteristic_poincare_hopf(&[
    DMatrix::identity(2, 2), // index +1 at south pole
    DMatrix::identity(2, 2), // index +1 at north pole
]);
assert_eq!(chi, 2);
assert_eq!(euler_characteristic_sphere(2), 2);
```

### Run a multi-agent system on a vector field

```rust
use lau_differential_topology::{VectorField, AgentSystem};
use nalgebra::DVector;

let vf = VectorField::new("swirl", 2, |p: &DVector<f64>| {
    DVector::from_vec(vec![-p[1], p[0]]) // rotational
});

let mut sys = AgentSystem::new(vf);
sys.add_agent(DVector::from_vec(vec![1.0, 0.0]));
sys.add_agent(DVector::from_vec(vec![0.0, 1.0]));

for _ in 0..100 {
    sys.step_rk4(0.01);
}

let clusters = sys.find_clusters(0.5);
println!("{} clusters formed", clusters.len());
```

---

## API Reference

### `smooth_manifold`

| Type / Function | Description |
|---|---|
| `Chart` | A coordinate chart (dimension + name). |
| `TransitionMap` | A smooth transition φⱼ ∘ φᵢ⁻¹ with Jacobian closure. |
| `Atlas` | Collection of charts + transitions; `add_chart()`, `add_transition()`, `transition_jacobian()`. |
| `SmoothManifold` | Manifold = atlas + name; `dimension()`. |
| `circle_manifold()` | S¹ with two overlapping angle charts. |
| `sphere_manifold_2d()` | S² with stereographic north/south charts. |
| `torus_manifold()` | T² with flat angle charts. |
| `n_sphere_manifold(n)` | Sⁿ with stereographic charts (exact Jacobians for n ≤ 3, identity fallback). |
| `euclidean_manifold(n)` | Rⁿ with single identity chart. |

### `tangent`

| Type / Function | Description |
|---|---|
| `TangentVector` | Tangent vector at a base point: `add()`, `scale()`, `inner_product()`, `norm()`. |
| `TangentSpace` | Tangent space at a point, optional Riemannian metric: `inner_product(v, w)`, `zero()`. |
| `TangentBundle` | Tangent bundle TM: `element()`, `projection()`, `fiber()`. |
| `CotangentVector` | Covector (dual to tangent): `apply(v)`. |
| `Differential` | Pushforward of a smooth map: `push_forward(v)`. |

### `vector_field`

| Type / Function | Description |
|---|---|
| `VectorField` | Smooth V: Rⁿ → Rⁿ as a closure: `evaluate()`, `find_equilibria_grid()`. |
| `EquilibriumType` | `Stable`, `Unstable`, `Saddle`, `Center`, `Degenerate`. |
| `classify_equilibrium(jac)` | Classify equilibrium from Jacobian eigenvalues. |
| `integral_curve(vf, start, dt, steps)` | RK4 integration of an integral curve. |
| `divergence(vf, p, h)` | Numerical divergence. |
| `curl_2d(vf, p, h)` | Numerical scalar curl in 2D. |
| `constant_vector_field`, `radial_vector_field`, `rotational_vector_field` | Predefined fields. |

### `differential_form`

| Type / Function | Description |
|---|---|
| `DifferentialForm` | A k-form on Rⁿ: `evaluate()`, `apply_to_vector()`, `add()`, `scale()`. |
| `wedge(α, β)` | Wedge product (k-form) ∧ (l-form) → (k+l)-form. |
| `d_from_gradient(∇f)` | Exterior derivative of a 0-form. |
| `d_1form(J)` | Exterior derivative of a 1-form from its Jacobian. |
| `pullback(ω, J)` | Pullback of a k-form by a smooth map with Jacobian J. |
| `hodge_star(ω)` | Hodge star on Rⁿ with standard metric. |
| `volume_form(n)`, `symplectic_form(n)` | Standard volume and symplectic forms. |

### `stokes`

| Type / Function | Description |
|---|---|
| `integrate_form(form, manifold, res)` | Integrate a k-form over a parametrized k-manifold. |
| `integrate_1d(f, a, b, n)` | 1D midpoint rule. |
| `integrate_2d(f, …)` | 2D midpoint rule. |
| `line_integral(form, curve, curve', a, b, n)` | Line integral of a 1-form. |
| `surface_area(surface, ∂u, ∂v, …)` | Surface area of a parametrized surface. |
| `verify_stokes_1form(…)` | Check ∫_∂M ω ≈ ∫_M dω within tolerance. |

### `transversality`

| Type / Function | Description |
|---|---|
| `is_transverse_intersection(dim, T₁, T₂, tol)` | Check if two submanifolds intersect transversally. |
| `transverse_intersection_dimension(d₁, d₂, d_M)` | dim(N₁ ∩ N₂) = d₁ + d₂ − d_M. |
| `is_map_transverse_to_submanifold(df, T_S, dim_N, tol)` | Check if a map is transverse to a submanifold. |
| `find_intersections(f₁, f₂, guesses, tol, iter)` | Newton-like intersection finder. |
| `intersection_number(pts)` | Signed count of oriented intersection points. |

### `degree`

| Type / Function | Description |
|---|---|
| `brouwer_degree(f, df, rv, preimages)` | Brouwer degree = Σ sign(det(dfᵢ)) over preimages. |
| `winding_number(γ, γ', z, n)` | Winding number of a closed curve around a point. |
| `local_degree(J)` | Sign of det(J) (+1, −1, or 0). |
| `antipodal_degree(n)`, `identity_degree()`, `reflection_degree()` | Named map degrees. |
| `verify_homotopy_invariance(deg₀, deg₁)` | Assert two degrees are equal (homotopy invariant). |

### `euler`

| Type / Function | Description |
|---|---|
| `index_of_zero(J)` | Index of an isolated zero (sign of det). |
| `euler_characteristic_poincare_hopf(jacobians)` | χ = Σ index(pᵢ). |
| `euler_characteristic_sphere(n)` | χ(Sⁿ): 2 if n even, 0 if odd. |
| `euler_characteristic_torus()` | χ(T²) = 0. |
| `euler_characteristic_surface_genus(g)` | χ = 2 − 2g. |
| `euler_characteristic_from_triangulation(V, E, F)` | χ = V − E + F. |
| `euler_characteristic_cw(counts)` | χ = Σ (−1)ᵏ nₖ. |
| `euler_characteristic_morse(counts)` | χ = Σ (−1)ᵏ mₖ. |
| `verify_gauss_bonnet(∫K dA, χ, tol)` | Check ∫ K dA = 2πχ. |

### `agent_dynamics`

| Type / Function | Description |
|---|---|
| `Agent` | Agent with position/velocity: `step()`, `step_rk4()`, `distance_to()`. |
| `AgentSystem` | Multi-agent system: `add_agent()`, `step()`, `step_rk4()`, `find_clusters()`. |
| `FlowAnalysis` | Result of `analyze_flow()`: equilibria, classifications, trajectories. |
| `DynamicsType` | `Convergent`, `Divergent`, `Oscillatory`, `Mixed`, `GradientLike`. |
| `analyze_flow(vf, bounds, …)` | Full flow analysis: find equilibria, classify, generate trajectories. |
| `numerical_jacobian(vf, p, h)` | Finite-difference Jacobian. |
| `poincare_return(vf, start, …)` | Poincaré section return map. |
| `lyapunov_exponent(traj, dt)` | Estimate Lyapunov exponent from a trajectory. |

---

## How It Works

The library is organized in layers, mirroring the mathematical progression:

1. **Manifolds** (`smooth_manifold`): A `SmoothManifold` holds an `Atlas` of `Chart`s connected by `TransitionMap`s. Transition Jacobians are stored as closures, so you can encode arbitrary coordinate changes. Pre-built factories (`sphere_manifold_2d`, `torus_manifold`, etc.) give you standard examples immediately.

2. **Tangent structures** (`tangent`): At each point on a manifold, the `TangentSpace` provides a vector space (optionally equipped with a Riemannian metric tensor). The `TangentBundle` collects all tangent spaces. `Differential` implements the pushforward (Jacobian applied to tangent vectors).

3. **Vector fields** (`vector_field`): A `VectorField` wraps a closure `Rⁿ → Rⁿ`. Equilibria are found via grid search; classification uses eigenvalue signs from the symmetric eigendecomposition of the Jacobian. Integral curves are computed with RK4. Divergence and curl use centered finite differences.

4. **Differential forms** (`differential_form`): k-forms are represented as coefficient arrays indexed by lexicographically ordered k-combinations. The `wedge` product merges index sets with shuffle-sign corrections. `pullback` computes minor determinants of the Jacobian. The `hodge_star` maps k-forms to (n−k)-forms using complement indices and permutation signs.

5. **Integration & Stokes** (`stokes`): Forms are integrated by pulling them back through parametrizations and summing over a midpoint-rule grid. `line_integral` computes ∫ ω(γ′(t)) dt. Stokes' theorem is verified numerically by comparing boundary and interior integrals.

6. **Transversality & Degree** (`transversality`, `degree`): Transverse intersection is checked by computing the rank of the combined tangent-space matrix (via SVD). The Brouwer degree counts preimages with sign. The winding number is computed as a line integral of the angular form.

7. **Euler characteristic** (`euler`): Multiple roads to χ: Poincaré–Hopf (sum of zero indices), triangulation (V − E + F), CW decomposition, Morse theory, and Gauss–Bonnet verification.

8. **Agent dynamics** (`agent_dynamics`): An applied layer that puts agents on manifolds, drives them with vector fields (Euler or RK4), and provides tools for flow analysis, clustering, Poincaré maps, and Lyapunov exponent estimation.

---

## The Math

### Smooth Manifolds
An n-dimensional smooth manifold M is a topological space locally homeomorphic to Rⁿ, with smooth (C^∞) transition maps between overlapping charts. An atlas {φᵢ: Uᵢ → Rⁿ} covers M, and the transition maps φⱼ ∘ φᵢ⁻¹ must be smooth with invertible Jacobian.

### Tangent Bundle
The tangent space T_pM at p ∈ M is an n-dimensional vector space of directional derivatives. The tangent bundle TM = {(p, v) : p ∈ M, v ∈ T_pM} is a 2n-dimensional manifold. A smooth map f: M → N induces a linear pushforward df_p: T_pM → T_{f(p)}N (the Jacobian).

### Differential Forms
A differential k-form ω at p is an alternating multilinear map (T_pM)ᵏ → R. Forms support the wedge product (α ∧ β is (k+l)-alternating), the exterior derivative d (which raises degree by 1 and satisfies d² = 0), and pullback by smooth maps.

### Stokes' Theorem
For a compact oriented (k+1)-manifold M with boundary ∂M and a smooth k-form ω:

    ∫_∂M ω = ∫_M dω

This generalizes the fundamental theorem of calculus, Green's theorem, and the divergence theorem.

### Transversality
Two submanifolds N₁, N₂ ⊂ M intersect transversally at p if T_pN₁ + T_pN₂ = T_pM. Transverse intersections are stable under perturbation. The dimension formula: dim(N₁ ∩ N₂) = dim N₁ + dim N₂ − dim M.

### Degree Theory
For a smooth map f: M → N between compact oriented n-manifolds, the Brouwer degree counts (with sign) the preimages of a regular value:

    deg(f) = Σ_{p ∈ f⁻¹(q)} sign(det(df_p))

The degree is a homotopy invariant: if f ≃ g, then deg(f) = deg(g).

### Euler Characteristic (Poincaré–Hopf)
For any vector field V on M with isolated zeros:

    χ(M) = Σ_p index_p(V)

where index_p(V) is the degree of the map V/|V|: S_ε^{n-1} → S^{n-1} around p. For S², χ = 2; for T², χ = 0.

### Gauss–Bonnet
For a closed oriented surface M with Gaussian curvature K:

    ∫_M K dA = 2π χ(M)

---

## License

MIT
