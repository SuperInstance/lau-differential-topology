//! # lau-differential-topology
//!
//! A library for differential topology: smooth manifolds, tangent bundles,
//! differential forms, Stokes' theorem, transversality, degree theory,
//! and Euler characteristic via Poincaré-Hopf.

pub mod smooth_manifold;
pub mod tangent;
pub mod vector_field;
pub mod differential_form;
pub mod stokes;
pub mod transversality;
pub mod degree;
pub mod euler;
pub mod agent_dynamics;

pub use smooth_manifold::*;
pub use tangent::*;
pub use vector_field::*;
pub use differential_form::*;
pub use stokes::*;
pub use transversality::*;
pub use degree::*;
pub use euler::*;
pub use agent_dynamics::*;
