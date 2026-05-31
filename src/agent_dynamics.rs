//! Application: agent dynamics on manifolds — flow analysis, equilibrium classification.

use nalgebra::{DVector, DMatrix};
use crate::vector_field::{VectorField, integral_curve, classify_equilibrium, EquilibriumType};
use crate::tangent::{TangentVector, TangentSpace};
use crate::degree::winding_number;
use crate::euler::{euler_characteristic_poincare_hopf, index_of_zero};

/// An agent moving on a manifold, governed by a vector field.
#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub position: DVector<f64>,
    pub velocity: DVector<f64>,
}

impl Agent {
    pub fn new(id: usize, position: DVector<f64>) -> Self {
        let dim = position.len();
        Self { id, position, velocity: DVector::zeros(dim) }
    }

    /// Update the agent's position using Euler integration.
    pub fn step(&mut self, vf: &VectorField, dt: f64) {
        let tv = vf.evaluate(&self.position);
        self.velocity = tv.components.clone();
        self.position += &self.velocity * dt;
    }

    /// Update using RK4.
    pub fn step_rk4(&mut self, vf: &VectorField, dt: f64) {
        let f = &vf.f;
        let k1 = f(&self.position);
        let p2 = &self.position + &k1 * (dt / 2.0);
        let k2 = f(&p2);
        let p3 = &self.position + &k2 * (dt / 2.0);
        let k3 = f(&p3);
        let p4 = &self.position + &k3 * dt;
        let k4 = f(&p4);
        self.velocity = &k1 + &k2 * 2.0 + &k3 * 2.0 + &k4;
        self.velocity *= dt / 6.0;
        self.position += &self.velocity;
        // Recompute velocity at new position
        self.velocity = f(&self.position);
    }

    /// Distance from another agent.
    pub fn distance_to(&self, other: &Agent) -> f64 {
        (&self.position - &other.position).norm()
    }
}

/// A multi-agent system on a manifold.
pub struct AgentSystem {
    pub agents: Vec<Agent>,
    pub vector_field: VectorField,
    pub dimension: usize,
}

impl AgentSystem {
    pub fn new(vector_field: VectorField) -> Self {
        let dim = vector_field.dimension;
        Self { agents: Vec::new(), vector_field, dimension: dim }
    }

    pub fn add_agent(&mut self, position: DVector<f64>) -> usize {
        let id = self.agents.len();
        self.agents.push(Agent::new(id, position));
        id
    }

    /// Advance all agents by one time step.
    pub fn step(&mut self, dt: f64) {
        for agent in &mut self.agents {
            agent.step(&self.vector_field, dt);
        }
    }

    /// Advance all agents using RK4.
    pub fn step_rk4(&mut self, dt: f64) {
        for agent in &mut self.agents {
            agent.step_rk4(&self.vector_field, dt);
        }
    }

    /// Find clusters of agents (agents within a distance threshold).
    pub fn find_clusters(&self, threshold: f64) -> Vec<Vec<usize>> {
        let n = self.agents.len();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if visited[i] { continue; }
            let mut cluster = Vec::new();
            let mut stack = vec![i];
            while let Some(j) = stack.pop() {
                if visited[j] { continue; }
                visited[j] = true;
                cluster.push(j);
                for k in 0..n {
                    if !visited[k] && self.agents[j].distance_to(&self.agents[k]) < threshold {
                        stack.push(k);
                    }
                }
            }
            clusters.push(cluster);
        }
        clusters
    }
}

/// Analyze the flow of a vector field and classify the dynamics.
pub struct FlowAnalysis {
    /// Equilibrium points found.
    pub equilibria: Vec<DVector<f64>>,
    /// Classification of each equilibrium.
    pub classifications: Vec<EquilibriumType>,
    /// Sample trajectories.
    pub trajectories: Vec<Vec<DVector<f64>>>,
}

/// Perform a complete flow analysis of a vector field.
pub fn analyze_flow(
    vf: &VectorField,
    bounds: &[(f64, f64)],
    grid_resolution: usize,
    trajectory_dt: f64,
    trajectory_steps: usize,
    num_trajectories: usize,
) -> FlowAnalysis {
    // Find equilibria
    let equilibria = vf.find_equilibria_grid(bounds, grid_resolution);

    // Classify each equilibrium using numerical Jacobian
    let mut classifications = Vec::new();
    let h = 1e-5;
    for eq in &equilibria {
        let jac = numerical_jacobian(vf, eq, h);
        classifications.push(classify_equilibrium(&jac));
    }

    // Generate sample trajectories from grid of initial conditions
    let mut trajectories = Vec::new();
    let dim = vf.dimension;
    let samples_per_dim = (num_trajectories as f64).powf(1.0 / dim as f64).ceil() as usize;
    let actual_num = samples_per_dim.pow(dim as u32);

    for idx in 0..actual_num.min(num_trajectories) {
        let mut start = DVector::zeros(dim);
        let mut remaining = idx;
        for d in 0..dim {
            let (lo, hi) = bounds[d];
            let i = remaining % samples_per_dim;
            remaining /= samples_per_dim;
            start[d] = lo + (i as f64 + 0.5) * (hi - lo) / samples_per_dim as f64;
        }
        let trajectory = integral_curve(vf, &start, trajectory_dt, trajectory_steps);
        trajectories.push(trajectory);
    }

    FlowAnalysis { equilibria, classifications, trajectories }
}

/// Compute the numerical Jacobian of a vector field at a point.
pub fn numerical_jacobian(vf: &VectorField, point: &DVector<f64>, h: f64) -> DMatrix<f64> {
    let n = vf.dimension;
    let f0 = (vf.f)(point);
    let mut jac = DMatrix::zeros(n, n);
    for i in 0..n {
        let mut p_plus = point.clone();
        p_plus[i] += h;
        let f_plus = (vf.f)(&p_plus);
        for j in 0..n {
            jac[(j, i)] = (f_plus[j] - f0[j]) / h;
        }
    }
    jac
}

/// Classify the overall dynamics of a flow.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsType {
    /// All trajectories converge to a single point.
    Convergent,
    /// All trajectories diverge.
    Divergent,
    /// Trajectories orbit around equilibria.
    Oscillatory,
    /// Mix of behaviors.
    Mixed,
    /// Gradient-like (all trajectories converge to equilibria).
    GradientLike,
}

/// Classify overall dynamics from a flow analysis.
pub fn classify_dynamics(analysis: &FlowAnalysis) -> DynamicsType {
    if analysis.equilibria.is_empty() {
        return DynamicsType::Divergent;
    }

    let has_stable = analysis.classifications.iter().any(|c| *c == EquilibriumType::Stable);
    let has_unstable = analysis.classifications.iter().any(|c| *c == EquilibriumType::Unstable);
    let has_saddle = analysis.classifications.iter().any(|c| *c == EquilibriumType::Saddle);
    let has_center = analysis.classifications.iter().any(|c| *c == EquilibriumType::Center);

    if has_center && !has_stable && !has_unstable && !has_saddle {
        return DynamicsType::Oscillatory;
    }
    if has_stable && !has_unstable && !has_saddle {
        return DynamicsType::Convergent;
    }
    if has_unstable && !has_stable {
        return DynamicsType::Divergent;
    }
    DynamicsType::Mixed
}

/// Compute the Poincaré return map for a periodic orbit.
/// For a point near a periodic orbit, find the next intersection with a Poincaré section.
pub fn poincare_return(
    vf: &VectorField,
    start: &DVector<f64>,
    section_normal: &DVector<f64>,
    section_point: &DVector<f64>,
    dt: f64,
    max_steps: usize,
) -> Option<DVector<f64>> {
    let curve = integral_curve(vf, start, dt, max_steps);

    let mut prev_dist = (&curve[0] - section_point).dot(section_normal);

    for i in 1..curve.len() {
        let curr_dist = (&curve[i] - section_point).dot(section_normal);
        if prev_dist * curr_dist < 0.0 {
            // Crossed the section; linear interpolation
            let t = prev_dist.abs() / (prev_dist.abs() + curr_dist.abs());
            let result = &curve[i - 1] * (1.0 - t) + &curve[i] * t;
            return Some(result);
        }
        prev_dist = curr_dist;
    }
    None
}

/// Compute the Lyapunov exponent estimate from a trajectory.
pub fn lyapunov_exponent(
    trajectory: &[DVector<f64>],
    dt: f64,
) -> f64 {
    if trajectory.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 2..trajectory.len() {
        let d1 = (&trajectory[i] - &trajectory[i - 1]).norm();
        let d0 = (&trajectory[i - 1] - &trajectory[i - 2]).norm();
        if d0 > 1e-12 {
            sum += (d1 / d0).ln();
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / (count as f64 * dt) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_step() {
        let vf = VectorField::new("constant", 2, |_: &DVector<f64>| DVector::from_vec(vec![1.0, 0.0]));
        let mut agent = Agent::new(0, DVector::from_vec(vec![0.0, 0.0]));
        agent.step(&vf, 0.1);
        assert!((agent.position[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_agent_system() {
        let vf = VectorField::new("constant", 2, |_: &DVector<f64>| DVector::from_vec(vec![1.0, 0.0]));
        let mut system = AgentSystem::new(vf);
        system.add_agent(DVector::from_vec(vec![0.0, 0.0]));
        system.add_agent(DVector::from_vec(vec![0.0, 1.0]));
        system.step(0.1);
        assert!((system.agents[0].position[0] - 0.1).abs() < 1e-10);
        assert!((system.agents[1].position[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_clusters() {
        let vf = VectorField::new("zero", 2, |_: &DVector<f64>| DVector::zeros(2));
        let mut system = AgentSystem::new(vf);
        system.add_agent(DVector::from_vec(vec![0.0, 0.0]));
        system.add_agent(DVector::from_vec(vec![0.1, 0.0]));
        system.add_agent(DVector::from_vec(vec![5.0, 5.0]));
        let clusters = system.find_clusters(1.0);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_analyze_flow_radial() {
        let vf = VectorField::new("radial", 2, |p: &DVector<f64>| p.clone());
        let analysis = analyze_flow(&vf, &[(-1.0, 1.0), (-1.0, 1.0)], 20, 0.01, 10, 4);
        assert!(!analysis.equilibria.is_empty());
    }

    #[test]
    fn test_classify_dynamics_oscillatory() {
        let vf = crate::vector_field::rotational_vector_field();
        let analysis = analyze_flow(&vf, &[(-1.0, 1.0), (-1.0, 1.0)], 20, 0.01, 10, 4);
        let dynamics = classify_dynamics(&analysis);
        // Rotational field has center equilibrium
        assert!(dynamics == DynamicsType::Oscillatory || dynamics == DynamicsType::Mixed);
    }

    #[test]
    fn test_numerical_jacobian() {
        let vf = VectorField::new("linear", 2, |p: &DVector<f64>| DVector::from_vec(vec![p[0], -p[1]]));
        let p = DVector::from_vec(vec![0.0, 0.0]);
        let jac = numerical_jacobian(&vf, &p, 1e-5);
        assert!((jac[(0, 0)] - 1.0).abs() < 1e-3);
        assert!((jac[(1, 1)] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_agent_rk4() {
        let vf = VectorField::new("constant", 2, |_: &DVector<f64>| DVector::from_vec(vec![1.0, 0.0]));
        let mut agent = Agent::new(0, DVector::from_vec(vec![0.0, 0.0]));
        agent.step_rk4(&vf, 0.1);
        assert!((agent.position[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lyapunov_exponent() {
        let trajectory: Vec<DVector<f64>> = (0..100).map(|i| {
            let t = i as f64 * 0.01;
            DVector::from_vec(vec![t.exp(), 0.0])
        }).collect();
        let le = lyapunov_exponent(&trajectory, 0.01);
        // Should be positive (exponential growth)
        assert!(le > 0.0, "Lyapunov exponent should be positive, got {}", le);
    }

    #[test]
    fn test_agent_distance() {
        let a1 = Agent::new(0, DVector::from_vec(vec![0.0, 0.0]));
        let a2 = Agent::new(1, DVector::from_vec(vec![3.0, 4.0]));
        assert!((a1.distance_to(&a2) - 5.0).abs() < 1e-10);
    }
}
