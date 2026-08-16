use solverang_contracts::{Ctx, NumericError};
use solverang_scientific::{
    BlockLayout, BlockResidual, BlockSpec, CoupledSolveConfig, CoupledStrategy, solve_coupled,
};

struct LinearCoupled {
    layout: BlockLayout,
    coupling: f64,
}
impl LinearCoupled {
    fn new(coupling: f64) -> Self {
        Self {
            layout: BlockLayout::new(vec![
                BlockSpec { name: "left".into(), offset: 0, len: 1, scale: 1.0 },
                BlockSpec { name: "right".into(), offset: 1, len: 1, scale: 1.0 },
            ]).unwrap(),
            coupling,
        }
    }
}
impl BlockResidual for LinearCoupled {
    fn layout(&self) -> &BlockLayout { &self.layout }
    fn residual(&self, _ctx: &Ctx, x: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        out[0] = x[0] + self.coupling * x[1] - 1.0;
        out[1] = self.coupling * x[0] + x[1];
        Ok(())
    }
    fn jvp(&self, _ctx: &Ctx, _x: &[f64], d: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        out[0] = d[0] + self.coupling * d[1];
        out[1] = self.coupling * d[0] + d[1];
        Ok(())
    }
}

fn config(strategy: CoupledStrategy, max_iterations: usize) -> CoupledSolveConfig {
    CoupledSolveConfig {
        strategy,
        max_iterations,
        absolute_tolerance: 1.0e-11,
        relative_tolerance: 1.0e-11,
        ..CoupledSolveConfig::default()
    }
}

#[test]
fn monolithic_and_block_newton_reach_the_same_reference_solution() {
    let ctx = Ctx::reproducible();
    let problem = LinearCoupled::new(0.7);
    let mono = solve_coupled(&problem, &ctx, &[0.0, 0.0], &config(CoupledStrategy::MonolithicNewton, 8)).unwrap();
    let block = solve_coupled(&problem, &ctx, &[0.0, 0.0], &config(CoupledStrategy::BlockNewton, 8)).unwrap();
    assert!(mono.converged && block.converged);
    for (a, b) in mono.solution.iter().zip(&block.solution) {
        assert!((a - b).abs() < 1.0e-11);
    }
    for trace in mono.trace.iter().chain(&block.trace) {
        assert_eq!(trace.block_scaled_residual_norms.len(), 2);
        assert_eq!(trace.block_scaled_residual_norms[0].0, "left");
        assert_eq!(trace.block_scaled_residual_norms[1].0, "right");
    }
}

#[test]
fn strong_coupling_exposes_staggered_failure_while_monolithic_converges() {
    let ctx = Ctx::reproducible();
    let problem = LinearCoupled::new(0.99);
    let mono = solve_coupled(&problem, &ctx, &[0.0, 0.0], &config(CoupledStrategy::MonolithicNewton, 4)).unwrap();
    let staggered = solve_coupled(&problem, &ctx, &[0.0, 0.0], &config(CoupledStrategy::GaussSeidel, 5)).unwrap();
    assert!(mono.converged);
    assert!(!staggered.converged, "strongly coupled staggered solve unexpectedly converged in five iterations");
    assert!(mono.trace.len() < staggered.trace.len());
}
