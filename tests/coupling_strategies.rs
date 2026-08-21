use methodus::{
    BlockLayout, BlockNonlinearOperator, BlockSpec, BlockStrategy, EvaluationContext, NewtonConfig,
    NonlinearOperator, NumericError, solve_blocks,
};

struct LinearCoupled {
    layout: BlockLayout,
    coupling: f64,
}

impl LinearCoupled {
    fn new(coupling: f64) -> Self {
        Self {
            layout: BlockLayout::new(vec![
                BlockSpec {
                    name: "left".into(),
                    length: 1,
                    residual_scale: 1.0,
                },
                BlockSpec {
                    name: "right".into(),
                    length: 1,
                    residual_scale: 1.0,
                },
            ])
            .unwrap(),
            coupling,
        }
    }
}

impl NonlinearOperator for LinearCoupled {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] + self.coupling * state[1] - 1.0;
        output[1] = self.coupling * state[0] + state[1];
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = direction[0] + self.coupling * direction[1];
        output[1] = self.coupling * direction[0] + direction[1];
        Ok(())
    }
}

impl BlockNonlinearOperator for LinearCoupled {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

fn config(max_iterations: usize) -> NewtonConfig {
    NewtonConfig {
        max_iterations,
        absolute_tolerance: 1.0e-11,
        relative_tolerance: 1.0e-11,
        ..NewtonConfig::default()
    }
}

#[test]
fn monolithic_block_solve_reaches_the_reference_solution() {
    let context = EvaluationContext::reproducible();
    let problem = LinearCoupled::new(0.7);
    let report = solve_blocks(
        &problem,
        &context,
        &[0.0, 0.0],
        BlockStrategy::Monolithic,
        &config(8),
    )
    .unwrap();
    assert!(report.converged);
    assert!((report.state[0] - 1.0 / (1.0 - 0.7_f64.powi(2))).abs() < 1.0e-11);
    assert!((report.state[1] + 0.7 / (1.0 - 0.7_f64.powi(2))).abs() < 1.0e-11);
    for trace in &report.trace {
        assert_eq!(trace.block_residual_norms.len(), 2);
        assert_eq!(trace.block_residual_norms[0].0, "left");
        assert_eq!(trace.block_residual_norms[1].0, "right");
    }
}

#[test]
fn strong_coupling_exposes_staggered_failure_while_monolithic_converges() {
    let context = EvaluationContext::reproducible();
    let problem = LinearCoupled::new(0.99);
    let monolithic = solve_blocks(
        &problem,
        &context,
        &[0.0, 0.0],
        BlockStrategy::Monolithic,
        &config(4),
    )
    .unwrap();
    let staggered = solve_blocks(
        &problem,
        &context,
        &[0.0, 0.0],
        BlockStrategy::GaussSeidel,
        &config(5),
    )
    .unwrap();
    assert!(monolithic.converged);
    assert!(!staggered.converged);
    assert!(monolithic.trace.len() < staggered.trace.len());
}
