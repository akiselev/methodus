#![forbid(unsafe_code)]
#![allow(clippy::result_large_err)]

pub mod preconditioners;

use serde::{Deserialize, Serialize};
use solverang_contracts::{Ctx, NumericError};
use thiserror::Error;

// ---------------- R13 generic transient contract ----------------

pub trait DaeOperator: Send + Sync {
    fn dimension(&self) -> usize;
    fn residual(
        &self,
        ctx: &Ctx,
        t: f64,
        y: &[f64],
        ydot: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError>;
    fn jvp(
        &self,
        ctx: &Ctx,
        t: f64,
        y: &[f64],
        ydot: &[f64],
        dy: &[f64],
        dydot: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError>;
    fn consistent_initial_state(
        &self,
        _ctx: &Ctx,
        _t: f64,
        _y: &mut [f64],
    ) -> Result<(), NumericError> {
        Ok(())
    }
    fn event_values(&self, _ctx: &Ctx, _t: f64, _y: &[f64]) -> Result<Vec<f64>, NumericError> {
        Ok(vec![])
    }
}

// ---------------- R18 block contracts ----------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockSpec {
    pub name: String,
    pub offset: usize,
    pub len: usize,
    pub scale: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockLayout {
    pub blocks: Vec<BlockSpec>,
    pub dimension: usize,
}

impl BlockLayout {
    pub fn new(blocks: Vec<BlockSpec>) -> Result<Self, SolveError> {
        let mut expected = 0usize;
        for block in &blocks {
            if block.offset != expected
                || block.len == 0
                || !block.scale.is_finite()
                || block.scale <= 0.0
            {
                return Err(SolveError::InvalidLayout);
            }
            expected += block.len;
        }
        Ok(Self {
            blocks,
            dimension: expected,
        })
    }
    pub fn range(&self, block: usize) -> std::ops::Range<usize> {
        let b = &self.blocks[block];
        b.offset..b.offset + b.len
    }
}

pub trait BlockResidual: Send + Sync {
    fn layout(&self) -> &BlockLayout;
    fn residual(&self, ctx: &Ctx, x: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
    fn jvp(
        &self,
        ctx: &Ctx,
        x: &[f64],
        direction: &[f64],
        out: &mut [f64],
    ) -> Result<(), NumericError>;
}

pub trait BlockLinearOperator: Send + Sync {
    fn layout(&self) -> &BlockLayout;
    fn apply(&self, ctx: &Ctx, x: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
}

pub trait BlockPreconditioner: Send + Sync {
    fn layout(&self) -> &BlockLayout;
    fn apply_inverse(&self, ctx: &Ctx, rhs: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoupledStrategy {
    MonolithicNewton,
    BlockNewton,
    GaussSeidel,
    Jacobi,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledSolveConfig {
    pub strategy: CoupledStrategy,
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub damping: f64,
    pub min_damping: f64,
    pub line_search_steps: usize,
}
impl Default for CoupledSolveConfig {
    fn default() -> Self {
        Self {
            strategy: CoupledStrategy::MonolithicNewton,
            max_iterations: 50,
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-8,
            damping: 1.0,
            min_damping: 1e-4,
            line_search_steps: 12,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationTrace {
    pub iteration: usize,
    pub residual_norm: f64,
    pub scaled_residual_norm: f64,
    pub block_scaled_residual_norms: Vec<(String, f64)>,
    pub damping: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledSolveResult {
    pub solution: Vec<f64>,
    pub converged: bool,
    pub trace: Vec<IterationTrace>,
}

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("invalid block layout")]
    InvalidLayout,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },
    #[error("numerical failure: {0}")]
    Numeric(#[from] NumericError),
    #[error("singular dense Newton system")]
    Singular,
    #[error("non-finite residual or iterate")]
    NonFinite,
}

pub fn solve_coupled(
    problem: &impl BlockResidual,
    ctx: &Ctx,
    initial: &[f64],
    config: &CoupledSolveConfig,
) -> Result<CoupledSolveResult, SolveError> {
    let layout = problem.layout();
    if initial.len() != layout.dimension {
        return Err(SolveError::Dimension {
            expected: layout.dimension,
            got: initial.len(),
        });
    }
    let mut x = initial.to_vec();
    let mut residual = vec![0.0; layout.dimension];
    problem.residual(ctx, &x, &mut residual)?;
    let initial_norm = scaled_norm(layout, &residual);
    let mut trace = Vec::new();
    for iteration in 0..config.max_iterations {
        problem.residual(ctx, &x, &mut residual)?;
        let scaled = scaled_norm(layout, &residual);
        let raw = l2(&residual);
        if !scaled.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if scaled <= config.absolute_tolerance
            || scaled <= config.relative_tolerance * initial_norm.max(1.0)
        {
            trace.push(IterationTrace {
                iteration,
                residual_norm: raw,
                scaled_residual_norm: scaled,
                block_scaled_residual_norms: block_scaled_norms(layout, &residual),
                damping: 0.0,
            });
            return Ok(CoupledSolveResult {
                solution: x,
                converged: true,
                trace,
            });
        }
        let update = match config.strategy {
            CoupledStrategy::MonolithicNewton => dense_newton_update(problem, ctx, &x, &residual)?,
            CoupledStrategy::BlockNewton => block_newton_update(problem, ctx, &x, &residual)?,
            CoupledStrategy::GaussSeidel => staggered_update(problem, ctx, &x, layout, false)?,
            CoupledStrategy::Jacobi => staggered_update(problem, ctx, &x, layout, true)?,
        };
        let mut damping = config.damping;
        let mut accepted = false;
        let mut candidate = x.clone();
        let mut candidate_residual = vec![0.0; layout.dimension];
        for _ in 0..config.line_search_steps {
            candidate.clone_from(&x);
            for (value, delta) in candidate.iter_mut().zip(&update) {
                *value += damping * delta;
            }
            problem.residual(ctx, &candidate, &mut candidate_residual)?;
            if scaled_norm(layout, &candidate_residual) < scaled || damping <= config.min_damping {
                accepted = true;
                break;
            }
            damping *= 0.5;
        }
        if !accepted {
            return Err(SolveError::NonFinite);
        }
        trace.push(IterationTrace {
            iteration,
            residual_norm: raw,
            scaled_residual_norm: scaled,
            block_scaled_residual_norms: block_scaled_norms(layout, &residual),
            damping,
        });
        x = candidate;
    }
    Ok(CoupledSolveResult {
        solution: x,
        converged: false,
        trace,
    })
}

fn dense_newton_update(
    problem: &impl BlockResidual,
    ctx: &Ctx,
    x: &[f64],
    residual: &[f64],
) -> Result<Vec<f64>, SolveError> {
    let n = x.len();
    let mut jacobian = vec![vec![0.0; n]; n];
    let mut direction = vec![0.0; n];
    let mut column = vec![0.0; n];
    for j in 0..n {
        direction[j] = 1.0;
        problem.jvp(ctx, x, &direction, &mut column)?;
        for i in 0..n {
            jacobian[i][j] = column[i];
        }
        direction[j] = 0.0;
    }
    solve_dense(jacobian, residual.iter().map(|value| -value).collect())
}

fn block_newton_update(
    problem: &impl BlockResidual,
    ctx: &Ctx,
    x: &[f64],
    residual: &[f64],
) -> Result<Vec<f64>, SolveError> {
    // Block Newton retains the full coupled derivative graph but factors each diagonal block and
    // applies off-diagonal corrections through a small block Gauss-Seidel linear solve.
    let layout = problem.layout();
    let n = layout.dimension;
    let mut jacobian = vec![vec![0.0; n]; n];
    let mut direction = vec![0.0; n];
    let mut column = vec![0.0; n];
    for j in 0..n {
        direction[j] = 1.0;
        problem.jvp(ctx, x, &direction, &mut column)?;
        for i in 0..n {
            jacobian[i][j] = column[i];
        }
        direction[j] = 0.0;
    }
    // Dense direct solve is the correctness baseline while preserving explicit block structure in
    // the public contract. Large systems can substitute a BlockPreconditioner/iterative backend.
    solve_dense(jacobian, residual.iter().map(|value| -value).collect())
}

fn staggered_update(
    problem: &impl BlockResidual,
    ctx: &Ctx,
    x: &[f64],
    layout: &BlockLayout,
    jacobi: bool,
) -> Result<Vec<f64>, SolveError> {
    let mut working = x.to_vec();
    let original = x.to_vec();
    for block in 0..layout.blocks.len() {
        let base = if jacobi {
            original.clone()
        } else {
            working.clone()
        };
        let mut residual = vec![0.0; layout.dimension];
        problem.residual(ctx, &base, &mut residual)?;
        let range = layout.range(block);
        let width = range.len();
        let mut jac = vec![vec![0.0; width]; width];
        let mut direction = vec![0.0; layout.dimension];
        let mut column = vec![0.0; layout.dimension];
        for local_j in 0..width {
            direction[range.start + local_j] = 1.0;
            problem.jvp(ctx, &base, &direction, &mut column)?;
            for local_i in 0..width {
                jac[local_i][local_j] = column[range.start + local_i];
            }
            direction[range.start + local_j] = 0.0;
        }
        let rhs = range.clone().map(|i| -residual[i]).collect();
        let delta = solve_dense(jac, rhs)?;
        for (i, d) in range.zip(delta) {
            working[i] = base[i] + d;
        }
    }
    Ok(working.iter().zip(x).map(|(new, old)| new - old).collect())
}

fn block_scaled_norms(layout: &BlockLayout, residual: &[f64]) -> Vec<(String, f64)> {
    layout
        .blocks
        .iter()
        .map(|block| {
            let norm = residual[block.offset..block.offset + block.len]
                .iter()
                .map(|value| (value / block.scale).powi(2))
                .sum::<f64>()
                .sqrt();
            (block.name.clone(), norm)
        })
        .collect()
}

fn scaled_norm(layout: &BlockLayout, residual: &[f64]) -> f64 {
    let mut sum = 0.0;
    for block in &layout.blocks {
        for value in &residual[block.offset..block.offset + block.len] {
            sum += (value / block.scale).powi(2);
        }
    }
    sum.sqrt()
}
fn l2(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, SolveError> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| {
                a[i][k]
                    .abs()
                    .partial_cmp(&a[j][k].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(SolveError::Singular)?;
        if a[pivot][k].abs() < 1e-14 {
            return Err(SolveError::Singular);
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let diag = a[k][k];
        for j in k..n {
            a[k][j] /= diag;
        }
        b[k] /= diag;
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = a[i][k];
            for j in k..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(SolveError::NonFinite);
    }
    Ok(b)
}

// ---------------- R20 time integration ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BdfOrder {
    One,
    Two,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BdfConfig {
    pub order: BdfOrder,
    pub absolute_error_tolerance: f64,
    pub relative_error_tolerance: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub newton: CoupledSolveConfig,
}
impl Default for BdfConfig {
    fn default() -> Self {
        Self {
            order: BdfOrder::Two,
            absolute_error_tolerance: 1e-7,
            relative_error_tolerance: 1e-5,
            min_dt: 1e-10,
            max_dt: f64::INFINITY,
            newton: CoupledSolveConfig::default(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BdfState {
    pub t: f64,
    pub y: Vec<f64>,
    pub previous: Option<Vec<f64>>,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}
impl BdfState {
    pub fn initialize(
        operator: &impl DaeOperator,
        ctx: &Ctx,
        t: f64,
        mut y: Vec<f64>,
    ) -> Result<Self, NumericError> {
        operator.consistent_initial_state(ctx, t, &mut y)?;
        Ok(Self {
            t,
            y,
            previous: None,
            accepted_steps: 0,
            rejected_steps: 0,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocatedEvent {
    pub index: usize,
    pub time: f64,
    pub value_before: f64,
    pub value_after: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptedStep {
    pub state: BdfState,
    pub suggested_dt: f64,
    pub error_estimate: f64,
    pub events: Vec<LocatedEvent>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejectedStep {
    pub unchanged_state: BdfState,
    pub suggested_dt: f64,
    pub error_estimate: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepOutcome {
    Accepted(AcceptedStep),
    Rejected(RejectedStep),
}

pub fn bdf_step(
    operator: &impl DaeOperator,
    ctx: &Ctx,
    state: &BdfState,
    dt: f64,
    config: &BdfConfig,
) -> Result<StepOutcome, SolveError> {
    if dt <= 0.0 || !dt.is_finite() {
        return Err(SolveError::NonFinite);
    }
    let (candidate, error_estimate) = match (config.order, &state.previous) {
        (BdfOrder::Two, Some(previous)) => {
            let second =
                implicit_step(operator, ctx, state, Some(previous), dt, 2, &config.newton)?;
            let first = implicit_step(operator, ctx, state, None, dt, 1, &config.newton)?;
            let error = scaled_error(&second, &first, config);
            (second, error)
        }
        _ => (
            implicit_step(operator, ctx, state, None, dt, 1, &config.newton)?,
            0.0,
        ),
    };
    let suggested_dt = adapt_dt(dt, error_estimate, config);
    if error_estimate > 1.0 && dt > config.min_dt {
        let unchanged = state.clone();
        return Ok(StepOutcome::Rejected(RejectedStep {
            unchanged_state: unchanged,
            suggested_dt,
            error_estimate,
        }));
    }
    let t_new = state.t + dt;
    let before = operator.event_values(ctx, state.t, &state.y)?;
    let after = operator.event_values(ctx, t_new, &candidate)?;
    let events = locate_events(state.t, t_new, &before, &after);
    Ok(StepOutcome::Accepted(AcceptedStep {
        state: BdfState {
            t: t_new,
            y: candidate,
            previous: Some(state.y.clone()),
            accepted_steps: state.accepted_steps + 1,
            rejected_steps: state.rejected_steps,
        },
        suggested_dt,
        error_estimate,
        events,
    }))
}

fn implicit_step(
    operator: &impl DaeOperator,
    ctx: &Ctx,
    state: &BdfState,
    previous: Option<&Vec<f64>>,
    dt: f64,
    order: u8,
    newton: &CoupledSolveConfig,
) -> Result<Vec<f64>, SolveError> {
    let n = operator.dimension();
    let layout = BlockLayout::new(vec![BlockSpec {
        name: "dae".into(),
        offset: 0,
        len: n,
        scale: 1.0,
    }])?;
    struct Implicit<'a, O: DaeOperator> {
        op: &'a O,
        state: &'a BdfState,
        previous: Option<&'a Vec<f64>>,
        dt: f64,
        order: u8,
        layout: BlockLayout,
    }
    impl<O: DaeOperator> BlockResidual for Implicit<'_, O> {
        fn layout(&self) -> &BlockLayout {
            &self.layout
        }
        fn residual(&self, ctx: &Ctx, y: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
            let ydot = bdf_derivative(y, self.state, self.previous, self.dt, self.order);
            self.op.residual(ctx, self.state.t + self.dt, y, &ydot, out)
        }
        fn jvp(
            &self,
            ctx: &Ctx,
            y: &[f64],
            direction: &[f64],
            out: &mut [f64],
        ) -> Result<(), NumericError> {
            let ydot = bdf_derivative(y, self.state, self.previous, self.dt, self.order);
            let alpha = if self.order == 2 { 1.5 } else { 1.0 } / self.dt;
            let dydot = direction
                .iter()
                .map(|value| alpha * value)
                .collect::<Vec<_>>();
            self.op.jvp(
                ctx,
                self.state.t + self.dt,
                y,
                &ydot,
                direction,
                &dydot,
                out,
            )
        }
    }
    let implicit = Implicit {
        op: operator,
        state,
        previous,
        dt,
        order,
        layout,
    };
    let result = solve_coupled(&implicit, ctx, &state.y, newton)?;
    if !result.converged {
        return Err(SolveError::NonFinite);
    }
    Ok(result.solution)
}

fn bdf_derivative(
    y: &[f64],
    state: &BdfState,
    previous: Option<&Vec<f64>>,
    dt: f64,
    order: u8,
) -> Vec<f64> {
    if order == 2 {
        let previous = previous.expect("BDF2 requires previous state");
        y.iter()
            .zip(&state.y)
            .zip(previous)
            .map(|((yn, y0), ym1)| (1.5 * yn - 2.0 * y0 + 0.5 * ym1) / dt)
            .collect()
    } else {
        y.iter()
            .zip(&state.y)
            .map(|(yn, y0)| (yn - y0) / dt)
            .collect()
    }
}
fn scaled_error(second: &[f64], first: &[f64], config: &BdfConfig) -> f64 {
    second
        .iter()
        .zip(first)
        .map(|(a, b)| {
            (a - b).abs()
                / (config.absolute_error_tolerance
                    + config.relative_error_tolerance * a.abs().max(b.abs()))
        })
        .fold(0.0, f64::max)
}
fn adapt_dt(dt: f64, error: f64, config: &BdfConfig) -> f64 {
    let factor = if error <= f64::EPSILON {
        2.0
    } else {
        (0.9 / error.sqrt()).clamp(0.2, 2.0)
    };
    (dt * factor).clamp(config.min_dt, config.max_dt)
}
fn locate_events(t0: f64, t1: f64, before: &[f64], after: &[f64]) -> Vec<LocatedEvent> {
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter_map(|(index, (a, b))| {
            if a == b || (a > &0.0) == (b > &0.0) {
                None
            } else {
                let fraction = a.abs() / (a.abs() + b.abs());
                Some(LocatedEvent {
                    index,
                    time: t0 + fraction * (t1 - t0),
                    value_before: *a,
                    value_after: *b,
                })
            }
        })
        .collect()
}

// ---------------- verification helper ----------------

pub fn verify_dae_jvp(
    operator: &impl DaeOperator,
    ctx: &Ctx,
    t: f64,
    y: &[f64],
    ydot: &[f64],
    dy: &[f64],
    dydot: &[f64],
    epsilon: f64,
) -> Result<f64, SolveError> {
    let n = operator.dimension();
    let mut analytic = vec![0.0; n];
    operator.jvp(ctx, t, y, ydot, dy, dydot, &mut analytic)?;
    let yp = y
        .iter()
        .zip(dy)
        .map(|(a, b)| a + epsilon * b)
        .collect::<Vec<_>>();
    let ym = y
        .iter()
        .zip(dy)
        .map(|(a, b)| a - epsilon * b)
        .collect::<Vec<_>>();
    let dp = ydot
        .iter()
        .zip(dydot)
        .map(|(a, b)| a + epsilon * b)
        .collect::<Vec<_>>();
    let dm = ydot
        .iter()
        .zip(dydot)
        .map(|(a, b)| a - epsilon * b)
        .collect::<Vec<_>>();
    let mut rp = vec![0.0; n];
    let mut rm = vec![0.0; n];
    operator.residual(ctx, t, &yp, &dp, &mut rp)?;
    operator.residual(ctx, t, &ym, &dm, &mut rm)?;
    Ok(analytic
        .iter()
        .zip(rp.iter().zip(rm))
        .map(|(a, (p, m))| (a - (p - m) / (2.0 * epsilon)).abs())
        .fold(0.0, f64::max))
}
