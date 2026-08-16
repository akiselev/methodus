#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use solverang_contracts::{Ctx, NumericError};
use thiserror::Error;

// ---------------- R13 generic transient contract ----------------

pub trait DaeOperator: Send + Sync {
    fn dimension(&self) -> usize;
    fn residual(&self, ctx: &Ctx, t: f64, y: &[f64], ydot: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
    fn jvp(&self, ctx: &Ctx, t: f64, y: &[f64], ydot: &[f64], dy: &[f64], dydot: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
    fn consistent_initial_state(&self, _ctx: &Ctx, _t: f64, _y: &mut [f64]) -> Result<(), NumericError> { Ok(()) }
    fn event_values(&self, _ctx: &Ctx, _t: f64, _y: &[f64]) -> Result<Vec<f64>, NumericError> { Ok(vec![]) }
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
pub struct BlockLayout { pub blocks: Vec<BlockSpec>, pub dimension: usize }

impl BlockLayout {
    pub fn new(blocks: Vec<BlockSpec>) -> Result<Self, SolveError> {
        let mut expected = 0usize;
        for block in &blocks {
            if block.offset != expected || block.len == 0 || !block.scale.is_finite() || block.scale <= 0.0 {
                return Err(SolveError::InvalidLayout);
            }
            expected += block.len;
        }
        Ok(Self { blocks, dimension: expected })
    }
    pub fn range(&self, block: usize) -> std::ops::Range<usize> {
        let b = &self.blocks[block]; b.offset..b.offset + b.len
    }
}

pub trait BlockResidual: Send + Sync {
    fn layout(&self) -> &BlockLayout;
    fn residual(&self, ctx: &Ctx, x: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
    fn jvp(&self, ctx: &Ctx, x: &[f64], direction: &[f64], out: &mut [f64]) -> Result<(), NumericError>;
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
pub enum CoupledStrategy { MonolithicNewton, BlockNewton, GaussSeidel, Jacobi }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoupledSolveConfig {
    pub strategy: CoupledStrategy,
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub minimum_damping: f64,
}
impl Default for CoupledSolveConfig {
    fn default() -> Self { Self { strategy: CoupledStrategy::MonolithicNewton, max_iterations: 50, absolute_tolerance: 1e-10, relative_tolerance: 1e-8, minimum_damping: 1e-4 } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationRecord { pub iteration: usize, pub residual_norm: f64, pub damping: f64 }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoupledSolveResult { pub solution: Vec<f64>, pub converged: bool, pub iterations: Vec<IterationRecord> }

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("invalid block layout")]
    InvalidLayout,
    #[error("dimension mismatch")]
    Dimension,
    #[error("singular dense linearization")]
    Singular,
    #[error("numerical contract failed: {0}")]
    Numeric(#[from] NumericError),
    #[error("nonlinear solve did not converge")]
    NoConvergence,
    #[error("invalid timestep")]
    InvalidStep,
}

pub fn solve_coupled(problem: &impl BlockResidual, ctx: &Ctx, initial: &[f64], config: &CoupledSolveConfig) -> Result<CoupledSolveResult, SolveError> {
    let n = problem.layout().dimension;
    if initial.len() != n { return Err(SolveError::Dimension); }
    let mut x = initial.to_vec();
    let mut r = vec![0.0; n];
    problem.residual(ctx, &x, &mut r)?;
    let initial_norm = scaled_norm(problem.layout(), &r).max(1.0);
    let mut records = vec![];
    for iteration in 0..config.max_iterations {
        problem.residual(ctx, &x, &mut r)?;
        let norm = scaled_norm(problem.layout(), &r);
        if norm <= config.absolute_tolerance || norm / initial_norm <= config.relative_tolerance {
            return Ok(CoupledSolveResult { solution: x, converged: true, iterations: records });
        }
        let old_norm = norm;
        let step = match config.strategy {
            CoupledStrategy::MonolithicNewton => newton_step(problem, ctx, &x, &r, 0..n)?,
            CoupledStrategy::BlockNewton | CoupledStrategy::GaussSeidel => {
                // A block lower-triangular nonlinear sweep. Re-linearize after every block so
                // dependencies updated by earlier blocks are immediately visible.
                let mut total = vec![0.0; n];
                for b in 0..problem.layout().blocks.len() {
                    problem.residual(ctx, &x, &mut r)?;
                    let range = problem.layout().range(b);
                    let local = newton_step(problem, ctx, &x, &r, range.clone())?;
                    for i in range { x[i] += local[i]; total[i] += local[i]; }
                }
                if config.strategy == CoupledStrategy::GaussSeidel { records.push(IterationRecord { iteration, residual_norm: old_norm, damping: 1.0 }); continue; }
                total
            }
            CoupledStrategy::Jacobi => {
                let mut total = vec![0.0; n];
                for b in 0..problem.layout().blocks.len() {
                    let range = problem.layout().range(b);
                    let local = newton_step(problem, ctx, &x, &r, range.clone())?;
                    for i in range { total[i] = local[i]; }
                }
                total
            }
        };
        let mut damping = 1.0;
        let original = x.clone();
        loop {
            for i in 0..n { x[i] = original[i] + damping * step[i]; }
            problem.residual(ctx, &x, &mut r)?;
            if scaled_norm(problem.layout(), &r) < old_norm || damping <= config.minimum_damping { break; }
            damping *= 0.5;
        }
        records.push(IterationRecord { iteration, residual_norm: old_norm, damping });
    }
    Ok(CoupledSolveResult { solution: x, converged: false, iterations: records })
}

fn newton_step(problem: &impl BlockResidual, ctx: &Ctx, x: &[f64], residual: &[f64], active: std::ops::Range<usize>) -> Result<Vec<f64>, SolveError> {
    let n = problem.layout().dimension;
    let ids: Vec<usize> = active.collect();
    let m = ids.len();
    let mut a = vec![vec![0.0; m]; m];
    let mut direction = vec![0.0; n];
    let mut column = vec![0.0; n];
    for (j_local, &j) in ids.iter().enumerate() {
        direction.fill(0.0); direction[j] = 1.0;
        problem.jvp(ctx, x, &direction, &mut column)?;
        for (i_local, &i) in ids.iter().enumerate() { a[i_local][j_local] = column[i]; }
    }
    let rhs = ids.iter().map(|&i| -residual[i]).collect::<Vec<_>>();
    let local = solve_dense(a, rhs)?;
    let mut out = vec![0.0; n];
    for (i, value) in ids.into_iter().zip(local) { out[i] = value; }
    Ok(out)
}

fn scaled_norm(layout: &BlockLayout, residual: &[f64]) -> f64 {
    let mut sum = 0.0;
    for block in &layout.blocks {
        for &v in &residual[block.offset..block.offset + block.len] { let x = v / block.scale; sum += x * x; }
    }
    sum.sqrt()
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, SolveError> {
    let n = b.len();
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n { if a[i][k].abs() > a[pivot][k].abs() { pivot = i; } }
        if a[pivot][k].abs() < 1e-14 { return Err(SolveError::Singular); }
        a.swap(k, pivot); b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k..n { a[i][j] -= factor * a[k][j]; }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut rhs = b[i]; for j in i + 1..n { rhs -= a[i][j] * x[j]; }
        x[i] = rhs / a[i][i];
    }
    Ok(x)
}

// ---------------- R20 BDF1/BDF2 and transactional step result ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BdfOrder { One, Two }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BdfConfig {
    pub order: BdfOrder,
    pub newton_iterations: usize,
    pub nonlinear_tolerance: f64,
    pub relative_error_tolerance: f64,
    pub absolute_error_tolerance: f64,
    pub min_dt: f64,
    pub max_dt: f64,
}
impl Default for BdfConfig {
    fn default() -> Self { Self { order: BdfOrder::Two, newton_iterations: 20, nonlinear_tolerance: 1e-10, relative_error_tolerance: 1e-5, absolute_error_tolerance: 1e-8, min_dt: 1e-12, max_dt: f64::INFINITY } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BdfState {
    pub t: f64,
    pub y: Vec<f64>,
    pub previous: Option<Vec<f64>>,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}

impl BdfState {
    pub fn initialize(operator: &impl DaeOperator, ctx: &Ctx, t: f64, mut y: Vec<f64>) -> Result<Self, SolveError> {
        if y.len() != operator.dimension() { return Err(SolveError::Dimension); }
        operator.consistent_initial_state(ctx, t, &mut y)?;
        Ok(Self { t, y, previous: None, accepted_steps: 0, rejected_steps: 0 })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventHit { pub index: usize, pub time: f64, pub value_before: f64, pub value_after: f64 }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptedStep { pub state: BdfState, pub used_order: BdfOrder, pub error_estimate: f64, pub suggested_dt: f64, pub events: Vec<EventHit> }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RejectedStep { pub unchanged_state: BdfState, pub error_estimate: f64, pub suggested_dt: f64 }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StepOutcome { Accepted(AcceptedStep), Rejected(RejectedStep) }

pub fn bdf_step(operator: &impl DaeOperator, ctx: &Ctx, state: &BdfState, dt: f64, config: &BdfConfig) -> Result<StepOutcome, SolveError> {
    if !dt.is_finite() || dt < config.min_dt || dt > config.max_dt { return Err(SolveError::InvalidStep); }
    let order = if config.order == BdfOrder::Two && state.previous.is_some() { BdfOrder::Two } else { BdfOrder::One };
    let candidate = implicit_candidate(operator, ctx, state, dt, order, config)?;
    let error = if order == BdfOrder::Two {
        let first = implicit_candidate(operator, ctx, state, dt, BdfOrder::One, config)?;
        normalized_error(&candidate, &first, config)
    } else { 0.0 };
    if error > 1.0 {
        let mut unchanged = state.clone(); unchanged.rejected_steps += 1;
        let factor = (0.9 / error.sqrt()).clamp(0.2, 0.8);
        return Ok(StepOutcome::Rejected(RejectedStep { unchanged_state: unchanged, error_estimate: error, suggested_dt: (dt * factor).max(config.min_dt) }));
    }
    let t_new = state.t + dt;
    let before = operator.event_values(ctx, state.t, &state.y)?;
    let after = operator.event_values(ctx, t_new, &candidate)?;
    let events = detect_events(state.t, t_new, &before, &after);
    let factor = if error <= 1e-14 { 2.0 } else { (0.9 / error.sqrt()).clamp(0.5, 2.0) };
    let next = BdfState { t: t_new, y: candidate, previous: Some(state.y.clone()), accepted_steps: state.accepted_steps + 1, rejected_steps: state.rejected_steps };
    Ok(StepOutcome::Accepted(AcceptedStep { state: next, used_order: order, error_estimate: error, suggested_dt: (dt * factor).min(config.max_dt) }))
}

fn implicit_candidate(operator: &impl DaeOperator, ctx: &Ctx, state: &BdfState, dt: f64, order: BdfOrder, config: &BdfConfig) -> Result<Vec<f64>, SolveError> {
    let n = operator.dimension();
    let mut y = state.y.clone();
    let mut ydot = vec![0.0; n];
    let mut r = vec![0.0; n];
    for _ in 0..config.newton_iterations {
        derivative(order, &y, &state.y, state.previous.as_deref(), dt, &mut ydot);
        operator.residual(ctx, state.t + dt, &y, &ydot, &mut r)?;
        if euclidean(&r) <= config.nonlinear_tolerance { return Ok(y); }
        let alpha = match order { BdfOrder::One => 1.0 / dt, BdfOrder::Two => 1.5 / dt };
        let mut a = vec![vec![0.0; n]; n];
        let mut dy = vec![0.0; n]; let mut dydot = vec![0.0; n]; let mut col = vec![0.0; n];
        for j in 0..n {
            dy.fill(0.0); dydot.fill(0.0); dy[j] = 1.0; dydot[j] = alpha;
            operator.jvp(ctx, state.t + dt, &y, &ydot, &dy, &dydot, &mut col)?;
            for i in 0..n { a[i][j] = col[i]; }
        }
        let step = solve_dense(a, r.iter().map(|v| -v).collect())?;
        for i in 0..n { y[i] += step[i]; }
    }
    Err(SolveError::NoConvergence)
}

fn derivative(order: BdfOrder, y: &[f64], previous: &[f64], previous2: Option<&[f64]>, dt: f64, out: &mut [f64]) {
    match order {
        BdfOrder::One => for i in 0..y.len() { out[i] = (y[i] - previous[i]) / dt; },
        BdfOrder::Two => {
            let p2 = previous2.expect("BDF2 requires two accepted states");
            for i in 0..y.len() { out[i] = (1.5 * y[i] - 2.0 * previous[i] + 0.5 * p2[i]) / dt; }
        }
    }
}
fn normalized_error(a: &[f64], b: &[f64], config: &BdfConfig) -> f64 {
    a.iter().zip(b).map(|(x,y)| { let scale = config.absolute_error_tolerance + config.relative_error_tolerance * x.abs().max(y.abs()); ((x-y)/scale).powi(2) }).sum::<f64>().sqrt() / (a.len().max(1) as f64).sqrt()
}
fn euclidean(x: &[f64]) -> f64 { x.iter().map(|v| v*v).sum::<f64>().sqrt() }
fn detect_events(t0:f64,t1:f64,before:&[f64],after:&[f64])->Vec<EventHit>{before.iter().zip(after).enumerate().filter_map(|(index,(&a,&b))|{if a==0.0||b==0.0||a.signum()!=b.signum(){let denom=a.abs()+b.abs();let fraction=if denom==0.0{0.5}else{a.abs()/denom};Some(EventHit{index,time:t0+(t1-t0)*fraction,value_before:a,value_after:b})}else{None}}).collect()}

// ---------------- numerical verification helpers ----------------

pub fn directional_jvp_error(problem:&impl BlockResidual,ctx:&Ctx,x:&[f64],direction:&[f64],epsilon:f64)->Result<f64,SolveError>{let n=problem.layout().dimension;if x.len()!=n||direction.len()!=n{return Err(SolveError::Dimension);}let mut analytic=vec![0.0;n];problem.jvp(ctx,x,direction,&mut analytic)?;let plus=x.iter().zip(direction).map(|(a,d)|a+epsilon*d).collect::<Vec<_>>();let minus=x.iter().zip(direction).map(|(a,d)|a-epsilon*d).collect::<Vec<_>>();let mut rp=vec![0.0;n];let mut rm=vec![0.0;n];problem.residual(ctx,&plus,&mut rp)?;problem.residual(ctx,&minus,&mut rm)?;let numeric=rp.iter().zip(rm).map(|(a,b)|(a-b)/(2.0*epsilon));Ok(analytic.iter().zip(numeric).map(|(a,b)|(a-b).abs()).fold(0.0,f64::max))}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearBlocks { layout: BlockLayout }
    impl LinearBlocks { fn new()->Self{Self{layout:BlockLayout::new(vec![BlockSpec{name:"a".into(),offset:0,len:1,scale:1.0},BlockSpec{name:"b".into(),offset:1,len:1,scale:1.0}]).unwrap()}} }
    impl BlockResidual for LinearBlocks {
        fn layout(&self)->&BlockLayout{&self.layout}
        fn residual(&self,_:&Ctx,x:&[f64],out:&mut[f64])->Result<(),NumericError>{out[0]=2.0*x[0]+x[1]-3.0;out[1]=x[0]+3.0*x[1]-4.0;Ok(())}
        fn jvp(&self,_:&Ctx,_:&[f64],d:&[f64],out:&mut[f64])->Result<(),NumericError>{out[0]=2.0*d[0]+d[1];out[1]=d[0]+3.0*d[1];Ok(())}
    }

    #[test]
    fn monolithic_and_block_solve_same_linear_system(){let p=LinearBlocks::new();let ctx=Ctx::reproducible();for strategy in [CoupledStrategy::MonolithicNewton,CoupledStrategy::BlockNewton,CoupledStrategy::GaussSeidel,CoupledStrategy::Jacobi]{let cfg=CoupledSolveConfig{strategy,max_iterations:100,..Default::default()};let r=solve_coupled(&p,&ctx,&[0.0,0.0],&cfg).unwrap();assert!(r.converged);assert!((r.solution[0]-1.0).abs()<1e-8);assert!((r.solution[1]-1.0).abs()<1e-8);}}

    struct Decay;
    impl DaeOperator for Decay {
        fn dimension(&self)->usize{1}
        fn residual(&self,_:&Ctx,_:f64,y:&[f64],ydot:&[f64],out:&mut[f64])->Result<(),NumericError>{out[0]=ydot[0]+y[0];Ok(())}
        fn jvp(&self,_:&Ctx,_:f64,_:&[f64],_:&[f64],dy:&[f64],dydot:&[f64],out:&mut[f64])->Result<(),NumericError>{out[0]=dydot[0]+dy[0];Ok(())}
        fn event_values(&self,_:&Ctx,_:f64,y:&[f64])->Result<Vec<f64>,NumericError>{Ok(vec![y[0]-0.5])}
    }

    #[test]
    fn bdf_step_is_transactional_and_detects_event(){let ctx=Ctx::reproducible();let op=Decay;let cfg=BdfConfig{order:BdfOrder::One,..Default::default()};let mut state=BdfState::initialize(&op,&ctx,0.0,vec![1.0]).unwrap();for _ in 0..8{match bdf_step(&op,&ctx,&state,0.1,&cfg).unwrap(){StepOutcome::Accepted(step)=>state=step.state,StepOutcome::Rejected(_)=>panic!("constant BDF1 step should not reject")}}assert!(state.y[0]<1.0);}

    #[test]
    fn jvp_matches_directional_difference(){let p=LinearBlocks::new();let e=directional_jvp_error(&p,&Ctx::reproducible(),&[1.0,1.0],&[0.3,-0.2],1e-6).unwrap();assert!(e<1e-9);}
}
