#![forbid(unsafe_code)]
//! Transitional Solverang-side compatibility surface for Malleus.
//!
//! The production compiler remains `sinbad-malleus`. This crate removes Solverang's absolute
//! `/home/dev/sinbad` dependency while retaining the historical opcode/JIT API. Its compiler
//! is a deterministic portable evaluator: it is an oracle/fallback, not the optimized
//! Cranelift backend. Sinbad's Resolvent adapter targets the authoritative Malleus crate.

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u16);
impl Reg {
    pub fn new(index: u16) -> Self {
        Self(index)
    }
    pub fn index(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintOp {
    LoadVar { dst: Reg, var_idx: u32 },
    LoadConst { dst: Reg, value: f64 },
    Add { dst: Reg, a: Reg, b: Reg },
    Sub { dst: Reg, a: Reg, b: Reg },
    Mul { dst: Reg, a: Reg, b: Reg },
    Div { dst: Reg, a: Reg, b: Reg },
    Neg { dst: Reg, src: Reg },
    Sqrt { dst: Reg, src: Reg },
    Sin { dst: Reg, src: Reg },
    Cos { dst: Reg, src: Reg },
    Atan2 { dst: Reg, y: Reg, x: Reg },
    Abs { dst: Reg, src: Reg },
    Max { dst: Reg, a: Reg, b: Reg },
    Min { dst: Reg, a: Reg, b: Reg },
    Exp { dst: Reg, src: Reg },
    Ln { dst: Reg, src: Reg },
    Pow { dst: Reg, base: Reg, exp: Reg },
    Tan { dst: Reg, src: Reg },
    Asin { dst: Reg, src: Reg },
    Acos { dst: Reg, src: Reg },
    Sinh { dst: Reg, src: Reg },
    Cosh { dst: Reg, src: Reg },
    Tanh { dst: Reg, src: Reg },
    StoreResidual { residual_idx: u32, src: Reg },
    StoreJacobianIndexed { output_idx: u32, src: Reg },
    StoreHessianIndexed { output_idx: u32, src: Reg },
}
impl ConstraintOp {
    pub fn uses_register(&self, r: Reg) -> bool {
        match self {
            Self::LoadVar { .. } | Self::LoadConst { .. } => false,
            Self::Add { a, b, .. }
            | Self::Sub { a, b, .. }
            | Self::Mul { a, b, .. }
            | Self::Div { a, b, .. }
            | Self::Max { a, b, .. }
            | Self::Min { a, b, .. } => *a == r || *b == r,
            Self::Atan2 { y, x, .. } => *y == r || *x == r,
            Self::Pow { base, exp, .. } => *base == r || *exp == r,
            Self::Neg { src, .. }
            | Self::Sqrt { src, .. }
            | Self::Sin { src, .. }
            | Self::Cos { src, .. }
            | Self::Abs { src, .. }
            | Self::Exp { src, .. }
            | Self::Ln { src, .. }
            | Self::Tan { src, .. }
            | Self::Asin { src, .. }
            | Self::Acos { src, .. }
            | Self::Sinh { src, .. }
            | Self::Cosh { src, .. }
            | Self::Tanh { src, .. }
            | Self::StoreResidual { src, .. }
            | Self::StoreJacobianIndexed { src, .. }
            | Self::StoreHessianIndexed { src, .. } => *src == r,
        }
    }
    pub fn defines_register(&self, r: Reg) -> bool {
        self.dst() == Some(r)
    }
    fn dst(&self) -> Option<Reg> {
        match self {
            Self::LoadVar { dst, .. }
            | Self::LoadConst { dst, .. }
            | Self::Add { dst, .. }
            | Self::Sub { dst, .. }
            | Self::Mul { dst, .. }
            | Self::Div { dst, .. }
            | Self::Neg { dst, .. }
            | Self::Sqrt { dst, .. }
            | Self::Sin { dst, .. }
            | Self::Cos { dst, .. }
            | Self::Atan2 { dst, .. }
            | Self::Abs { dst, .. }
            | Self::Max { dst, .. }
            | Self::Min { dst, .. }
            | Self::Exp { dst, .. }
            | Self::Ln { dst, .. }
            | Self::Pow { dst, .. }
            | Self::Tan { dst, .. }
            | Self::Asin { dst, .. }
            | Self::Acos { dst, .. }
            | Self::Sinh { dst, .. }
            | Self::Cosh { dst, .. }
            | Self::Tanh { dst, .. } => Some(*dst),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JacobianEntry {
    pub row: u32,
    pub col: u32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HessianEntry {
    pub row: u32,
    pub col: u32,
}
#[derive(Clone, Debug)]
pub struct CompiledConstraints {
    pub residual_ops: Vec<ConstraintOp>,
    pub jacobian_ops: Vec<ConstraintOp>,
    pub hessian_ops: Vec<ConstraintOp>,
    pub n_residuals: usize,
    pub n_vars: usize,
    pub jacobian_nnz: usize,
    pub jacobian_pattern: Vec<JacobianEntry>,
    pub hessian_nnz: usize,
    pub hessian_pattern: Vec<HessianEntry>,
    pub max_register: u16,
}
impl CompiledConstraints {
    pub fn new(n_vars: usize, n_residuals: usize) -> Self {
        Self {
            residual_ops: vec![],
            jacobian_ops: vec![],
            hessian_ops: vec![],
            n_residuals,
            n_vars,
            jacobian_nnz: 0,
            jacobian_pattern: vec![],
            hessian_nnz: 0,
            hessian_pattern: vec![],
            max_register: 0,
        }
    }
    pub fn total_ops(&self) -> usize {
        self.residual_ops.len() + self.jacobian_ops.len() + self.hessian_ops.len()
    }
    pub fn validate(&self) -> Result<(), ValidationError> {
        for op in self
            .residual_ops
            .iter()
            .chain(&self.jacobian_ops)
            .chain(&self.hessian_ops)
        {
            if let ConstraintOp::LoadVar { var_idx, .. } = op {
                if *var_idx as usize >= self.n_vars {
                    return Err(ValidationError::VariableIndexOutOfBounds);
                }
            }
        }
        for op in &self.residual_ops {
            if let ConstraintOp::StoreResidual { residual_idx, .. } = op {
                if *residual_idx as usize >= self.n_residuals {
                    return Err(ValidationError::ResidualIndexOutOfBounds);
                }
            }
        }
        if self.jacobian_nnz != self.jacobian_pattern.len() {
            return Err(ValidationError::JacobianPatternMismatch);
        }
        if self.hessian_nnz != self.hessian_pattern.len() {
            return Err(ValidationError::HessianPatternMismatch);
        }
        Ok(())
    }
    pub fn densify_jacobian_ops(&self, n_rows: usize) -> Vec<ConstraintOp> {
        self.jacobian_ops
            .iter()
            .map(|op| match op {
                ConstraintOp::StoreJacobianIndexed { output_idx, src } => {
                    let e = self.jacobian_pattern[*output_idx as usize];
                    ConstraintOp::StoreJacobianIndexed {
                        output_idx: (e.col as usize * n_rows + e.row as usize) as u32,
                        src: *src,
                    }
                }
                x => x.clone(),
            })
            .collect()
    }
    pub fn fuse_ops(&self) -> (Vec<ConstraintOp>, u16) {
        let mut out = self.residual_ops.clone();
        out.extend(self.jacobian_ops.clone());
        let max = out
            .iter()
            .filter_map(ConstraintOp::dst)
            .map(|r| r.0)
            .max()
            .unwrap_or(0);
        (out, max)
    }
    pub fn fuse_ops_dense(&self, n_rows: usize) -> (Vec<ConstraintOp>, u16) {
        let mut out = self.residual_ops.clone();
        out.extend(self.densify_jacobian_ops(n_rows));
        let max = out
            .iter()
            .filter_map(ConstraintOp::dst)
            .map(|r| r.0)
            .max()
            .unwrap_or(0);
        (out, max)
    }
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum ValidationError {
    #[error("residual index out of bounds")]
    ResidualIndexOutOfBounds,
    #[error("variable index out of bounds")]
    VariableIndexOutOfBounds,
    #[error("Jacobian pattern mismatch")]
    JacobianPatternMismatch,
    #[error("Hessian pattern mismatch")]
    HessianPatternMismatch,
}

#[derive(Debug, Default)]
pub struct OpcodeEmitter {
    ops: Vec<ConstraintOp>,
    next: u16,
    jac: Vec<JacobianEntry>,
    hess: Vec<HessianEntry>,
    current: u32,
}
impl OpcodeEmitter {
    pub fn new() -> Self {
        Self::default()
    }
    fn reg(&mut self) -> Reg {
        let r = Reg(self.next);
        self.next += 1;
        r
    }
    pub fn max_register(&self) -> u16 {
        self.next.saturating_sub(1)
    }
    pub fn ops(&self) -> &[ConstraintOp] {
        &self.ops
    }
    pub fn into_ops(self) -> Vec<ConstraintOp> {
        self.ops
    }
    pub fn set_residual_index(&mut self, i: u32) {
        self.current = i
    }
    pub fn jacobian_entries(&self) -> &[JacobianEntry] {
        &self.jac
    }
    pub fn take_jacobian_entries(&mut self) -> Vec<JacobianEntry> {
        std::mem::take(&mut self.jac)
    }
    pub fn take_hessian_entries(&mut self) -> Vec<HessianEntry> {
        std::mem::take(&mut self.hess)
    }
    pub fn load_var(&mut self, i: u32) -> Reg {
        let d = self.reg();
        self.ops.push(ConstraintOp::LoadVar { dst: d, var_idx: i });
        d
    }
    pub fn const_f64(&mut self, v: f64) -> Reg {
        let d = self.reg();
        self.ops.push(ConstraintOp::LoadConst { dst: d, value: v });
        d
    }
    pub fn zero(&mut self) -> Reg {
        self.const_f64(0.)
    }
    pub fn one(&mut self) -> Reg {
        self.const_f64(1.)
    }
}
macro_rules! bin {
    ($n:ident,$v:ident) => {
        pub fn $n(&mut self, a: Reg, b: Reg) -> Reg {
            let d = self.reg();
            self.ops.push(ConstraintOp::$v { dst: d, a, b });
            d
        }
    };
}
macro_rules! una {
    ($n:ident,$v:ident) => {
        pub fn $n(&mut self, src: Reg) -> Reg {
            let d = self.reg();
            self.ops.push(ConstraintOp::$v { dst: d, src });
            d
        }
    };
}
impl OpcodeEmitter {
    bin!(add, Add);
    bin!(sub, Sub);
    bin!(mul, Mul);
    bin!(div, Div);
    bin!(max, Max);
    bin!(min, Min);
    una!(neg, Neg);
    una!(sqrt, Sqrt);
    una!(sin, Sin);
    una!(cos, Cos);
    una!(abs, Abs);
    una!(exp, Exp);
    una!(ln, Ln);
    una!(tan, Tan);
    una!(asin, Asin);
    una!(acos, Acos);
    una!(sinh, Sinh);
    una!(cosh, Cosh);
    una!(tanh, Tanh);
    pub fn atan2(&mut self, y: Reg, x: Reg) -> Reg {
        let d = self.reg();
        self.ops.push(ConstraintOp::Atan2 { dst: d, y, x });
        d
    }
    pub fn pow(&mut self, base: Reg, exp: Reg) -> Reg {
        let d = self.reg();
        self.ops.push(ConstraintOp::Pow { dst: d, base, exp });
        d
    }
    pub fn square(&mut self, x: Reg) -> Reg {
        self.mul(x, x)
    }
    pub fn safe_distance(&mut self, x: Reg, e: f64) -> Reg {
        let d = self.sqrt(x);
        let e = self.const_f64(e);
        self.max(d, e)
    }
    pub fn store_residual(&mut self, i: u32, src: Reg) {
        self.ops.push(ConstraintOp::StoreResidual {
            residual_idx: i,
            src,
        })
    }
    pub fn store_jacobian(&mut self, row: u32, col: u32, src: Reg) {
        let i = self.jac.len() as u32;
        self.jac.push(JacobianEntry { row, col });
        self.ops
            .push(ConstraintOp::StoreJacobianIndexed { output_idx: i, src })
    }
    pub fn store_jacobian_current(&mut self, col: u32, src: Reg) {
        self.store_jacobian(self.current, col, src)
    }
    pub fn store_hessian(&mut self, row: u32, col: u32, src: Reg) {
        let i = self.hess.len() as u32;
        self.hess.push(HessianEntry { row, col });
        self.ops
            .push(ConstraintOp::StoreHessianIndexed { output_idx: i, src })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JitMode {
    #[default]
    Auto,
    ForceJit,
    ForceInterpreted,
}
#[derive(Clone, Debug)]
pub struct JITConfig {
    pub jit_threshold: usize,
    pub estimated_iterations: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub mode: JitMode,
}
impl Default for JITConfig {
    fn default() -> Self {
        Self {
            jit_threshold: 1000,
            estimated_iterations: 50,
            max_iterations: 200,
            tolerance: 1e-8,
            mode: JitMode::Auto,
        }
    }
}
impl JITConfig {
    pub fn always_jit() -> Self {
        Self {
            mode: JitMode::ForceJit,
            ..Default::default()
        }
    }
    pub fn always_interpreted() -> Self {
        Self {
            mode: JitMode::ForceInterpreted,
            ..Default::default()
        }
    }
    pub fn for_large_problems() -> Self {
        Self {
            jit_threshold: 500,
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }
}
pub fn jit_available() -> bool {
    true
}

#[derive(Clone, Debug, Error)]
pub enum JITError {
    #[error("JIT backend is not available")]
    NotAvailable,
    #[error("invalid compiled constraints: {0}")]
    Validation(String),
    #[error("execution error: {0}")]
    Execution(String),
}

pub struct JITCompiler;
impl JITCompiler {
    pub fn new() -> Result<Self, JITError> {
        Ok(Self)
    }
    pub fn compile(&mut self, c: &CompiledConstraints) -> Result<JITFunction, JITError> {
        c.validate()
            .map_err(|e| JITError::Validation(e.to_string()))?;
        Ok(JITFunction {
            compiled: c.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct JITFunction {
    compiled: CompiledConstraints,
}
impl JITFunction {
    pub fn variable_count(&self) -> usize {
        self.compiled.n_vars
    }
    pub fn residual_count(&self) -> usize {
        self.compiled.n_residuals
    }
    pub fn evaluate_residuals(&self, vars: &[f64], out: &mut [f64]) {
        run(
            &self.compiled.residual_ops,
            self.compiled.max_register,
            vars,
            Some(out),
            None,
            None,
        )
    }
    pub fn evaluate_jacobian(&self, vars: &[f64], out: &mut [f64]) {
        run(
            &self.compiled.jacobian_ops,
            self.compiled.max_register,
            vars,
            None,
            Some(out),
            None,
        )
    }
    pub fn evaluate_hessian(&self, vars: &[f64], out: &mut [f64]) {
        run(
            &self.compiled.hessian_ops,
            self.compiled.max_register,
            vars,
            None,
            None,
            Some(out),
        )
    }
    pub fn evaluate_both(&self, vars: &[f64], residual: &mut [f64], jacobian: &mut [f64]) {
        self.evaluate_residuals(vars, residual);
        self.evaluate_jacobian(vars, jacobian)
    }
    pub fn evaluate_both_dense(
        &self,
        vars: &[f64],
        residual: &mut [f64],
        dense_jacobian: &mut [f64],
    ) {
        self.evaluate_residuals(vars, residual);
        let ops = self
            .compiled
            .densify_jacobian_ops(self.compiled.n_residuals);
        run(
            &ops,
            self.compiled.max_register,
            vars,
            None,
            Some(dense_jacobian),
            None,
        )
    }
}

fn run(
    ops: &[ConstraintOp],
    max: u16,
    vars: &[f64],
    mut residual: Option<&mut [f64]>,
    mut jac: Option<&mut [f64]>,
    mut hess: Option<&mut [f64]>,
) {
    let mut r = vec![0.; max as usize + 1];
    for op in ops {
        match *op {
            ConstraintOp::LoadVar { dst, var_idx } => {
                r[dst.0 as usize] = vars.get(var_idx as usize).copied().unwrap_or(f64::NAN)
            }
            ConstraintOp::LoadConst { dst, value } => r[dst.0 as usize] = value,
            ConstraintOp::Add { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize] + r[b.0 as usize]
            }
            ConstraintOp::Sub { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize] - r[b.0 as usize]
            }
            ConstraintOp::Mul { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize] * r[b.0 as usize]
            }
            ConstraintOp::Div { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize] / r[b.0 as usize]
            }
            ConstraintOp::Neg { dst, src } => r[dst.0 as usize] = -r[src.0 as usize],
            ConstraintOp::Sqrt { dst, src } => r[dst.0 as usize] = r[src.0 as usize].sqrt(),
            ConstraintOp::Sin { dst, src } => r[dst.0 as usize] = r[src.0 as usize].sin(),
            ConstraintOp::Cos { dst, src } => r[dst.0 as usize] = r[src.0 as usize].cos(),
            ConstraintOp::Atan2 { dst, y, x } => {
                r[dst.0 as usize] = r[y.0 as usize].atan2(r[x.0 as usize])
            }
            ConstraintOp::Abs { dst, src } => r[dst.0 as usize] = r[src.0 as usize].abs(),
            ConstraintOp::Max { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize].max(r[b.0 as usize])
            }
            ConstraintOp::Min { dst, a, b } => {
                r[dst.0 as usize] = r[a.0 as usize].min(r[b.0 as usize])
            }
            ConstraintOp::Exp { dst, src } => r[dst.0 as usize] = r[src.0 as usize].exp(),
            ConstraintOp::Ln { dst, src } => r[dst.0 as usize] = r[src.0 as usize].ln(),
            ConstraintOp::Pow { dst, base, exp } => {
                r[dst.0 as usize] = r[base.0 as usize].powf(r[exp.0 as usize])
            }
            ConstraintOp::Tan { dst, src } => r[dst.0 as usize] = r[src.0 as usize].tan(),
            ConstraintOp::Asin { dst, src } => r[dst.0 as usize] = r[src.0 as usize].asin(),
            ConstraintOp::Acos { dst, src } => r[dst.0 as usize] = r[src.0 as usize].acos(),
            ConstraintOp::Sinh { dst, src } => r[dst.0 as usize] = r[src.0 as usize].sinh(),
            ConstraintOp::Cosh { dst, src } => r[dst.0 as usize] = r[src.0 as usize].cosh(),
            ConstraintOp::Tanh { dst, src } => r[dst.0 as usize] = r[src.0 as usize].tanh(),
            ConstraintOp::StoreResidual { residual_idx, src } => {
                if let Some(o) = residual.as_deref_mut() {
                    if let Some(x) = o.get_mut(residual_idx as usize) {
                        *x = r[src.0 as usize]
                    }
                }
            }
            ConstraintOp::StoreJacobianIndexed { output_idx, src } => {
                if let Some(o) = jac.as_deref_mut() {
                    if let Some(x) = o.get_mut(output_idx as usize) {
                        *x = r[src.0 as usize]
                    }
                }
            }
            ConstraintOp::StoreHessianIndexed { output_idx, src } => {
                if let Some(o) = hess.as_deref_mut() {
                    if let Some(x) = o.get_mut(output_idx as usize) {
                        *x = r[src.0 as usize]
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompiledNewtonStep;
