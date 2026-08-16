#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Solver execution context. Physics/runtime systems may wrap richer contexts around this
/// seam; Solverang itself only requires an explicit value instead of ambient global state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ctx {
    #[serde(default)]
    pub reproducible: bool,
}
impl Ctx {
    pub fn real_os_default() -> Self {
        Self {
            reproducible: false,
        }
    }
    pub fn reproducible() -> Self {
        Self { reproducible: true }
    }
}

#[derive(Clone, Debug, PartialEq, Error, Serialize, Deserialize)]
pub enum NumericError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("non-finite numerical value: {what}")]
    NonFinite { what: String },
    #[error("singular operator: {what}")]
    Singular { what: String },
    #[error("unsupported numerical operation: {what}")]
    Unsupported { what: String },
    #[error("numerical backend failure: {message}")]
    Backend { message: String },
}

pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + PartialEq
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
{
    fn zero() -> Self;
    fn one() -> Self;
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

pub trait SparseIndex: Copy + Ord + Send + Sync + 'static {
    fn from_usize(i: usize) -> Self;
    fn to_usize(self) -> usize;
}
impl SparseIndex for u32 {
    fn from_usize(i: usize) -> Self {
        i as u32
    }
    fn to_usize(self) -> usize {
        self as usize
    }
}
impl SparseIndex for u64 {
    fn from_usize(i: usize) -> Self {
        i as u64
    }
    fn to_usize(self) -> usize {
        self as usize
    }
}
impl SparseIndex for usize {
    fn from_usize(i: usize) -> Self {
        i
    }
    fn to_usize(self) -> usize {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Orientation {
    Csr,
    Csc,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SortOrder {
    Unsorted,
    InnerSorted,
    Canonical,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparsePattern<I = u32> {
    pub nrows: usize,
    pub ncols: usize,
    pub orientation: Orientation,
    pub offsets: Vec<I>,
    pub indices: Vec<I>,
    pub order: SortOrder,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseMatrix<S, I = u32> {
    pub pattern: Arc<SparsePattern<I>>,
    pub values: Vec<S>,
}
impl<S, I> SparseMatrix<S, I> {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    pub fn nrows(&self) -> usize {
        self.pattern.nrows
    }
    pub fn ncols(&self) -> usize {
        self.pattern.ncols
    }
}

#[derive(Clone, Debug)]
pub struct CooMatrix<S> {
    nrows: usize,
    ncols: usize,
    entries: Vec<(usize, usize, S)>,
}
impl<S: Scalar> CooMatrix<S> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            entries: vec![],
        }
    }
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            entries: Vec::with_capacity(capacity),
        }
    }
    pub fn push(&mut self, row: usize, col: usize, value: S) {
        self.entries.push((row, col, value))
    }
    pub fn finish_csr(self) -> SparseMatrix<S> {
        self.compress(Orientation::Csr)
    }
    pub fn finish_csc(self) -> SparseMatrix<S> {
        self.compress(Orientation::Csc)
    }
    fn compress(mut self, orientation: Orientation) -> SparseMatrix<S> {
        match orientation {
            Orientation::Csr => self.entries.sort_by_key(|(r, c, _)| (*r, *c)),
            Orientation::Csc => self.entries.sort_by_key(|(r, c, _)| (*c, *r)),
        }
        let mut merged: Vec<(usize, usize, S)> = vec![];
        for (r, c, v) in self.entries {
            if let Some(last) = merged.last_mut() {
                if last.0 == r && last.1 == c {
                    last.2 += v;
                    continue;
                }
            }
            merged.push((r, c, v))
        }
        let major = match orientation {
            Orientation::Csr => self.nrows,
            Orientation::Csc => self.ncols,
        };
        let mut offsets = vec![0u32; major + 1];
        let mut indices = Vec::with_capacity(merged.len());
        let mut values = Vec::with_capacity(merged.len());
        let mut cursor = 0usize;
        for m in 0..major {
            offsets[m] = cursor as u32;
            while cursor < merged.len() {
                let (r, c, v) = merged[cursor];
                let current = match orientation {
                    Orientation::Csr => r,
                    Orientation::Csc => c,
                };
                if current != m {
                    break;
                }
                indices.push(match orientation {
                    Orientation::Csr => c as u32,
                    Orientation::Csc => r as u32,
                });
                values.push(v);
                cursor += 1;
            }
        }
        offsets[major] = cursor as u32;
        SparseMatrix {
            pattern: Arc::new(SparsePattern {
                nrows: self.nrows,
                ncols: self.ncols,
                orientation,
                offsets,
                indices,
                order: SortOrder::Canonical,
            }),
            values,
        }
    }
}

impl<S: Scalar, I: SparseIndex> SparseMatrix<S, I> {
    pub fn apply(&self, x: &[S], out: &mut [S]) -> Result<(), NumericError> {
        if x.len() != self.ncols() {
            return Err(NumericError::DimensionMismatch {
                expected: self.ncols(),
                got: x.len(),
            });
        }
        if out.len() != self.nrows() {
            return Err(NumericError::DimensionMismatch {
                expected: self.nrows(),
                got: out.len(),
            });
        }
        for v in out.iter_mut() {
            *v = S::zero()
        }
        match self.pattern.orientation {
            Orientation::Csr => {
                for r in 0..self.nrows() {
                    for k in
                        self.pattern.offsets[r].to_usize()..self.pattern.offsets[r + 1].to_usize()
                    {
                        out[r] += self.values[k] * x[self.pattern.indices[k].to_usize()]
                    }
                }
            }
            Orientation::Csc => {
                for c in 0..self.ncols() {
                    for k in
                        self.pattern.offsets[c].to_usize()..self.pattern.offsets[c + 1].to_usize()
                    {
                        out[self.pattern.indices[k].to_usize()] += self.values[k] * x[c]
                    }
                }
            }
        }
        Ok(())
    }
}

pub trait Jacobian<S: Scalar>: Send + Sync {
    fn n(&self) -> usize;
    fn residual(&self, ctx: &Ctx, x: &[S], out: &mut [S]) -> Result<(), NumericError>;
    fn jvp(&self, ctx: &Ctx, x: &[S], v: &[S], out: &mut [S]) -> Result<(), NumericError> {
        let mut matrix = dense_zero::<S>(self.n());
        self.assemble_into(ctx, x, &mut matrix)?;
        matrix.apply(v, out)
    }
    fn assemble_into(
        &self,
        ctx: &Ctx,
        x: &[S],
        out: &mut SparseMatrix<S>,
    ) -> Result<(), NumericError>;
}

pub fn dense_zero<S: Scalar>(n: usize) -> SparseMatrix<S> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            coo.push(i, j, S::zero())
        }
    }
    coo.finish_csr()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorCoeffs<S> {
    pub mass: S,
    pub damp: S,
    pub stiff: S,
}
impl<S: Scalar> IntegratorCoeffs<S> {
    pub fn bdf(c: S) -> Self {
        Self {
            mass: c,
            damp: S::zero(),
            stiff: S::one(),
        }
    }
    pub fn generalized_alpha(m: S, d: S, k: S) -> Self {
        Self {
            mass: m,
            damp: d,
            stiff: k,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DaeIndex {
    Ode,
    Index1,
    Index2,
    Index3,
    HigherReduced,
}

pub trait DaeResidual<S: Scalar>: Jacobian<S> {
    fn residual_at(&self, ctx: &Ctx, t: f64, x: &[S], out: &mut [S]) -> Result<(), NumericError>;
    fn charge(&self, ctx: &Ctx, t: f64, x: &[S], out: &mut [S]) -> Result<(), NumericError>;
    fn mass_apply(
        &self,
        ctx: &Ctx,
        t: f64,
        x: &[S],
        v: &[S],
        out: &mut [S],
    ) -> Result<(), NumericError>;
    fn iteration_matrix(
        &self,
        ctx: &Ctx,
        t: f64,
        x: &[S],
        coeffs: &IntegratorCoeffs<S>,
        out: &mut SparseMatrix<S>,
    ) -> Result<(), NumericError> {
        let _ = (ctx, t, x, coeffs, out);
        Err(NumericError::Unsupported {
            what: "assembled DAE iteration matrix".into(),
        })
    }
    fn dae_index_hint(&self) -> DaeIndex;
}

impl fmt::Display for DaeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coo_is_deterministic_and_sums_duplicates() {
        let mut c = CooMatrix::<f64>::new(2, 2);
        c.push(1, 0, 2.);
        c.push(0, 1, 3.);
        c.push(1, 0, 4.);
        let m = c.finish_csr();
        assert_eq!(m.values, vec![3., 6.]);
        assert_eq!(m.pattern.offsets, vec![0, 1, 2]);
    }
    #[test]
    fn dense_contract_applies() {
        let m = dense_zero::<f64>(2);
        let mut y = vec![1.; 2];
        m.apply(&[2., 3.], &mut y).unwrap();
        assert_eq!(y, vec![0., 0.]);
    }
}
