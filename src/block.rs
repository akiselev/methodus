use std::collections::HashSet;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{LinearOperator, NonlinearOperator, Preconditioner, SolveError};

/// User-facing description of one contiguous solver block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockSpec {
    pub name: String,
    pub length: usize,
    pub residual_scale: f64,
}

/// A block with its resolved position in the global vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    name: String,
    start: usize,
    length: usize,
    residual_scale: f64,
}

impl Block {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    #[must_use]
    pub const fn residual_scale(&self) -> f64 {
        self.residual_scale
    }

    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.length
    }
}

/// Validated, contiguous partition of a solver vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockLayout {
    blocks: Vec<Block>,
    dimension: usize,
}

impl BlockLayout {
    pub fn new(specifications: Vec<BlockSpec>) -> Result<Self, SolveError> {
        if specifications.is_empty() {
            return Err(SolveError::InvalidLayout {
                reason: "at least one block is required".into(),
            });
        }

        let mut names = HashSet::new();
        let mut blocks = Vec::with_capacity(specifications.len());
        let mut start = 0usize;
        for specification in specifications {
            if specification.name.trim().is_empty() {
                return Err(SolveError::InvalidLayout {
                    reason: "block names must not be empty".into(),
                });
            }
            if !names.insert(specification.name.clone()) {
                return Err(SolveError::InvalidLayout {
                    reason: format!("duplicate block name `{}`", specification.name),
                });
            }
            if specification.length == 0 {
                return Err(SolveError::InvalidLayout {
                    reason: format!("block `{}` has zero length", specification.name),
                });
            }
            if !specification.residual_scale.is_finite() || specification.residual_scale <= 0.0 {
                return Err(SolveError::InvalidLayout {
                    reason: format!(
                        "block `{}` must have a finite positive residual scale",
                        specification.name
                    ),
                });
            }
            let next = start.checked_add(specification.length).ok_or_else(|| {
                SolveError::InvalidLayout {
                    reason: "block dimensions overflow usize".into(),
                }
            })?;
            blocks.push(Block {
                name: specification.name,
                start,
                length: specification.length,
                residual_scale: specification.residual_scale,
            });
            start = next;
        }
        Ok(Self {
            blocks,
            dimension: start,
        })
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    #[must_use]
    pub fn block(&self, index: usize) -> Option<&Block> {
        self.blocks.get(index)
    }
}

/// A nonlinear operator whose vector has a meaningful block partition.
pub trait BlockNonlinearOperator: NonlinearOperator {
    fn block_layout(&self) -> &BlockLayout;
}

/// A linear operator whose input and output share a block partition.
pub trait BlockLinearOperator: LinearOperator {
    fn block_layout(&self) -> &BlockLayout;
}

/// A preconditioner whose action respects a block partition.
pub trait BlockPreconditioner: Preconditioner {
    fn block_layout(&self) -> &BlockLayout;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_assigns_contiguous_ranges() {
        let layout = BlockLayout::new(vec![
            BlockSpec {
                name: "a".into(),
                length: 2,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: "b".into(),
                length: 3,
                residual_scale: 4.0,
            },
        ])
        .unwrap();
        assert_eq!(layout.dimension(), 5);
        assert_eq!(layout.blocks()[0].range(), 0..2);
        assert_eq!(layout.blocks()[1].range(), 2..5);
    }

    #[test]
    fn layout_rejects_duplicate_names() {
        let error = BlockLayout::new(vec![
            BlockSpec {
                name: "same".into(),
                length: 1,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: "same".into(),
                length: 1,
                residual_scale: 1.0,
            },
        ])
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidLayout { .. }));
    }
}
