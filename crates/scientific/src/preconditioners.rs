use crate::{BlockLayout, BlockPreconditioner};
use solverang_contracts::{Ctx, NumericError};

/// Physics-neutral block-diagonal scaling preconditioner. `inverse_diagonal` is laid out in
/// the same global order as `BlockLayout`; callers may fill it from any field/block Jacobian
/// approximation without Solverang learning the field's physical meaning.
#[derive(Clone, Debug)]
pub struct BlockDiagonalPreconditioner {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
}

impl BlockDiagonalPreconditioner {
    pub fn new(layout: BlockLayout, inverse_diagonal: Vec<f64>) -> Result<Self, NumericError> {
        if inverse_diagonal.len() != layout.dimension
            || inverse_diagonal.iter().any(|x| !x.is_finite())
        {
            return Err(NumericError::DimensionMismatch {
                expected: layout.dimension,
                got: inverse_diagonal.len(),
            });
        }
        Ok(Self {
            layout,
            inverse_diagonal,
        })
    }
}

impl BlockPreconditioner for BlockDiagonalPreconditioner {
    fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    fn apply_inverse(&self, _ctx: &Ctx, rhs: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        if rhs.len() != self.layout.dimension || out.len() != self.layout.dimension {
            return Err(NumericError::DimensionMismatch {
                expected: self.layout.dimension,
                got: rhs.len(),
            });
        }
        for ((y, b), inv) in out.iter_mut().zip(rhs).zip(&self.inverse_diagonal) {
            *y = b * inv;
        }
        Ok(())
    }
}

/// Sparse lower-triangular block coupling used as a forward-substitution preconditioner.
/// Each off-diagonal entry is an explicit dense row-major block `(row_block, col_block, data)`.
/// Diagonal inversion remains an elementwise approximation, so construction is cheap and fully
/// generic; richer exact block solvers can still implement `BlockPreconditioner` directly.
#[derive(Clone, Debug)]
pub struct BlockLowerTriangularPreconditioner {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
    lower_blocks: Vec<LowerBlock>,
}

#[derive(Clone, Debug)]
pub struct LowerBlock {
    pub row_block: usize,
    pub col_block: usize,
    pub values: Vec<f64>,
}

impl BlockLowerTriangularPreconditioner {
    pub fn new(
        layout: BlockLayout,
        inverse_diagonal: Vec<f64>,
        lower_blocks: Vec<LowerBlock>,
    ) -> Result<Self, NumericError> {
        if inverse_diagonal.len() != layout.dimension {
            return Err(NumericError::DimensionMismatch {
                expected: layout.dimension,
                got: inverse_diagonal.len(),
            });
        }
        for block in &lower_blocks {
            if block.row_block >= layout.blocks.len() || block.col_block >= block.row_block {
                return Err(NumericError::Unsupported {
                    what: "invalid lower-triangular block index".into(),
                });
            }
            let rows = layout.blocks[block.row_block].len;
            let cols = layout.blocks[block.col_block].len;
            if block.values.len() != rows * cols {
                return Err(NumericError::DimensionMismatch {
                    expected: rows * cols,
                    got: block.values.len(),
                });
            }
        }
        Ok(Self {
            layout,
            inverse_diagonal,
            lower_blocks,
        })
    }
}

impl BlockPreconditioner for BlockLowerTriangularPreconditioner {
    fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    fn apply_inverse(&self, _ctx: &Ctx, rhs: &[f64], out: &mut [f64]) -> Result<(), NumericError> {
        if rhs.len() != self.layout.dimension || out.len() != self.layout.dimension {
            return Err(NumericError::DimensionMismatch {
                expected: self.layout.dimension,
                got: rhs.len(),
            });
        }
        out.fill(0.0);
        for row_block in 0..self.layout.blocks.len() {
            let row = &self.layout.blocks[row_block];
            let mut local = rhs[row.offset..row.offset + row.len].to_vec();
            for block in self
                .lower_blocks
                .iter()
                .filter(|b| b.row_block == row_block)
            {
                let col = &self.layout.blocks[block.col_block];
                for (i, local_i) in local.iter_mut().enumerate().take(row.len) {
                    let correction = (0..col.len)
                        .map(|j| block.values[i * col.len + j] * out[col.offset + j])
                        .sum::<f64>();
                    *local_i -= correction;
                }
            }
            for (i, value) in local.into_iter().enumerate() {
                out[row.offset + i] = value * self.inverse_diagonal[row.offset + i];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockSpec;

    fn layout() -> BlockLayout {
        BlockLayout::new(vec![
            BlockSpec {
                name: "a".into(),
                offset: 0,
                len: 1,
                scale: 1.0,
            },
            BlockSpec {
                name: "b".into(),
                offset: 1,
                len: 1,
                scale: 1.0,
            },
        ])
        .unwrap()
    }

    #[test]
    fn diagonal_and_triangular_are_distinct_physics_neutral_actions() {
        let ctx = Ctx::reproducible();
        let d = BlockDiagonalPreconditioner::new(layout(), vec![0.5, 0.25]).unwrap();
        let mut out = vec![0.0; 2];
        d.apply_inverse(&ctx, &[2.0, 8.0], &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);

        let t = BlockLowerTriangularPreconditioner::new(
            layout(),
            vec![0.5, 0.25],
            vec![LowerBlock {
                row_block: 1,
                col_block: 0,
                values: vec![2.0],
            }],
        )
        .unwrap();
        t.apply_inverse(&ctx, &[2.0, 8.0], &mut out).unwrap();
        assert_eq!(out, vec![1.0, 1.5]);
    }
}
