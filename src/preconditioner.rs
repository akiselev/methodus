use serde::{Deserialize, Serialize};

use crate::{BlockLayout, BlockPreconditioner, EvaluationContext, NumericError, Preconditioner};

/// Elementwise inverse diagonal organized by a validated block layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "BlockDiagonalPreconditionerData")]
pub struct BlockDiagonalPreconditioner {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
}

#[derive(Deserialize)]
struct BlockDiagonalPreconditionerData {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
}

impl TryFrom<BlockDiagonalPreconditionerData> for BlockDiagonalPreconditioner {
    type Error = NumericError;

    fn try_from(data: BlockDiagonalPreconditionerData) -> Result<Self, Self::Error> {
        Self::new(data.layout, data.inverse_diagonal)
    }
}

impl BlockDiagonalPreconditioner {
    pub fn new(layout: BlockLayout, inverse_diagonal: Vec<f64>) -> Result<Self, NumericError> {
        NumericError::require_len("block diagonal", inverse_diagonal.len(), layout.dimension())?;
        NumericError::require_finite("block diagonal", &inverse_diagonal)?;
        Ok(Self {
            layout,
            inverse_diagonal,
        })
    }
}

impl Preconditioner for BlockDiagonalPreconditioner {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn apply_inverse(
        &self,
        _context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len(
            "block diagonal right-hand side",
            right_hand_side.len(),
            self.dimension(),
        )?;
        NumericError::require_len("block diagonal output", output.len(), self.dimension())?;
        NumericError::require_finite("block diagonal right-hand side", right_hand_side)?;
        for ((result, value), inverse) in output
            .iter_mut()
            .zip(right_hand_side)
            .zip(&self.inverse_diagonal)
        {
            *result = value * inverse;
        }
        NumericError::require_finite("block diagonal output", output)
    }
}

impl BlockPreconditioner for BlockDiagonalPreconditioner {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

/// Dense row-major coupling from an earlier block into a later block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowerBlock {
    pub row_block: usize,
    pub column_block: usize,
    pub values: Vec<f64>,
}

/// Forward-substitution preconditioner with elementwise diagonal inverses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "BlockLowerTriangularPreconditionerData")]
pub struct BlockLowerTriangularPreconditioner {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
    lower_blocks: Vec<LowerBlock>,
}

#[derive(Deserialize)]
struct BlockLowerTriangularPreconditionerData {
    layout: BlockLayout,
    inverse_diagonal: Vec<f64>,
    lower_blocks: Vec<LowerBlock>,
}

impl TryFrom<BlockLowerTriangularPreconditionerData> for BlockLowerTriangularPreconditioner {
    type Error = NumericError;

    fn try_from(data: BlockLowerTriangularPreconditionerData) -> Result<Self, Self::Error> {
        Self::new(data.layout, data.inverse_diagonal, data.lower_blocks)
    }
}

impl BlockLowerTriangularPreconditioner {
    pub fn new(
        layout: BlockLayout,
        inverse_diagonal: Vec<f64>,
        lower_blocks: Vec<LowerBlock>,
    ) -> Result<Self, NumericError> {
        NumericError::require_len(
            "block lower-triangular diagonal",
            inverse_diagonal.len(),
            layout.dimension(),
        )?;
        NumericError::require_finite("block lower-triangular diagonal", &inverse_diagonal)?;
        for (index, block) in lower_blocks.iter().enumerate() {
            if block.row_block >= layout.blocks().len() || block.column_block >= block.row_block {
                return Err(NumericError::InvalidInput {
                    message: format!("lower block {index} is not strictly below the diagonal"),
                });
            }
            let rows = layout.blocks()[block.row_block].length();
            let columns = layout.blocks()[block.column_block].length();
            let expected_values =
                rows.checked_mul(columns)
                    .ok_or_else(|| NumericError::InvalidInput {
                        message: format!("lower block {index} dimensions overflow usize"),
                    })?;
            NumericError::require_len(
                &format!("lower block {index}"),
                block.values.len(),
                expected_values,
            )?;
            NumericError::require_finite(&format!("lower block {index}"), &block.values)?;
        }
        Ok(Self {
            layout,
            inverse_diagonal,
            lower_blocks,
        })
    }
}

impl Preconditioner for BlockLowerTriangularPreconditioner {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn apply_inverse(
        &self,
        _context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len(
            "block lower-triangular right-hand side",
            right_hand_side.len(),
            self.dimension(),
        )?;
        NumericError::require_len(
            "block lower-triangular output",
            output.len(),
            self.dimension(),
        )?;
        NumericError::require_finite("block lower-triangular right-hand side", right_hand_side)?;
        output.fill(0.0);
        for row_block in 0..self.layout.blocks().len() {
            let row = &self.layout.blocks()[row_block];
            let mut local = right_hand_side[row.range()].to_vec();
            for block in self
                .lower_blocks
                .iter()
                .filter(|block| block.row_block == row_block)
            {
                let column = &self.layout.blocks()[block.column_block];
                for (local_row, local_value) in local.iter_mut().enumerate() {
                    let correction = (0..column.length())
                        .map(|local_column| {
                            block.values[local_row * column.length() + local_column]
                                * output[column.start() + local_column]
                        })
                        .sum::<f64>();
                    *local_value -= correction;
                }
            }
            for (local_row, value) in local.into_iter().enumerate() {
                let global_row = row.start() + local_row;
                output[global_row] = value * self.inverse_diagonal[global_row];
            }
        }
        NumericError::require_finite("block lower-triangular output", output)
    }
}

impl BlockPreconditioner for BlockLowerTriangularPreconditioner {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

/// Block-diagonal composition of caller-supplied per-block preconditioners.
///
/// SV2-B6's bounded reference implementation of the block preconditioner
/// contract a Schur-complement/pressure-mass saddle-point shape needs (e.g.
/// Stokes): each block of a [`BlockLayout`] is preconditioned independently
/// by a caller-supplied [`Preconditioner`] — a velocity-block approximation
/// composed block-diagonally with a pressure-mass or Schur-complement
/// approximation, with no coupling between blocks. This is not a full
/// preconditioner library; callers construct whatever per-block
/// approximation their operator needs (including a nested
/// [`BlockDiagonalPreconditioner`] or [`BlockLowerTriangularPreconditioner`])
/// and compose it here.
pub struct CompositeBlockPreconditioner<'a> {
    layout: BlockLayout,
    blocks: Vec<&'a dyn Preconditioner>,
}

impl std::fmt::Debug for CompositeBlockPreconditioner<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompositeBlockPreconditioner")
            .field("layout", &self.layout)
            .field("block_count", &self.blocks.len())
            .finish()
    }
}

impl<'a> CompositeBlockPreconditioner<'a> {
    /// Builds a validated block-diagonal composition.
    ///
    /// # Errors
    /// Refuses a block count that does not match the layout, or any block
    /// whose dimension does not match its layout entry's length.
    pub fn new(
        layout: BlockLayout,
        blocks: Vec<&'a dyn Preconditioner>,
    ) -> Result<Self, NumericError> {
        NumericError::require_len(
            "composite block preconditioner block count",
            blocks.len(),
            layout.blocks().len(),
        )?;
        for (index, (block, spec)) in blocks.iter().zip(layout.blocks()).enumerate() {
            if block.dimension() != spec.length() {
                return Err(NumericError::DimensionMismatch {
                    operation: format!(
                        "composite block preconditioner block {index} (`{}`)",
                        spec.name()
                    ),
                    expected: spec.length(),
                    actual: block.dimension(),
                });
            }
        }
        Ok(Self { layout, blocks })
    }
}

impl Preconditioner for CompositeBlockPreconditioner<'_> {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn apply_inverse(
        &self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len(
            "composite block preconditioner right-hand side",
            right_hand_side.len(),
            self.dimension(),
        )?;
        NumericError::require_len(
            "composite block preconditioner output",
            output.len(),
            self.dimension(),
        )?;
        NumericError::require_finite(
            "composite block preconditioner right-hand side",
            right_hand_side,
        )?;
        for (block, spec) in self.blocks.iter().zip(self.layout.blocks()) {
            let range = spec.range();
            block.apply_inverse(context, &right_hand_side[range.clone()], &mut output[range])?;
        }
        NumericError::require_finite("composite block preconditioner output", output)
    }
}

impl BlockPreconditioner for CompositeBlockPreconditioner<'_> {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
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
                length: 1,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: "b".into(),
                length: 1,
                residual_scale: 1.0,
            },
        ])
        .unwrap()
    }

    #[test]
    fn diagonal_and_lower_triangular_actions_are_distinct() {
        let context = EvaluationContext::reproducible();
        let diagonal = BlockDiagonalPreconditioner::new(layout(), vec![0.5, 0.25]).unwrap();
        let mut output = vec![0.0; 2];
        diagonal
            .apply_inverse(&context, &[2.0, 8.0], &mut output)
            .unwrap();
        assert_eq!(output, vec![1.0, 2.0]);

        let triangular = BlockLowerTriangularPreconditioner::new(
            layout(),
            vec![0.5, 0.25],
            vec![LowerBlock {
                row_block: 1,
                column_block: 0,
                values: vec![2.0],
            }],
        )
        .unwrap();
        triangular
            .apply_inverse(&context, &[2.0, 8.0], &mut output)
            .unwrap();
        assert_eq!(output, vec![1.0, 1.5]);
    }

    #[test]
    fn deserialization_revalidates_preconditioner_dimensions() {
        let malformed = r#"{
            "layout": {
                "blocks": [{
                    "name": "a",
                    "start": 0,
                    "length": 1,
                    "residual_scale": 1.0
                }],
                "dimension": 1
            },
            "inverse_diagonal": []
        }"#;
        assert!(serde_json::from_str::<BlockDiagonalPreconditioner>(malformed).is_err());
    }

    #[test]
    fn composite_block_preconditioner_applies_each_block_independently() {
        // A saddle-point-shaped 3-block layout: two "velocity" blocks and
        // one "pressure" block, each preconditioned by an unrelated
        // caller-supplied approximation (e.g. a pressure-mass diagonal).
        let saddle_layout = BlockLayout::new(vec![
            BlockSpec {
                name: "velocity".into(),
                length: 2,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: "pressure".into(),
                length: 1,
                residual_scale: 1.0,
            },
        ])
        .unwrap();
        let velocity = BlockDiagonalPreconditioner::new(
            BlockLayout::new(vec![BlockSpec {
                name: "velocity".into(),
                length: 2,
                residual_scale: 1.0,
            }])
            .unwrap(),
            vec![0.5, 0.25],
        )
        .unwrap();
        let pressure = BlockDiagonalPreconditioner::new(
            BlockLayout::new(vec![BlockSpec {
                name: "pressure".into(),
                length: 1,
                residual_scale: 1.0,
            }])
            .unwrap(),
            vec![2.0],
        )
        .unwrap();
        let composite =
            CompositeBlockPreconditioner::new(saddle_layout, vec![&velocity, &pressure]).unwrap();
        let mut output = vec![0.0; 3];
        composite
            .apply_inverse(
                &EvaluationContext::reproducible(),
                &[2.0, 8.0, 3.0],
                &mut output,
            )
            .unwrap();
        assert_eq!(output, vec![1.0, 2.0, 6.0]);
    }

    #[test]
    fn composite_block_preconditioner_refuses_mismatched_block_dimensions() {
        let mismatched = BlockDiagonalPreconditioner::new(
            BlockLayout::new(vec![BlockSpec {
                name: "wrong".into(),
                length: 3,
                residual_scale: 1.0,
            }])
            .unwrap(),
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        let matching = BlockDiagonalPreconditioner::new(
            BlockLayout::new(vec![BlockSpec {
                name: "b".into(),
                length: 1,
                residual_scale: 1.0,
            }])
            .unwrap(),
            vec![1.0],
        )
        .unwrap();
        let error =
            CompositeBlockPreconditioner::new(layout(), vec![&mismatched, &matching]).unwrap_err();
        assert!(matches!(error, NumericError::DimensionMismatch { .. }));
    }

    #[test]
    fn composite_block_preconditioner_refuses_a_block_count_mismatch() {
        let single = BlockDiagonalPreconditioner::new(
            BlockLayout::new(vec![BlockSpec {
                name: "a".into(),
                length: 1,
                residual_scale: 1.0,
            }])
            .unwrap(),
            vec![1.0],
        )
        .unwrap();
        let error = CompositeBlockPreconditioner::new(layout(), vec![&single]).unwrap_err();
        assert!(matches!(error, NumericError::DimensionMismatch { .. }));
    }
}
