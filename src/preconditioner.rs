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
}
