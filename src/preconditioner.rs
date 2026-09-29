use serde::{Deserialize, Serialize};

use crate::{
    BlockLayout, BlockPreconditioner, EvaluationContext, NumericError, OperatorSymmetry,
    Preconditioner,
};

/// Builds the existing block-diagonal inverse from an owner-supplied exact
/// diagonal. No global probing, zero replacement or silent fallback.
#[derive(Clone, Copy, Debug)]
pub struct JacobiFactory;
impl crate::PreconditionerFactory for JacobiFactory {
    fn build<'a>(
        &'a self,
        context: &EvaluationContext,
        jacobian: &dyn crate::LinearOperator,
        _state: &[f64],
    ) -> Result<Option<Box<dyn Preconditioner + 'a>>, NumericError> {
        let diagonal = jacobian
            .diagonal(context)?
            .ok_or_else(|| NumericError::InvalidInput {
                message: "Jacobi requires an owner-supplied Jacobian diagonal".into(),
            })?;
        NumericError::require_len("Jacobi diagonal", diagonal.len(), jacobian.rows())?;
        NumericError::require_finite("Jacobi diagonal", &diagonal)?;
        let inverse = diagonal.iter().map(|value| 1.0 / value).collect::<Vec<_>>();
        NumericError::require_finite("Jacobi inverse", &inverse)?;
        let layout = BlockLayout::new(vec![crate::BlockSpec {
            name: "diagonal".into(),
            length: inverse.len(),
            residual_scale: 1.0,
        }])
        .map_err(|error| NumericError::InvalidInput {
            message: error.to_string(),
        })?;
        Ok(Some(Box::new(BlockDiagonalPreconditioner::new(
            layout, inverse,
        )?)))
    }
}

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

    // An elementwise diagonal scaling is symmetric under the Euclidean
    // inner product by construction.
    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Symmetric
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

    // Forward substitution over a nontrivial lower block is nonsymmetric in
    // general.
    fn symmetry(&self) -> OperatorSymmetry {
        OperatorSymmetry::Nonsymmetric
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

/// Off-diagonal block coupling action for [`BlockGaussSeidelPreconditioner`]:
/// the action of `A_{row_block,column_block}` from one block onto another.
/// Multiple actions may name the same `(row_block, column_block)` pair;
/// their contributions sum.
pub trait BlockCouplingAction: Send + Sync {
    fn row_block(&self) -> usize;
    fn column_block(&self) -> usize;
    /// Writes `A_{row_block,column_block} · input` into `output`
    /// (overwriting, not accumulating): `input` has the column block's
    /// length, `output` has the row block's length.
    fn apply(
        &self,
        context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError>;
}

/// Sweep pattern for [`BlockGaussSeidelPreconditioner`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GaussSeidelSweep {
    /// Blocks visited `0..n`: a block-lower-triangular preconditioner when
    /// every supplied coupling has `column_block < row_block`.
    Forward,
    /// Blocks visited `n..0`: a block-upper-triangular preconditioner when
    /// every supplied coupling has `column_block > row_block`.
    Backward,
    /// A forward sweep followed by a backward sweep, each continuing from
    /// the other's result (symmetric/SSOR-style Gauss-Seidel).
    Symmetric,
}

/// Block Gauss-Seidel / block-triangular composite preconditioner: an
/// owner-supplied block layout, an owner-supplied approximate inverse
/// ([`Preconditioner`]) per diagonal block — the existing block-diagonal
/// (Jacobi) preconditioner or an owner inner Krylov solve, e.g. — and
/// owner-supplied off-diagonal block actions ([`BlockCouplingAction`]). One
/// application performs a multiplicative block sweep: each visited block's
/// local right-hand side is corrected by the off-diagonal contributions
/// from the blocks already updated in this sweep (blocks not yet visited
/// keep the previous sweep's value, or zero on the first visit), then
/// solved by that block's own diagonal preconditioner. No global probing,
/// zero replacement or silent fallback is used.
///
/// [`GaussSeidelSweep::Forward`] over a strictly lower coupling set with
/// exact per-block inverses solves a block-lower-triangular system exactly
/// in one application; [`GaussSeidelSweep::Backward`] does the same for a
/// strictly upper coupling set.
pub struct BlockGaussSeidelPreconditioner<'a> {
    layout: BlockLayout,
    diagonal_solves: Vec<&'a dyn Preconditioner>,
    couplings: Vec<&'a dyn BlockCouplingAction>,
    sweep: GaussSeidelSweep,
}

impl std::fmt::Debug for BlockGaussSeidelPreconditioner<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlockGaussSeidelPreconditioner")
            .field("layout", &self.layout)
            .field("diagonal_block_count", &self.diagonal_solves.len())
            .field("coupling_count", &self.couplings.len())
            .field("sweep", &self.sweep)
            .finish()
    }
}

impl<'a> BlockGaussSeidelPreconditioner<'a> {
    /// Builds a validated block Gauss-Seidel / block-triangular
    /// preconditioner.
    ///
    /// # Errors
    /// Refuses a diagonal-solve count that does not match the layout, any
    /// diagonal solve whose dimension does not match its block's length,
    /// and any coupling naming an out-of-range block or coupling a block
    /// to itself.
    pub fn new(
        layout: BlockLayout,
        diagonal_solves: Vec<&'a dyn Preconditioner>,
        couplings: Vec<&'a dyn BlockCouplingAction>,
        sweep: GaussSeidelSweep,
    ) -> Result<Self, NumericError> {
        NumericError::require_len(
            "block Gauss-Seidel diagonal-solve count",
            diagonal_solves.len(),
            layout.blocks().len(),
        )?;
        for (index, (solve, spec)) in diagonal_solves.iter().zip(layout.blocks()).enumerate() {
            if solve.dimension() != spec.length() {
                return Err(NumericError::DimensionMismatch {
                    operation: format!(
                        "block Gauss-Seidel diagonal solve {index} (`{}`)",
                        spec.name()
                    ),
                    expected: spec.length(),
                    actual: solve.dimension(),
                });
            }
        }
        for (index, coupling) in couplings.iter().enumerate() {
            let row = coupling.row_block();
            let column = coupling.column_block();
            if row >= layout.blocks().len() || column >= layout.blocks().len() {
                return Err(NumericError::InvalidInput {
                    message: format!(
                        "block Gauss-Seidel coupling {index} names an out-of-range block \
                         (row {row}, column {column}, {} blocks declared)",
                        layout.blocks().len()
                    ),
                });
            }
            if row == column {
                return Err(NumericError::InvalidInput {
                    message: format!(
                        "block Gauss-Seidel coupling {index} couples block {row} to itself; \
                         diagonal contributions belong in `diagonal_solves`"
                    ),
                });
            }
        }
        Ok(Self {
            layout,
            diagonal_solves,
            couplings,
            sweep,
        })
    }

    fn apply_sweep(
        &self,
        context: &EvaluationContext,
        order: &[usize],
        right_hand_side: &[f64],
        state: &mut [f64],
    ) -> Result<(), NumericError> {
        for &row_block in order {
            let block = &self.layout.blocks()[row_block];
            let range = block.range();
            let mut local = right_hand_side[range.clone()].to_vec();
            for coupling in self
                .couplings
                .iter()
                .filter(|coupling| coupling.row_block() == row_block)
            {
                let column = &self.layout.blocks()[coupling.column_block()];
                let mut correction = vec![0.0; block.length()];
                coupling.apply(context, &state[column.range()], &mut correction)?;
                NumericError::require_finite("block Gauss-Seidel coupling action", &correction)?;
                for (value, term) in local.iter_mut().zip(&correction) {
                    *value -= term;
                }
            }
            NumericError::require_finite("block Gauss-Seidel local right-hand side", &local)?;
            self.diagonal_solves[row_block].apply_inverse(context, &local, &mut state[range])?;
        }
        Ok(())
    }
}

impl Preconditioner for BlockGaussSeidelPreconditioner<'_> {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    // Only a symmetric sweep whose every diagonal solve is itself declared
    // symmetric earns a symmetric declaration; forward/backward
    // (triangular) sweeps are nonsymmetric in general.
    fn symmetry(&self) -> OperatorSymmetry {
        if self.sweep == GaussSeidelSweep::Symmetric
            && self
                .diagonal_solves
                .iter()
                .all(|solve| solve.symmetry() == OperatorSymmetry::Symmetric)
        {
            OperatorSymmetry::Symmetric
        } else {
            OperatorSymmetry::Nonsymmetric
        }
    }

    fn apply_inverse(
        &self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len(
            "block Gauss-Seidel right-hand side",
            right_hand_side.len(),
            self.dimension(),
        )?;
        NumericError::require_len("block Gauss-Seidel output", output.len(), self.dimension())?;
        NumericError::require_finite("block Gauss-Seidel right-hand side", right_hand_side)?;
        output.fill(0.0);
        let block_count = self.layout.blocks().len();
        match self.sweep {
            GaussSeidelSweep::Forward => {
                let order: Vec<usize> = (0..block_count).collect();
                self.apply_sweep(context, &order, right_hand_side, output)?;
            }
            GaussSeidelSweep::Backward => {
                let order: Vec<usize> = (0..block_count).rev().collect();
                self.apply_sweep(context, &order, right_hand_side, output)?;
            }
            GaussSeidelSweep::Symmetric => {
                let forward: Vec<usize> = (0..block_count).collect();
                let backward: Vec<usize> = (0..block_count).rev().collect();
                self.apply_sweep(context, &forward, right_hand_side, output)?;
                self.apply_sweep(context, &backward, right_hand_side, output)?;
            }
        }
        NumericError::require_finite("block Gauss-Seidel output", output)
    }
}

impl BlockPreconditioner for BlockGaussSeidelPreconditioner<'_> {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BlockSpec, ConjugateGradientConfig, GmresConfig, LinearOperator, SolveError,
        solve_conjugate_gradient, solve_gmres,
    };

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

    // --- Block Gauss-Seidel / block-triangular preconditioner ---

    /// A dense (small, test-only) block coupling action.
    struct DenseCoupling {
        row: usize,
        column: usize,
        rows: Vec<Vec<f64>>,
    }

    impl BlockCouplingAction for DenseCoupling {
        fn row_block(&self) -> usize {
            self.row
        }

        fn column_block(&self) -> usize {
            self.column
        }

        fn apply(
            &self,
            _context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            for (row, value) in self.rows.iter().zip(output.iter_mut()) {
                *value = row.iter().zip(input).map(|(a, b)| a * b).sum();
            }
            Ok(())
        }
    }

    /// An exact dense diagonal-block solve (Gaussian elimination), for
    /// testing exactness independent of an approximate inner solve.
    struct DenseBlockSolve {
        rows: Vec<Vec<f64>>,
        declared_symmetry: OperatorSymmetry,
    }

    impl Preconditioner for DenseBlockSolve {
        fn dimension(&self) -> usize {
            self.rows.len()
        }

        fn symmetry(&self) -> OperatorSymmetry {
            self.declared_symmetry
        }

        fn apply_inverse(
            &self,
            _context: &EvaluationContext,
            right_hand_side: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            let solution =
                crate::nonlinear::solve_dense(self.rows.clone(), right_hand_side.to_vec())
                    .map_err(|error| NumericError::InvalidInput {
                        message: error.to_string(),
                    })?;
            output.copy_from_slice(&solution);
            Ok(())
        }
    }

    /// A dense full-system operator for exercising Krylov solves against a
    /// known block-coupled matrix.
    struct DenseSystem {
        rows: Vec<Vec<f64>>,
        symmetry: OperatorSymmetry,
    }

    impl LinearOperator for DenseSystem {
        fn rows(&self) -> usize {
            self.rows.len()
        }

        fn columns(&self) -> usize {
            self.rows.len()
        }

        fn symmetry(&self) -> OperatorSymmetry {
            self.symmetry
        }

        fn apply(
            &self,
            _context: &EvaluationContext,
            input: &[f64],
            output: &mut [f64],
        ) -> Result<(), NumericError> {
            for (row, value) in self.rows.iter().zip(output.iter_mut()) {
                *value = row.iter().zip(input).map(|(a, b)| a * b).sum();
            }
            Ok(())
        }
    }

    fn two_scalar_block_layout() -> BlockLayout {
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
    fn forward_gauss_seidel_solves_a_block_lower_triangular_system_exactly() {
        // A = [[2, 0], [1.5, 3]]; strictly-lower coupling only.
        let diagonal_a = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let diagonal_b = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let lower = DenseCoupling {
            row: 1,
            column: 0,
            rows: vec![vec![1.5]],
        };
        let preconditioner = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&diagonal_a, &diagonal_b],
            vec![&lower],
            GaussSeidelSweep::Forward,
        )
        .unwrap();
        let mut output = vec![0.0; 2];
        preconditioner
            .apply_inverse(
                &EvaluationContext::reproducible(),
                &[4.0, 10.0],
                &mut output,
            )
            .unwrap();
        // x0 = 4/2 = 2; x1 = (10 - 1.5*2)/3 = 7/3.
        assert!((output[0] - 2.0).abs() < 1.0e-12);
        assert!((output[1] - 7.0 / 3.0).abs() < 1.0e-12);
        // The action A*x reproduces the right-hand side exactly.
        let system = DenseSystem {
            rows: vec![vec![2.0, 0.0], vec![1.5, 3.0]],
            symmetry: OperatorSymmetry::Nonsymmetric,
        };
        let mut reproduced = vec![0.0; 2];
        system
            .apply(&EvaluationContext::reproducible(), &output, &mut reproduced)
            .unwrap();
        assert!((reproduced[0] - 4.0).abs() < 1.0e-10);
        assert!((reproduced[1] - 10.0).abs() < 1.0e-10);
    }

    #[test]
    fn backward_gauss_seidel_solves_a_block_upper_triangular_system_exactly() {
        // A = [[2, 1.5], [0, 3]]; strictly-upper coupling only.
        let diagonal_a = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let diagonal_b = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let upper = DenseCoupling {
            row: 0,
            column: 1,
            rows: vec![vec![1.5]],
        };
        let preconditioner = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&diagonal_a, &diagonal_b],
            vec![&upper],
            GaussSeidelSweep::Backward,
        )
        .unwrap();
        let mut output = vec![0.0; 2];
        preconditioner
            .apply_inverse(
                &EvaluationContext::reproducible(),
                &[10.0, 6.0],
                &mut output,
            )
            .unwrap();
        // x1 = 6/3 = 2; x0 = (10 - 1.5*2)/2 = 3.5.
        assert!((output[1] - 2.0).abs() < 1.0e-12);
        assert!((output[0] - 3.5).abs() < 1.0e-12);
    }

    #[test]
    fn gauss_seidel_symmetry_declaration_follows_sweep_and_inner_solves() {
        let symmetric_diagonal = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let unknown_diagonal = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Unknown,
        };
        let coupling = DenseCoupling {
            row: 1,
            column: 0,
            rows: vec![vec![1.5]],
        };

        let forward = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&symmetric_diagonal, &symmetric_diagonal],
            vec![&coupling],
            GaussSeidelSweep::Forward,
        )
        .unwrap();
        assert_eq!(forward.symmetry(), OperatorSymmetry::Nonsymmetric);

        let symmetric_sweep_with_symmetric_inner = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&symmetric_diagonal, &symmetric_diagonal],
            vec![&coupling],
            GaussSeidelSweep::Symmetric,
        )
        .unwrap();
        assert_eq!(
            symmetric_sweep_with_symmetric_inner.symmetry(),
            OperatorSymmetry::Symmetric
        );

        let symmetric_sweep_with_unknown_inner = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&symmetric_diagonal, &unknown_diagonal],
            vec![&coupling],
            GaussSeidelSweep::Symmetric,
        )
        .unwrap();
        assert_eq!(
            symmetric_sweep_with_unknown_inner.symmetry(),
            OperatorSymmetry::Nonsymmetric
        );

        // Conjugate gradient refuses the forward (declared-nonsymmetric)
        // preconditioner outright, even over a symmetric positive-definite
        // operator that CG would otherwise accept.
        let spd_system = DenseSystem {
            rows: vec![vec![4.0, 1.0], vec![1.0, 4.0]],
            symmetry: OperatorSymmetry::Symmetric,
        };
        let error = solve_conjugate_gradient(
            &spd_system,
            Some(&forward),
            &EvaluationContext::reproducible(),
            &[1.0, 1.0],
            &[0.0, 0.0],
            &ConjugateGradientConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, SolveError::InvalidConfiguration { .. }));

        // The declared-symmetric symmetric-sweep preconditioner is accepted.
        solve_conjugate_gradient(
            &spd_system,
            Some(&symmetric_sweep_with_symmetric_inner),
            &EvaluationContext::reproducible(),
            &[1.0, 1.0],
            &[0.0, 0.0],
            &ConjugateGradientConfig::default(),
        )
        .unwrap();
    }

    #[test]
    fn gauss_seidel_refuses_a_diagonal_solve_count_mismatch() {
        let diagonal = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let error = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&diagonal],
            vec![],
            GaussSeidelSweep::Forward,
        )
        .unwrap_err();
        assert!(matches!(error, NumericError::DimensionMismatch { .. }));
    }

    #[test]
    fn gauss_seidel_refuses_a_diagonal_solve_dimension_mismatch() {
        let wrong = DenseBlockSolve {
            rows: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let right = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let error = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&wrong, &right],
            vec![],
            GaussSeidelSweep::Forward,
        )
        .unwrap_err();
        assert!(matches!(error, NumericError::DimensionMismatch { .. }));
    }

    #[test]
    fn gauss_seidel_refuses_an_out_of_range_or_self_coupling() {
        let a = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let b = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let out_of_range = DenseCoupling {
            row: 5,
            column: 0,
            rows: vec![vec![1.0]],
        };
        let error = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&a, &b],
            vec![&out_of_range],
            GaussSeidelSweep::Forward,
        )
        .unwrap_err();
        assert!(matches!(error, NumericError::InvalidInput { .. }));

        let self_coupled = DenseCoupling {
            row: 0,
            column: 0,
            rows: vec![vec![1.0]],
        };
        let error = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&a, &b],
            vec![&self_coupled],
            GaussSeidelSweep::Forward,
        )
        .unwrap_err();
        assert!(matches!(error, NumericError::InvalidInput { .. }));
    }

    #[test]
    fn gauss_seidel_refuses_nonfinite_right_hand_side() {
        let a = DenseBlockSolve {
            rows: vec![vec![2.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let b = DenseBlockSolve {
            rows: vec![vec![3.0]],
            declared_symmetry: OperatorSymmetry::Symmetric,
        };
        let preconditioner = BlockGaussSeidelPreconditioner::new(
            two_scalar_block_layout(),
            vec![&a, &b],
            vec![],
            GaussSeidelSweep::Forward,
        )
        .unwrap();
        let mut output = vec![0.0; 2];
        let error = preconditioner
            .apply_inverse(
                &EvaluationContext::reproducible(),
                &[f64::NAN, 1.0],
                &mut output,
            )
            .unwrap_err();
        assert!(matches!(error, NumericError::NonFinite { .. }));
    }

    /// Builds an `n`-block system of `block_length`-length blocks: strongly
    /// diagonally dominant per-block diagonals coupled by weaker
    /// off-diagonal blocks, so preconditioning measurably helps GMRES.
    fn coupled_dense_system(
        block_count: usize,
        block_length: usize,
    ) -> (Vec<Vec<f64>>, BlockLayout) {
        let dimension = block_count * block_length;
        let mut rows = vec![vec![0.0; dimension]; dimension];
        for block in 0..block_count {
            for local_row in 0..block_length {
                let global_row = block * block_length + local_row;
                for local_col in 0..block_length {
                    let global_col = block * block_length + local_col;
                    // Strong intra-block coupling: a scalar (elementwise)
                    // Jacobi preconditioner, which only sees the diagonal,
                    // is a poor approximation of this block's own inverse.
                    // The diagonal magnitude also varies by block so an
                    // elementwise Jacobi preconditioner is not simply a
                    // no-op uniform rescaling.
                    let diagonal = 5.0 + 2.0 * block as f64;
                    rows[global_row][global_col] = if local_row == local_col {
                        diagonal
                    } else {
                        3.5
                    };
                }
                for other in 0..block_count {
                    if other == block {
                        continue;
                    }
                    for local_col in 0..block_length {
                        let global_col = other * block_length + local_col;
                        // Weak inter-block coupling: what remains after an
                        // exact block-diagonal (Gauss-Seidel) solve.
                        let sign = if other > block { 1.0 } else { -1.0 };
                        rows[global_row][global_col] =
                            sign * 0.4 / (1 + local_row.abs_diff(local_col)) as f64;
                    }
                }
            }
        }
        let specs = (0..block_count)
            .map(|index| BlockSpec {
                name: format!("block{index}"),
                length: block_length,
                residual_scale: 1.0,
            })
            .collect();
        (rows, BlockLayout::new(specs).unwrap())
    }

    fn jacobi_preconditioner(
        layout: &BlockLayout,
        rows: &[Vec<f64>],
    ) -> BlockDiagonalPreconditioner {
        let inverse = (0..rows.len())
            .map(|index| 1.0 / rows[index][index])
            .collect();
        BlockDiagonalPreconditioner::new(layout.clone(), inverse).unwrap()
    }

    fn dense_block_solves(rows: &[Vec<f64>], layout: &BlockLayout) -> Vec<DenseBlockSolve> {
        layout
            .blocks()
            .iter()
            .map(|block| {
                let range = block.range();
                let sub_rows = range
                    .clone()
                    .map(|row| range.clone().map(|col| rows[row][col]).collect())
                    .collect();
                DenseBlockSolve {
                    rows: sub_rows,
                    declared_symmetry: OperatorSymmetry::Unknown,
                }
            })
            .collect()
    }

    fn dense_couplings(rows: &[Vec<f64>], layout: &BlockLayout) -> Vec<DenseCoupling> {
        let mut couplings = Vec::new();
        for (row_index, row_block) in layout.blocks().iter().enumerate() {
            for (column_index, column_block) in layout.blocks().iter().enumerate() {
                if row_index == column_index {
                    continue;
                }
                let sub_rows = row_block
                    .range()
                    .map(|row| column_block.range().map(|col| rows[row][col]).collect())
                    .collect();
                couplings.push(DenseCoupling {
                    row: row_index,
                    column: column_index,
                    rows: sub_rows,
                });
            }
        }
        couplings
    }

    fn gmres_iterations(
        rows: Vec<Vec<f64>>,
        preconditioner: Option<&dyn Preconditioner>,
        dimension: usize,
    ) -> usize {
        let system = DenseSystem {
            rows,
            symmetry: OperatorSymmetry::Nonsymmetric,
        };
        let right_hand_side = vec![1.0; dimension];
        let report = solve_gmres(
            &system,
            preconditioner,
            &EvaluationContext::reproducible(),
            &right_hand_side,
            &vec![0.0; dimension],
            &GmresConfig {
                max_iterations: 200,
                restart: 200,
                absolute_tolerance: 2.0e-3,
                relative_tolerance: 0.0,
            },
        )
        .unwrap();
        assert!(report.converged);
        report.trace.len() - 1
    }

    #[test]
    fn block_gauss_seidel_measurably_reduces_gmres_iterations_over_two_blocks() {
        let (rows, layout) = coupled_dense_system(2, 3);
        let none = gmres_iterations(rows.clone(), None, layout.dimension());
        let jacobi_preconditioner = jacobi_preconditioner(&layout, &rows);
        let jacobi = gmres_iterations(
            rows.clone(),
            Some(&jacobi_preconditioner),
            layout.dimension(),
        );
        let diagonal_solves = dense_block_solves(&rows, &layout);
        let diagonal_refs: Vec<&dyn Preconditioner> = diagonal_solves
            .iter()
            .map(|solve| solve as &dyn Preconditioner)
            .collect();
        let couplings = dense_couplings(&rows, &layout);
        let coupling_refs: Vec<&dyn BlockCouplingAction> = couplings
            .iter()
            .map(|coupling| coupling as &dyn BlockCouplingAction)
            .collect();
        let gauss_seidel = BlockGaussSeidelPreconditioner::new(
            layout.clone(),
            diagonal_refs,
            coupling_refs,
            GaussSeidelSweep::Forward,
        )
        .unwrap();
        let gs = gmres_iterations(rows, Some(&gauss_seidel), layout.dimension());
        // Both preconditioners measurably reduce the iteration count
        // needed against no preconditioner; block Gauss-Seidel's exact
        // per-block (rather than elementwise) diagonal solve is never
        // worse than plain Jacobi.
        assert_eq!((none, jacobi, gs), (4, 2, 2));
    }

    #[test]
    fn block_gauss_seidel_measurably_reduces_gmres_iterations_over_three_blocks() {
        let (rows, layout) = coupled_dense_system(3, 3);
        let none = gmres_iterations(rows.clone(), None, layout.dimension());
        let jacobi_preconditioner = jacobi_preconditioner(&layout, &rows);
        let jacobi = gmres_iterations(
            rows.clone(),
            Some(&jacobi_preconditioner),
            layout.dimension(),
        );
        let diagonal_solves = dense_block_solves(&rows, &layout);
        let diagonal_refs: Vec<&dyn Preconditioner> = diagonal_solves
            .iter()
            .map(|solve| solve as &dyn Preconditioner)
            .collect();
        let couplings = dense_couplings(&rows, &layout);
        let coupling_refs: Vec<&dyn BlockCouplingAction> = couplings
            .iter()
            .map(|coupling| coupling as &dyn BlockCouplingAction)
            .collect();
        let gauss_seidel = BlockGaussSeidelPreconditioner::new(
            layout.clone(),
            diagonal_refs,
            coupling_refs,
            GaussSeidelSweep::Forward,
        )
        .unwrap();
        let gs = gmres_iterations(rows, Some(&gauss_seidel), layout.dimension());
        // Here the exact per-block solve also measurably beats elementwise
        // Jacobi, not only no preconditioner.
        assert_eq!((none, jacobi, gs), (5, 3, 2));
    }
}
