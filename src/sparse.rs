use serde::{Deserialize, Serialize};

use crate::{EvaluationContext, LinearOperator, NumericError};

/// Canonical compressed-sparse-row matrix with sorted, unique columns per row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CsrMatrixData")]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct CsrMatrixData {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

impl TryFrom<CsrMatrixData> for CsrMatrix {
    type Error = NumericError;

    fn try_from(data: CsrMatrixData) -> Result<Self, Self::Error> {
        Self::new(
            data.rows,
            data.columns,
            data.row_offsets,
            data.column_indices,
            data.values,
        )
    }
}

impl CsrMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, NumericError> {
        let expected_offsets = rows
            .checked_add(1)
            .ok_or_else(|| NumericError::InvalidInput {
                message: "CSR row count cannot be represented with a terminal offset".into(),
            })?;
        if row_offsets.len() != expected_offsets {
            return Err(NumericError::DimensionMismatch {
                operation: "CSR row offsets".into(),
                expected: expected_offsets,
                actual: row_offsets.len(),
            });
        }
        if column_indices.len() != values.len() {
            return Err(NumericError::DimensionMismatch {
                operation: "CSR column/value arrays".into(),
                expected: column_indices.len(),
                actual: values.len(),
            });
        }
        if row_offsets.first() != Some(&0) || row_offsets.last() != Some(&values.len()) {
            return Err(NumericError::InvalidInput {
                message: "CSR offsets must start at zero and end at nnz".into(),
            });
        }
        NumericError::require_finite("CSR values", &values)?;
        for row in 0..rows {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            if start > end || end > values.len() {
                return Err(NumericError::InvalidInput {
                    message: format!("CSR offsets are invalid at row {row}"),
                });
            }
            let row_columns = &column_indices[start..end];
            if row_columns.iter().any(|column| *column >= columns) {
                return Err(NumericError::InvalidInput {
                    message: format!("CSR column is out of bounds at row {row}"),
                });
            }
            if row_columns.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(NumericError::InvalidInput {
                    message: format!("CSR columns are not strictly sorted at row {row}"),
                });
            }
        }
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
        })
    }

    /// Sorts triplets, sums duplicate coordinates, and emits canonical CSR.
    pub fn from_triplets(
        rows: usize,
        columns: usize,
        mut entries: Vec<(usize, usize, f64)>,
    ) -> Result<Self, NumericError> {
        let offset_count = rows
            .checked_add(1)
            .ok_or_else(|| NumericError::InvalidInput {
                message: "CSR row count cannot be represented with a terminal offset".into(),
            })?;
        for (index, (row, column, value)) in entries.iter().copied().enumerate() {
            if row >= rows || column >= columns {
                return Err(NumericError::InvalidInput {
                    message: format!("triplet {index} is out of bounds"),
                });
            }
            if !value.is_finite() {
                return Err(NumericError::NonFinite {
                    operation: "sparse triplet values".into(),
                    index,
                });
            }
        }
        entries.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.total_cmp(&right.2))
        });

        let mut merged: Vec<(usize, usize, f64)> = Vec::with_capacity(entries.len());
        for (row, column, value) in entries {
            match merged.last_mut() {
                Some((last_row, last_column, last_value))
                    if *last_row == row && *last_column == column =>
                {
                    let sum = *last_value + value;
                    if !sum.is_finite() {
                        return Err(NumericError::InvalidInput {
                            message: "summing duplicate sparse entries overflowed".into(),
                        });
                    }
                    *last_value = sum;
                }
                _ => merged.push((row, column, value)),
            }
        }
        merged.retain(|(_, _, value)| *value != 0.0);

        let mut row_offsets = vec![0usize; offset_count];
        for (row, _, _) in &merged {
            let next_row = row
                .checked_add(1)
                .ok_or_else(|| NumericError::InvalidInput {
                    message: "CSR row offset index overflowed".into(),
                })?;
            row_offsets[next_row] =
                row_offsets[next_row]
                    .checked_add(1)
                    .ok_or_else(|| NumericError::InvalidInput {
                        message: "CSR row entry count overflowed".into(),
                    })?;
        }
        for row in 0..rows {
            let next_row = row + 1;
            row_offsets[next_row] = row_offsets[next_row]
                .checked_add(row_offsets[row])
                .ok_or_else(|| NumericError::InvalidInput {
                    message: "CSR cumulative row offset overflowed".into(),
                })?;
        }
        let column_indices = merged.iter().map(|(_, column, _)| *column).collect();
        let values = merged.into_iter().map(|(_, _, value)| value).collect();
        Self::new(rows, columns, row_offsets, column_indices, values)
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl LinearOperator for CsrMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply(
        &self,
        _context: &EvaluationContext,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        NumericError::require_len("CSR input", input.len(), self.columns)?;
        NumericError::require_len("CSR output", output.len(), self.rows)?;
        NumericError::require_finite("CSR input", input)?;
        output.fill(0.0);
        for (row, output_value) in output.iter_mut().enumerate() {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                *output_value += self.values[entry] * input[self.column_indices[entry]];
            }
        }
        NumericError::require_finite("CSR output", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_are_sorted_and_duplicates_are_summed() {
        let matrix =
            CsrMatrix::from_triplets(2, 2, vec![(1, 0, 2.0), (0, 1, 3.0), (1, 0, 4.0)]).unwrap();
        assert_eq!(matrix.row_offsets(), &[0, 1, 2]);
        assert_eq!(matrix.column_indices(), &[1, 0]);
        assert_eq!(matrix.values(), &[3.0, 6.0]);
    }

    #[test]
    fn duplicate_summation_is_independent_of_input_order() {
        let first =
            CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0e16), (0, 0, -1.0e16), (0, 0, 1.0)])
                .unwrap();
        let second =
            CsrMatrix::from_triplets(1, 1, vec![(0, 0, 1.0), (0, 0, 1.0e16), (0, 0, -1.0e16)])
                .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn deserialization_revalidates_csr_invariants() {
        let malformed = r#"{
            "rows": 1,
            "columns": 1,
            "row_offsets": [0, 2],
            "column_indices": [0, 0],
            "values": [1.0, 2.0]
        }"#;
        assert!(serde_json::from_str::<CsrMatrix>(malformed).is_err());
    }

    #[test]
    fn matrix_applies_in_place() {
        let matrix =
            CsrMatrix::from_triplets(2, 2, vec![(0, 0, 2.0), (0, 1, -1.0), (1, 1, 4.0)]).unwrap();
        let mut output = vec![0.0; 2];
        matrix
            .apply(&EvaluationContext::reproducible(), &[3.0, 5.0], &mut output)
            .unwrap();
        assert_eq!(output, vec![1.0, 20.0]);
    }
}
