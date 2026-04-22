#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::Arbitrary;
use fsci_sparse::{BsrMatrix, DokMatrix, FormatConvertible, LilMatrix, Shape2D, SparseError};
use libfuzzer_sys::fuzz_target;

const MAX_DIM: usize = 64;
const MAX_ENTRIES: usize = 1024;
const MAX_OPS: usize = 96;
const MAX_BLOCK_DIM: usize = 8;
const MAX_MUTATION_INDEX: usize = MAX_DIM + 16;

#[derive(Clone, Copy, Debug, Arbitrary)]
struct Entry {
    row: u8,
    col: u8,
    value: i8,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum Mutation {
    Insert { row: u8, col: u8, value: i8 },
    Remove { row: u8, col: u8 },
}

#[derive(Clone, Debug, Arbitrary)]
struct SparseFormatsInput {
    rows: u8,
    cols: u8,
    block_rows: u8,
    block_cols: u8,
    entries: Vec<Entry>,
    mutations: Vec<Mutation>,
}

#[derive(Clone, Copy, Debug)]
enum NormalizedMutation {
    Insert { row: usize, col: usize, value: f64 },
    Remove { row: usize, col: usize },
}

fn clamp_dimension(raw: u8) -> usize {
    usize::from(raw) % (MAX_DIM + 1)
}

fn clamp_block_dimension(raw: u8) -> usize {
    usize::from(raw) % (MAX_BLOCK_DIM + 1)
}

fn clamp_mutation_index(raw: u8) -> usize {
    usize::from(raw) % MAX_MUTATION_INDEX
}

fn normalize_value(raw: i8) -> f64 {
    f64::from(i16::from(raw).rem_euclid(9) - 4)
}

fn normalized_triplets(input: &SparseFormatsInput) -> (Shape2D, Vec<(usize, usize, f64)>) {
    let rows = clamp_dimension(input.rows);
    let cols = clamp_dimension(input.cols);
    let shape = Shape2D::new(rows, cols);
    if rows == 0 || cols == 0 {
        return (shape, Vec::new());
    }

    let triplets = input
        .entries
        .iter()
        .take(MAX_ENTRIES)
        .map(|entry| {
            (
                usize::from(entry.row) % rows,
                usize::from(entry.col) % cols,
                normalize_value(entry.value),
            )
        })
        .collect();
    (shape, triplets)
}

fn normalized_mutations(input: &SparseFormatsInput) -> Vec<NormalizedMutation> {
    input
        .mutations
        .iter()
        .take(MAX_OPS)
        .map(|mutation| match mutation {
            Mutation::Insert { row, col, value } => NormalizedMutation::Insert {
                row: clamp_mutation_index(*row),
                col: clamp_mutation_index(*col),
                value: normalize_value(*value),
            },
            Mutation::Remove { row, col } => NormalizedMutation::Remove {
                row: clamp_mutation_index(*row),
                col: clamp_mutation_index(*col),
            },
        })
        .collect()
}

fn triplet_vectors(triplets: &[(usize, usize, f64)]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut data = Vec::with_capacity(triplets.len());
    let mut row_indices = Vec::with_capacity(triplets.len());
    let mut col_indices = Vec::with_capacity(triplets.len());

    for &(row, col, value) in triplets {
        data.push(value);
        row_indices.push(row);
        col_indices.push(col);
    }

    (data, row_indices, col_indices)
}

fn dense_from_triplets(shape: Shape2D, triplets: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for &(row, col, value) in triplets {
        dense[row][col] += value;
    }
    dense
}

fn dense_from_map(shape: Shape2D, entries: &BTreeMap<(usize, usize), f64>) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for (&(row, col), &value) in entries {
        dense[row][col] = value;
    }
    dense
}

fn dense_from_coo(matrix: &fsci_sparse::CooMatrix) -> Vec<Vec<f64>> {
    let shape = matrix.shape();
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..matrix.nnz() {
        dense[matrix.row_indices()[idx]][matrix.col_indices()[idx]] += matrix.data()[idx];
    }
    dense
}

fn initial_lil_entries(triplets: &[(usize, usize, f64)]) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for &(row, col, value) in triplets {
        let key = (row, col);
        *entries.entry(key).or_insert(0.0) += value;
    }
    entries
}

fn initial_dok_entries(triplets: &[(usize, usize, f64)]) -> BTreeMap<(usize, usize), f64> {
    let mut entries = BTreeMap::new();
    for &(row, col, value) in triplets {
        let key = (row, col);
        let updated = entries.get(&key).copied().unwrap_or(0.0) + value;
        if updated == 0.0 {
            entries.remove(&key);
        } else {
            entries.insert(key, updated);
        }
    }
    entries
}

fn assert_dense_eq(actual: &[Vec<f64>], expected: &[Vec<f64>], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: row count mismatch"
    );
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_row.len(),
            expected_row.len(),
            "{context}: col count mismatch on row {row_idx}"
        );
        for (col_idx, (actual_value, expected_value)) in
            actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            assert_eq!(
                actual_value, expected_value,
                "{context}: mismatch at ({row_idx}, {col_idx})"
            );
        }
    }
}

fn assert_index_out_of_bounds<T>(
    result: Result<T, SparseError>,
    shape: Shape2D,
    row: usize,
    col: usize,
    context: &str,
) {
    let (expected_axis, expected_index, expected_bound) = if row >= shape.rows {
        ("row", row, shape.rows)
    } else {
        ("col", col, shape.cols)
    };
    assert!(
        matches!(
            result,
            Err(SparseError::IndexOutOfBounds {
                axis,
                index,
                bound,
            }) if axis == expected_axis && index == expected_index && bound == expected_bound
        ),
        "{context}: expected out-of-bounds error"
    );
}

fn expected_bsr_blocks(
    triplets: &[(usize, usize, f64)],
    block_shape: Shape2D,
) -> BTreeSet<(usize, usize)> {
    let mut blocks = BTreeSet::new();
    for &(row, col, _) in triplets {
        blocks.insert((row / block_shape.rows, col / block_shape.cols));
    }
    blocks
}

fn assert_lil_invariants(
    lil: &LilMatrix,
    shape: Shape2D,
    entries: &BTreeMap<(usize, usize), f64>,
    context: &str,
) {
    assert_eq!(lil.nnz(), entries.len(), "{context}: nnz mismatch");
    for row_indices in lil.row_indices() {
        for pair in row_indices.windows(2) {
            assert!(pair[0] < pair[1], "{context}: row indices must stay sorted");
        }
    }

    let expected_dense = dense_from_map(shape, entries);
    for row in 0..shape.rows {
        for col in 0..shape.cols {
            let expected_value = entries.get(&(row, col)).copied().unwrap_or(0.0);
            let actual_value = lil.get(row, col);
            assert!(actual_value.is_ok(), "{context}: get should succeed");
            let Ok(actual_value) = actual_value else {
                return;
            };
            assert_eq!(actual_value, expected_value, "{context}: get mismatch");

            let contains_result = lil.contains(row, col);
            assert!(
                contains_result.is_ok(),
                "{context}: contains should succeed"
            );
            let Ok(actual_contains) = contains_result else {
                return;
            };
            assert_eq!(
                actual_contains,
                entries.contains_key(&(row, col)),
                "{context}: contains mismatch"
            );
        }
    }

    let coo_result = lil.to_coo();
    assert!(coo_result.is_ok(), "{context}: lil->coo should succeed");
    let Ok(coo) = coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&coo), &expected_dense, context);

    let csr_result = lil.to_csr();
    assert!(csr_result.is_ok(), "{context}: lil->csr should succeed");
    let Ok(csr) = csr_result else {
        return;
    };
    let csr_coo_result = csr.to_coo();
    assert!(csr_coo_result.is_ok(), "{context}: csr->coo should succeed");
    let Ok(csr_coo) = csr_coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&csr_coo), &expected_dense, context);

    let csc_result = lil.to_csc();
    assert!(csc_result.is_ok(), "{context}: lil->csc should succeed");
    let Ok(csc) = csc_result else {
        return;
    };
    let csc_coo_result = csc.to_coo();
    assert!(csc_coo_result.is_ok(), "{context}: csc->coo should succeed");
    let Ok(csc_coo) = csc_coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&csc_coo), &expected_dense, context);
}

fn assert_dok_invariants(
    dok: &DokMatrix,
    shape: Shape2D,
    entries: &BTreeMap<(usize, usize), f64>,
    context: &str,
) {
    assert_eq!(dok.nnz(), entries.len(), "{context}: nnz mismatch");

    let expected_dense = dense_from_map(shape, entries);
    for row in 0..shape.rows {
        for col in 0..shape.cols {
            let expected_value = entries.get(&(row, col)).copied().unwrap_or(0.0);
            let actual_value = dok.get(row, col);
            assert!(actual_value.is_ok(), "{context}: get should succeed");
            let Ok(actual_value) = actual_value else {
                return;
            };
            assert_eq!(actual_value, expected_value, "{context}: get mismatch");

            let contains_result = dok.contains(row, col);
            assert!(
                contains_result.is_ok(),
                "{context}: contains should succeed"
            );
            let Ok(actual_contains) = contains_result else {
                return;
            };
            assert_eq!(
                actual_contains,
                entries.contains_key(&(row, col)),
                "{context}: contains mismatch"
            );
        }
    }

    let coo_result = dok.to_coo();
    assert!(coo_result.is_ok(), "{context}: dok->coo should succeed");
    let Ok(coo) = coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&coo), &expected_dense, context);

    let csr_result = dok.to_csr();
    assert!(csr_result.is_ok(), "{context}: dok->csr should succeed");
    let Ok(csr) = csr_result else {
        return;
    };
    let csr_coo_result = csr.to_coo();
    assert!(csr_coo_result.is_ok(), "{context}: csr->coo should succeed");
    let Ok(csr_coo) = csr_coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&csr_coo), &expected_dense, context);

    let csc_result = dok.to_csc();
    assert!(csc_result.is_ok(), "{context}: dok->csc should succeed");
    let Ok(csc) = csc_result else {
        return;
    };
    let csc_coo_result = csc.to_coo();
    assert!(csc_coo_result.is_ok(), "{context}: csc->coo should succeed");
    let Ok(csc_coo) = csc_coo_result else {
        return;
    };
    assert_dense_eq(&dense_from_coo(&csc_coo), &expected_dense, context);
}

fuzz_target!(|input: SparseFormatsInput| {
    let (shape, triplets) = normalized_triplets(&input);
    let (data, row_indices, col_indices) = triplet_vectors(&triplets);
    let mutations = normalized_mutations(&input);
    let expected_dense = dense_from_triplets(shape, &triplets);

    let lil_result = LilMatrix::from_triplets(
        shape,
        data.clone(),
        row_indices.clone(),
        col_indices.clone(),
    );
    assert!(
        lil_result.is_ok(),
        "LIL constructor should succeed for normalized triplets"
    );
    let Ok(mut lil) = lil_result else {
        return;
    };
    let mut expected_lil_entries = initial_lil_entries(&triplets);
    assert_lil_invariants(&lil, shape, &expected_lil_entries, "lil/from_triplets");

    let dok_result = DokMatrix::from_triplets(
        shape,
        data.clone(),
        row_indices.clone(),
        col_indices.clone(),
    );
    assert!(
        dok_result.is_ok(),
        "DOK constructor should succeed for normalized triplets"
    );
    let Ok(mut dok) = dok_result else {
        return;
    };
    let mut expected_dok_entries = initial_dok_entries(&triplets);
    assert_dok_invariants(&dok, shape, &expected_dok_entries, "dok/from_triplets");
    assert_dense_eq(
        &dense_from_map(shape, &expected_dok_entries),
        &expected_dense,
        "dok initial dense semantics",
    );

    let block_shape = Shape2D::new(
        clamp_block_dimension(input.block_rows),
        clamp_block_dimension(input.block_cols),
    );
    let bsr_result = BsrMatrix::from_triplets(
        shape,
        block_shape,
        data.clone(),
        row_indices.clone(),
        col_indices.clone(),
    );
    let valid_block_shape = block_shape.rows != 0
        && block_shape.cols != 0
        && shape.rows % block_shape.rows == 0
        && shape.cols % block_shape.cols == 0;
    if valid_block_shape {
        assert!(
            bsr_result.is_ok(),
            "BSR constructor should succeed for valid block shapes"
        );
        let Ok(bsr) = bsr_result else {
            return;
        };
        assert!(
            bsr.canonical_meta().sorted_indices,
            "BSR indices should be sorted"
        );
        assert!(
            bsr.canonical_meta().deduplicated,
            "BSR blocks should be deduplicated"
        );
        assert_eq!(
            bsr.nnz_blocks(),
            expected_bsr_blocks(&triplets, block_shape).len(),
            "BSR occupied block count mismatch"
        );

        let bsr_coo_result = bsr.to_coo();
        assert!(bsr_coo_result.is_ok(), "BSR should convert to COO");
        let Ok(bsr_coo) = bsr_coo_result else {
            return;
        };
        assert_dense_eq(
            &dense_from_coo(&bsr_coo),
            &expected_dense,
            "bsr/to_coo dense semantics",
        );

        let csr_result = bsr.to_csr();
        assert!(csr_result.is_ok(), "BSR should convert to CSR");
        let Ok(csr) = csr_result else {
            return;
        };
        let csr_coo_result = csr.to_coo();
        assert!(
            csr_coo_result.is_ok(),
            "BSR CSR roundtrip should convert to COO"
        );
        let Ok(csr_coo) = csr_coo_result else {
            return;
        };
        assert_dense_eq(
            &dense_from_coo(&csr_coo),
            &expected_dense,
            "bsr/to_csr dense semantics",
        );

        let csc_result = bsr.to_csc();
        assert!(csc_result.is_ok(), "BSR should convert to CSC");
        let Ok(csc) = csc_result else {
            return;
        };
        let csc_coo_result = csc.to_coo();
        assert!(
            csc_coo_result.is_ok(),
            "BSR CSC roundtrip should convert to COO"
        );
        let Ok(csc_coo) = csc_coo_result else {
            return;
        };
        assert_dense_eq(
            &dense_from_coo(&csc_coo),
            &expected_dense,
            "bsr/to_csc dense semantics",
        );

        for (row, dense_row) in expected_dense.iter().enumerate() {
            for (col, expected_value) in dense_row.iter().enumerate() {
                let get_result = bsr.get(row, col);
                assert!(get_result.is_ok(), "BSR get should succeed");
                let Ok(actual_value) = get_result else {
                    return;
                };
                assert_eq!(
                    actual_value, *expected_value,
                    "BSR get mismatch at ({row}, {col})"
                );
            }
        }
    } else {
        assert!(
            matches!(bsr_result, Err(SparseError::InvalidShape { .. })),
            "BSR should reject invalid block shapes"
        );
    }

    for mutation in mutations {
        match mutation {
            NormalizedMutation::Insert { row, col, value } => {
                if row >= shape.rows || col >= shape.cols {
                    assert_index_out_of_bounds(
                        lil.insert(row, col, value),
                        shape,
                        row,
                        col,
                        "lil insert",
                    );
                    assert_index_out_of_bounds(
                        dok.insert(row, col, value),
                        shape,
                        row,
                        col,
                        "dok insert",
                    );
                    continue;
                }

                let lil_expected_previous = expected_lil_entries.insert((row, col), value);
                let lil_insert_result = lil.insert(row, col, value);
                assert!(lil_insert_result.is_ok(), "LIL insert should succeed");
                let Ok(lil_previous) = lil_insert_result else {
                    return;
                };
                assert_eq!(
                    lil_previous, lil_expected_previous,
                    "LIL insert previous value mismatch"
                );

                let dok_expected_previous = expected_dok_entries.get(&(row, col)).copied();
                if value == 0.0 {
                    expected_dok_entries.remove(&(row, col));
                } else {
                    expected_dok_entries.insert((row, col), value);
                }
                let dok_insert_result = dok.insert(row, col, value);
                assert!(dok_insert_result.is_ok(), "DOK insert should succeed");
                let Ok(dok_previous) = dok_insert_result else {
                    return;
                };
                assert_eq!(
                    dok_previous, dok_expected_previous,
                    "DOK insert previous value mismatch"
                );
            }
            NormalizedMutation::Remove { row, col } => {
                if row >= shape.rows || col >= shape.cols {
                    assert_index_out_of_bounds(lil.remove(row, col), shape, row, col, "lil remove");
                    assert_index_out_of_bounds(dok.remove(row, col), shape, row, col, "dok remove");
                    continue;
                }

                let lil_expected_previous = expected_lil_entries.remove(&(row, col));
                let lil_remove_result = lil.remove(row, col);
                assert!(lil_remove_result.is_ok(), "LIL remove should succeed");
                let Ok(lil_previous) = lil_remove_result else {
                    return;
                };
                assert_eq!(
                    lil_previous, lil_expected_previous,
                    "LIL remove previous value mismatch"
                );

                let dok_expected_previous = expected_dok_entries.remove(&(row, col));
                let dok_remove_result = dok.remove(row, col);
                assert!(dok_remove_result.is_ok(), "DOK remove should succeed");
                let Ok(dok_previous) = dok_remove_result else {
                    return;
                };
                assert_eq!(
                    dok_previous, dok_expected_previous,
                    "DOK remove previous value mismatch"
                );
            }
        }
    }

    assert_lil_invariants(&lil, shape, &expected_lil_entries, "lil/post-mutations");
    assert_dok_invariants(&dok, shape, &expected_dok_entries, "dok/post-mutations");
});
