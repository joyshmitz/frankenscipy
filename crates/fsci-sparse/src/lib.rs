pub mod construct;
pub mod formats;
pub mod linalg;
pub mod ops;

pub use construct::{diags, eye, random};
pub use formats::{
    CanonicalMeta, ConstructionLogEntry, CooMatrix, CscMatrix, CsrMatrix, NalgebraBridge, Shape2D,
    SparseError, SparseFormat, SparseResult,
};
pub use linalg::{
    IluOptions, LuOptions, PermutationOrdering, SolveOptions, SolveResult, SparseBackend,
    SparseIluFactorization, SparseLuFactorization, spilu, splu, spsolve,
};
pub use ops::{
    ConversionLogEntry, FormatConvertible, add_coo, add_csc, add_csr, coo_to_csr_with_mode,
    csc_to_csr_with_mode, csr_to_csc_with_mode, scale_coo, scale_csc, scale_csr, spmv_coo,
    spmv_csc, spmv_csr, sub_coo, sub_csc, sub_csr,
};

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use super::*;

    #[test]
    fn rejects_invalid_csr_pointer_shape() {
        let shape = Shape2D::new(2, 2);
        let err = CsrMatrix::from_components(shape, vec![1.0, 2.0], vec![0, 1], vec![0, 1], false)
            .expect_err("indptr length must be rows + 1");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn converts_coo_to_csr_and_preserves_dense_semantics() {
        let shape = Shape2D::new(2, 2);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, 2.0, 3.0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr conversion");
        assert_eq!(csr.shape(), shape);
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn roundtrip_conversion_identity_all_formats() {
        let shape = Shape2D::new(3, 3);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, 2.0, -3.0, 4.0],
            vec![0, 0, 1, 2],
            vec![0, 2, 1, 2],
            false,
        )
        .expect("valid coo");

        let csr = coo.to_csr().expect("coo->csr");
        let csc = csr.to_csc().expect("csr->csc");
        let back_to_coo = csc.to_coo().expect("csc->coo");

        assert_eq!(dense_from_coo(&coo), dense_from_coo(&back_to_coo));
    }

    #[test]
    fn spmv_matches_dense_for_all_formats() {
        let shape = Shape2D::new(3, 3);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, -2.0, 3.0, 4.0],
            vec![0, 1, 1, 2],
            vec![0, 0, 2, 1],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr");
        let csc = coo.to_csc().expect("coo->csc");
        let vector = vec![2.0, -1.0, 3.0];

        let expected = dense_matvec(&dense_from_coo(&coo), &vector);
        assert_eq!(spmv_coo(&coo, &vector).expect("coo spmv"), expected);
        assert_eq!(spmv_csr(&csr, &vector).expect("csr spmv"), expected);
        assert_eq!(spmv_csc(&csc, &vector).expect("csc spmv"), expected);
    }

    #[test]
    fn strict_mode_preserves_metadata_and_emits_structured_logs() {
        let shape = Shape2D::new(2, 3);
        let csr = CsrMatrix::from_components(
            shape,
            vec![10.0, 1.0, 2.0],
            vec![2, 0, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("valid non-canonical csr");
        assert!(!csr.canonical_meta().sorted_indices);

        let (csc, conv1) =
            csr_to_csc_with_mode(&csr, RuntimeMode::Strict, "op-conv-1").expect("strict csr->csc");
        let (roundtrip, conv2) =
            csc_to_csr_with_mode(&csc, RuntimeMode::Strict, "op-conv-2").expect("strict csc->csr");
        assert_eq!(roundtrip.canonical_meta(), csr.canonical_meta());

        let construction_log = roundtrip.construction_log(
            RuntimeMode::Strict,
            "op-construct-1",
            "passed_invariant_validation",
        );
        let json = construction_log.to_json();
        assert!(json.contains("\"operation_id\":\"op-construct-1\""));
        assert!(json.contains("\"validation_result\":\"passed_invariant_validation\""));

        let conv1_json = conv1.to_json();
        let conv2_json = conv2.to_json();
        assert!(conv1_json.contains("\"from_format\":\"csr\""));
        assert!(conv1_json.contains("\"to_format\":\"csc\""));
        assert!(conv2_json.contains("\"from_format\":\"csc\""));
        assert!(conv2_json.contains("\"to_format\":\"csr\""));
    }

    #[test]
    fn zero_and_single_element_matrices_are_supported() {
        let zero = eye(0).expect("zero eye");
        assert_eq!(zero.shape(), Shape2D::new(0, 0));
        assert_eq!(zero.nnz(), 0);
        assert!(spmv_csr(&zero, &[]).expect("zero spmv").is_empty());

        let single =
            CooMatrix::from_triplets(Shape2D::new(1, 1), vec![5.0], vec![0], vec![0], false)
                .expect("single coo");
        let single_csr = single.to_csr().expect("single csr");
        let single_csc = single.to_csc().expect("single csc");
        assert_eq!(spmv_coo(&single, &[2.0]).expect("coo single"), vec![10.0]);
        assert_eq!(
            spmv_csr(&single_csr, &[2.0]).expect("csr single"),
            vec![10.0]
        );
        assert_eq!(
            spmv_csc(&single_csc, &[2.0]).expect("csc single"),
            vec![10.0]
        );
    }

    fn dense_from_coo(coo: &CooMatrix) -> Vec<Vec<f64>> {
        let shape = coo.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        for idx in 0..coo.nnz() {
            dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
        }
        dense
    }

    fn dense_matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
            .collect()
    }
}
