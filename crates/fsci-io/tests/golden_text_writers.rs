use fsci_io::{MatArray, mmwrite, savemat_text, write_csv, write_json_array};

#[test]
fn io_text_writers_match_committed_goldens() {
    let csv = write_csv(
        Some(&["time", "value"]),
        &[vec![0.0, 1.5], vec![1.0, 2.25], vec![2.0, 3.0]],
        ',',
    )
    .expect("CSV should serialize");
    assert_eq!(csv, include_str!("goldens/write_csv_signal.csv"));

    let json = write_json_array(&[1.5, 2.25, 3.0]).expect("JSON array should serialize");
    assert_eq!(
        json,
        include_str!("goldens/write_json_array.txt").trim_end_matches('\n')
    );

    let mm =
        mmwrite(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Matrix Market should serialize");
    assert_eq!(mm, include_str!("goldens/mmwrite_dense.mtx"));

    let mat = savemat_text(&[MatArray {
        name: "signal".to_owned(),
        rows: 2,
        cols: 2,
        data: vec![1.0, 2.0, 3.5, 4.0],
    }])
    .expect("MAT text should serialize");
    assert_eq!(mat, include_str!("goldens/savemat_text_signal.mat.txt"));
}
