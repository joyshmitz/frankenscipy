use fsci_linalg::{DecompOptions, signm};
fn run(l: &str, r: Result<Vec<Vec<f64>>, fsci_linalg::LinalgError>) {
    match r {
        Ok(m) => {
            for (i, row) in m.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    println!("{l},{i},{j},{v:.17e}");
                }
            }
        }
        Err(e) => println!("{l},ERR,{e:?}"),
    }
}
fn main() {
    let o = || DecompOptions::default();
    // s1: eigenvalues 1±3i (right half-plane) -> signm should be I
    let s1 = vec![vec![3.0, -5.0], vec![2.0, -1.0]];
    // s2: eigenvalues -1±2i (left half-plane) -> signm should be -I
    let s2 = vec![vec![-1.0, -2.0], vec![2.0, -1.0]];
    // s3: 3x3 mixed: block with Re>0 + real negative eigenvalue
    let s3 = vec![
        vec![0.5, -1.2, 0.3],
        vec![1.1, 0.4, -0.2],
        vec![0.0, 0.0, -2.0],
    ];
    run("signm_s1", signm(&s1, o()));
    run("signm_s2", signm(&s2, o()));
    run("signm_s3", signm(&s3, o()));
}
