//! Probe: sqrtm on complex-eigenvalue matrices vs scipy.linalg.sqrtm (gitignored).
use fsci_linalg::{DecompOptions, sqrtm};

fn dump(func: &str, m: &[Vec<f64>]) {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            println!("{func},{r},{c},{v:.17e}");
        }
    }
}
fn run(func: &str, r: Result<Vec<Vec<f64>>, fsci_linalg::LinalgError>) {
    match r {
        Ok(m) => dump(func, &m),
        Err(e) => println!("{func},ERR,{e:?}"),
    }
}
fn main() {
    let o = || DecompOptions::default();
    // m2: complex eigenvalues
    let m2 = vec![
        vec![0.5, -1.2, 0.3],
        vec![1.1, 0.4, -0.2],
        vec![0.0, 0.6, 0.9],
    ];
    // m4: 2x2 rotation-ish block + real eigenvalue
    let m4 = vec![
        vec![2.0, -1.0, 0.0],
        vec![1.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    // m5: all-real spectrum, nonsymmetric (regression guard for the real path)
    let m5 = vec![
        vec![4.0, 1.0, 2.0],
        vec![0.0, 3.0, 0.5],
        vec![0.0, 0.0, 2.0],
    ];
    for (lbl, a) in [("m2", &m2), ("m4", &m4), ("m5", &m5)] {
        run(&format!("sqrtm_{lbl}"), sqrtm(a, o()));
    }
}
