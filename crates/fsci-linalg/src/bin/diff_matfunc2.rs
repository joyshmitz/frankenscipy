//! Probe: logm/fracpow/signm on a complex-eigenvalue matrix vs scipy (gitignored).
use fsci_linalg::{DecompOptions, expm, fractional_matrix_power, logm, signm};

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
    // m2: nonsymmetric, complex eigenvalues (same as diff_matfuncs)
    let m2 = vec![
        vec![0.5, -1.2, 0.3],
        vec![1.1, 0.4, -0.2],
        vec![0.0, 0.6, 0.9],
    ];
    // m4: positive-definite-ish nonsymmetric with complex eigenvalues but positive real parts
    let m4 = vec![
        vec![2.0, -1.0, 0.0],
        vec![1.0, 2.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    for (lbl, a) in [("m2", &m2), ("m4", &m4)] {
        run(&format!("logm_{lbl}"), logm(a, o()));
        run(
            &format!("fracpow0.5_{lbl}"),
            fractional_matrix_power(a, 0.5, o()),
        );
        run(&format!("signm_{lbl}"), signm(a, o()));
        run(&format!("expm_{lbl}"), expm(a, o()));
    }
}
