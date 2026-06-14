//! Differential oracle probe: matrix-equation solvers + polar vs scipy.linalg (gitignored).
//! Lines: `name,r,c,value`. Inputs match the python comparator.
use fsci_linalg::{
    DecompOptions, polar, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are,
    solve_discrete_lyapunov, solve_sylvester,
};

fn dump(name: &str, m: &[Vec<f64>]) {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            println!("{name},{r},{c},{v:.17e}");
        }
    }
}
fn run(name: &str, r: Result<Vec<Vec<f64>>, fsci_linalg::LinalgError>) {
    match r {
        Ok(m) => dump(name, &m),
        Err(e) => println!("{name},ERR,{e:?}"),
    }
}

fn main() {
    let o = || DecompOptions::default();

    // Sylvester: A X + X B = Q
    let a = vec![vec![1.0, 2.0], vec![-1.0, 3.0]];
    let b = vec![vec![0.5, -1.0], vec![2.0, 1.5]];
    let qsyl = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    run("sylvester", solve_sylvester(&a, &b, &qsyl, o()));

    // Continuous Lyapunov: A X + X A^T = Q  (Q symmetric)
    let al = vec![vec![-3.0, 1.0], vec![1.0, -2.0]];
    let ql = vec![vec![1.0, 0.5], vec![0.5, 2.0]];
    run("clyap", solve_continuous_lyapunov(&al, &ql, o()));

    // Discrete Lyapunov: A X A^T - X + Q = 0
    let ad = vec![vec![0.5, 0.1], vec![0.2, 0.3]];
    run("dlyap", solve_discrete_lyapunov(&ad, &ql, o()));

    // Continuous ARE: A^T X + X A - X B R^-1 B^T X + Q = 0
    let aare = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
    let bare = vec![vec![0.0], vec![1.0]];
    let qare = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let rare = vec![vec![1.0]];
    run(
        "care",
        solve_continuous_are(&aare, &bare, &qare, &rare, o()),
    );

    // Discrete ARE
    let adare = vec![vec![0.9, 0.1], vec![0.0, 0.8]];
    run("dare", solve_discrete_are(&adare, &bare, &qare, &rare, o()));

    // Polar decomposition A = U P
    let ap = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    match polar(&ap, o()) {
        Ok(pr) => {
            dump("polar_u", &pr.u);
            dump("polar_p", &pr.p);
        }
        Err(e) => println!("polar,ERR,{e:?}"),
    }
}
