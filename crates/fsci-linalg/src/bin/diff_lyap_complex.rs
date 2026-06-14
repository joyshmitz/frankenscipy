//! Probe: Lyapunov/Sylvester/ARE with COMPLEX-eigenvalue A vs scipy.linalg (gitignored).
use fsci_linalg::{
    DecompOptions, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are,
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
    // Continuous Lyapunov: A stable with COMPLEX eigenvalues (-0.5 ± i)
    let ac = vec![vec![-0.5, -1.0], vec![1.0, -0.5]];
    let q = vec![vec![1.0, 0.5], vec![0.5, 2.0]];
    run("clyap_cplx", solve_continuous_lyapunov(&ac, &q, o()));

    // Discrete Lyapunov: |eig|<1 with COMPLEX eigenvalues (0.3 ± 0.5i, |λ|≈0.58)
    let ad = vec![vec![0.3, -0.5], vec![0.5, 0.3]];
    run("dlyap_cplx", solve_discrete_lyapunov(&ad, &q, o()));

    // Sylvester with complex eigenvalues on both A and B (already clean — regression guard)
    let a = vec![vec![1.0, 2.0], vec![-1.0, 3.0]];
    let b = vec![vec![0.5, -1.0], vec![2.0, 1.5]];
    let qs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    run("sylv_cplx", solve_sylvester(&a, &b, &qs, o()));

    // Continuous ARE with complex-eigenvalue A
    let aare = vec![vec![0.0, 1.0], vec![-2.0, -0.3]]; // eigenvalues -0.15 ± 1.41i
    let bare = vec![vec![0.0], vec![1.0]];
    let qare = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let rare = vec![vec![1.0]];
    run(
        "care_cplx",
        solve_continuous_are(&aare, &bare, &qare, &rare, o()),
    );

    // Discrete ARE with complex-eigenvalue A
    let adare = vec![vec![0.3, -0.5], vec![0.5, 0.3]];
    run(
        "dare_cplx",
        solve_discrete_are(&adare, &bare, &qare, &rare, o()),
    );
}
