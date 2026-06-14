//! multivariate root() probe vs scipy.optimize.root (gitignored).
use fsci_opt::root::{MultivariateRootMethod, MultivariateRootOptions, root};
fn main() {
    let f1 = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] * x[1] - 1.0];
    let f2 = |x: &[f64]| {
        vec![
            x[0] + x[1] + x[2] - 3.0,
            x[0] * x[0] + x[1] * x[1] - x[2] - 1.0,
            x[0] * x[1] * x[2] - 1.0,
        ]
    };
    use MultivariateRootMethod::*;
    for (mn, m) in [
        ("hybr", Hybr),
        ("broyden1", Broyden1),
        ("broyden2", Broyden2),
        ("anderson", Anderson),
        ("lm", Lm),
    ] {
        let o = MultivariateRootOptions {
            method: m,
            ..Default::default()
        };
        match root(f1, &[2.0, 0.5], o) {
            Ok(r) => println!(
                "s1_{mn},x,{:?},conv,{},resid,{:.2e}",
                r.x,
                r.converged,
                r.fun.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
            ),
            Err(e) => println!("s1_{mn},ERR,{e:?}"),
        }
        let o2 = MultivariateRootOptions {
            method: m,
            ..Default::default()
        };
        match root(f2, &[1.5, 1.5, 0.5], o2) {
            Ok(r) => println!(
                "s2_{mn},x,{:?},conv,{},resid,{:.2e}",
                r.x,
                r.converged,
                r.fun.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
            ),
            Err(e) => println!("s2_{mn},ERR,{e:?}"),
        }
    }
}
