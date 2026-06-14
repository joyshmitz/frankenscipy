//! Differential oracle probe: general eig eigenvalues vs scipy.linalg.eig (gitignored).
//! Lines: `name,ev,k,re,im` (sorted eigenvalues) and `name,resid,k,value` (real-ev residual).
use fsci_linalg::{DecompOptions, eig};

fn main() {
    let o = || DecompOptions::default();
    let mats: Vec<(&str, Vec<Vec<f64>>)> = vec![
        (
            "sym",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 3.0, 1.0],
                vec![0.0, 1.0, 2.0],
            ],
        ),
        (
            "real_nonsym",
            vec![
                vec![3.0, 0.4, 0.1],
                vec![0.0, 2.0, 0.3],
                vec![0.0, 0.0, 1.5],
            ],
        ),
        ("rot", vec![vec![0.0, -1.0], vec![1.0, 0.0]]),
        (
            "complex3",
            vec![
                vec![0.5, -1.2, 0.3],
                vec![1.1, 0.4, -0.2],
                vec![0.0, 0.6, 0.9],
            ],
        ),
        (
            "scalerot",
            vec![
                vec![1.0, -2.0, 0.0],
                vec![2.0, 1.0, 0.0],
                vec![0.0, 0.0, 4.0],
            ],
        ),
    ];
    for (name, a) in &mats {
        let n = a.len();
        let r = match eig(a, o()) {
            Ok(r) => r,
            Err(e) => {
                println!("{name},ERR,{e:?}");
                continue;
            }
        };
        // sort eigenvalues by (re, im)
        let mut evs: Vec<(f64, f64)> = r
            .eigenvalues_re
            .iter()
            .zip(&r.eigenvalues_im)
            .map(|(&re, &im)| (re, im))
            .collect();
        evs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (k, (re, im)) in evs.iter().enumerate() {
            println!("{name},ev,{k},{re:.17e},{im:.17e}");
        }
        // residual for each real eigenvalue's eigenvector column: ||A v - lambda v||_inf
        for j in 0..n {
            if r.eigenvalues_im[j].abs() > 1e-9 {
                continue; // complex column: storage convention varies, skip
            }
            let lam = r.eigenvalues_re[j];
            let v: Vec<f64> = (0..n).map(|i| r.eigenvectors[i][j]).collect();
            let mut resid: f64 = 0.0;
            for i in 0..n {
                let av: f64 = (0..n).map(|k| a[i][k] * v[k]).sum();
                resid = resid.max((av - lam * v[i]).abs());
            }
            println!("{name},resid,{j},{resid:.6e}");
        }
    }
}
