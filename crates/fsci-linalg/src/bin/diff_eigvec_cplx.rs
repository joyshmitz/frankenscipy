use fsci_linalg::{DecompOptions, eig};
fn main() {
    // rot: eigenvalues ±i. dgeev packs complex eigvec as vr (col j) + i vi (col j+1).
    let mats: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("rot", vec![vec![0.0, -1.0], vec![1.0, 0.0]]),
        (
            "complex3",
            vec![
                vec![0.5, -1.2, 0.3],
                vec![1.1, 0.4, -0.2],
                vec![0.0, 0.6, 0.9],
            ],
        ),
    ];
    for (name, a) in &mats {
        let n = a.len();
        let r = eig(a, DecompOptions::default()).unwrap();
        for j in 0..n {
            println!(
                "{name},lam,{j},{:.17e},{:.17e}",
                r.eigenvalues_re[j], r.eigenvalues_im[j]
            );
            for i in 0..n {
                println!("{name},vec,{j}_{i},{:.17e}", r.eigenvectors[i][j]);
            }
        }
        // try to interpret as dgeev packing: for a conj pair (j, j+1), v = col_j + i col_{j+1}
        // check residual ||A v - lam v|| for the first of each pair
        let mut j = 0;
        while j < n {
            if r.eigenvalues_im[j].abs() > 1e-9 && j + 1 < n {
                let lam = (r.eigenvalues_re[j], r.eigenvalues_im[j]);
                let vr: Vec<f64> = (0..n).map(|i| r.eigenvectors[i][j]).collect();
                let vi: Vec<f64> = (0..n).map(|i| r.eigenvectors[i][j + 1]).collect();
                // (A (vr+i vi) - (lr+i li)(vr+i vi))
                let mut maxres = 0.0f64;
                for i in 0..n {
                    let avr: f64 = (0..n).map(|k| a[i][k] * vr[k]).sum();
                    let avi: f64 = (0..n).map(|k| a[i][k] * vi[k]).sum();
                    let re = avr - (lam.0 * vr[i] - lam.1 * vi[i]);
                    let im = avi - (lam.0 * vi[i] + lam.1 * vr[i]);
                    maxres = maxres.max((re * re + im * im).sqrt());
                }
                println!("{name},packresid,{j},{maxres:.6e}");
                j += 2;
            } else {
                j += 1;
            }
        }
    }
}
