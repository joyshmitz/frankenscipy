//! expm_frechet / expm_cond probe vs scipy.linalg.
//! Lines: `tag,...`. The python comparator uses the same matrices.
use fsci_linalg::{DecompOptions, expm_cond, expm_frechet};

fn main() {
    let opts = DecompOptions::default();

    let cases: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![
        (
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.5, 0.1], vec![0.2, 0.3]],
        ),
        (
            vec![
                vec![-0.3, 0.2, 0.6],
                vec![0.6, 0.3, -0.1],
                vec![-0.7, 1.2, 0.9],
            ],
            vec![
                vec![0.1, 0.0, -0.2],
                vec![0.3, 0.4, 0.1],
                vec![0.0, -0.1, 0.2],
            ],
        ),
    ];

    for (ci, (a, e)) in cases.iter().enumerate() {
        let (_expm_a, l) = expm_frechet(a, e, opts).unwrap();
        for (i, row) in l.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                println!("frechet{ci},{i},{j},{v:.17e}");
            }
        }
        let k = expm_cond(a, opts).unwrap();
        println!("cond{ci},{k:.17e}");
    }
}
