//! Differential oracle probe: matrix functions vs scipy.linalg (gitignored).
//! Lines: `func,mat,r,c,value` or `func,mat,ERR`. Matrices defined identically in python cmp.
use fsci_linalg::{
    DecompOptions, coshm, cosm, expm, fractional_matrix_power, logm, signm, sinhm, sinm, sqrtm,
};

fn dump(func: &str, mat: &str, m: &[Vec<f64>]) {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            println!("{func},{mat},{r},{c},{v:.17e}");
        }
    }
}

fn run(func: &str, mat: &str, r: Result<Vec<Vec<f64>>, fsci_linalg::LinalgError>) {
    match r {
        Ok(m) => dump(func, mat, &m),
        Err(e) => println!("{func},{mat},ERR,{e:?}"),
    }
}

fn main() {
    let o = || DecompOptions::default();

    // M1: small spd-ish general matrix (well conditioned, real eigenvalues)
    let m1 = vec![
        vec![4.0, 1.0, 0.5],
        vec![1.0, 3.0, 0.2],
        vec![0.5, 0.2, 2.0],
    ];
    // M2: nonsymmetric with complex eigenvalues (rotation-like block)
    let m2 = vec![
        vec![0.5, -1.2, 0.3],
        vec![1.1, 0.4, -0.2],
        vec![0.0, 0.6, 0.9],
    ];
    // M3: diagonalizable nonsymmetric, distinct positive eigenvalues (logm/sqrtm safe)
    let m3 = vec![
        vec![3.0, 0.4, 0.1],
        vec![0.0, 2.0, 0.3],
        vec![0.0, 0.0, 1.5],
    ];

    for (mat, a) in [("m1", &m1), ("m2", &m2), ("m3", &m3)] {
        run("expm", mat, expm(a, o()));
        run("sinm", mat, sinm(a, o()));
        run("cosm", mat, cosm(a, o()));
        run("sinhm", mat, sinhm(a, o()));
        run("coshm", mat, coshm(a, o()));
        run("signm", mat, signm(a, o()));
    }
    // logm / sqrtm / frac power on matrices with positive real spectra (m1, m3)
    for (mat, a) in [("m1", &m1), ("m3", &m3)] {
        run("logm", mat, logm(a, o()));
        run("sqrtm", mat, sqrtm(a, o()));
        run("fracpow0.5", mat, fractional_matrix_power(a, 0.5, o()));
        run("fracpow1.5", mat, fractional_matrix_power(a, 1.5, o()));
    }
}
