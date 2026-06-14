//! Differential probe: special-matrix constructors vs scipy.linalg (gitignored).
//! Lines: `name|rows|cols|v00;v01;...` row-major.
use fsci_linalg as la;
fn dump(name: &str, m: &[Vec<f64>]) {
    let r = m.len();
    let c = if r > 0 { m[0].len() } else { 0 };
    let mut s = Vec::new();
    for row in m {
        for &v in row {
            s.push(format!("{v:.17e}"));
        }
    }
    println!("{name}|{r}|{c}|{}", s.join(";"));
}
fn main() {
    for n in [2usize, 3, 4, 5, 6, 7] {
        dump(&format!("hilbert_{n}"), &la::hilbert(n));
        dump(&format!("invhilbert_{n}"), &la::invhilbert(n));
        dump(&format!("pascal_sym_{n}"), &la::pascal(n, true));
        dump(&format!("pascal_low_{n}"), &la::pascal(n, false));
        dump(&format!("helmert_{n}"), &la::helmert(n));
        dump(&format!("helmertfull_{n}"), &la::helmert_full(n));
    }
    // toeplitz / circulant / hankel
    dump(
        "toeplitz",
        &la::toeplitz(&[1.0, 2.0, 3.0, 4.0], Some(&[1.0, 5.0, 6.0])),
    );
    dump("circulant", &la::circulant(&[1.0, 2.0, 3.0, 4.0]));
    dump(
        "hankel",
        &la::hankel(&[1.0, 2.0, 3.0, 4.0], Some(&[4.0, 7.0, 8.0, 9.0])),
    );
    // fiedler / fiedler_companion / companion / leslie
    dump("fiedler", &la::fiedler(&[1.0, 2.0, 3.0, 4.0]));
    dump(
        "fiedler_companion",
        &la::fiedler_companion(&[1.0, -3.0, 2.0, 5.0, 1.0]),
    );
    dump(
        "companion",
        &la::companion(&[1.0, -6.0, 11.0, -6.0]).unwrap(),
    );
    dump(
        "leslie",
        &la::leslie(&[0.1, 2.0, 1.0, 0.1], &[0.2, 0.8, 0.7]).unwrap(),
    );
    // tri / convolution_matrix
    dump("tri", &la::tri(4, 5, 1));
    dump(
        "convmat_full",
        &la::convolution_matrix(&[1.0, 2.0, 3.0], 4, "full"),
    );
    dump(
        "convmat_same",
        &la::convolution_matrix(&[1.0, 2.0, 3.0], 4, "same"),
    );
    dump(
        "convmat_valid",
        &la::convolution_matrix(&[1.0, 2.0, 3.0], 4, "valid"),
    );
}
