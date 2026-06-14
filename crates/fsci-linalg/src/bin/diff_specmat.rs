//! Differential oracle probe: special-matrix constructors vs scipy.linalg (gitignored).
//! Lines: `name,r,c,value`. Inputs match the python comparator.
use fsci_linalg::{
    circulant, companion, fiedler, fiedler_companion, hadamard, hankel, helmert, helmert_full,
    hilbert, invhilbert, leslie, pascal, toeplitz, tri, vander,
};

fn dump(name: &str, m: &[Vec<f64>]) {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            println!("{name},{r},{c},{v:.17e}");
        }
    }
}

fn main() {
    dump("hilbert5", &hilbert(5));
    dump("invhilbert4", &invhilbert(4));
    dump("invhilbert5", &invhilbert(5));
    dump("pascal_sym4", &pascal(4, true));
    dump("pascal_low4", &pascal(4, false));
    dump(
        "toeplitz",
        &toeplitz(&[1.0, 2.0, 3.0, 4.0], Some(&[1.0, 5.0, 6.0])),
    );
    dump("circulant", &circulant(&[1.0, 2.0, 3.0, 4.0]));
    dump(
        "hankel",
        &hankel(&[1.0, 2.0, 3.0, 4.0], Some(&[4.0, 7.0, 8.0, 9.0])),
    );
    dump("tri_5_4_1", &tri(5, 4, 1));
    dump("tri_4_4_-1", &tri(4, 4, -1));
    dump("vander_inc", &vander(&[1.0, 2.0, 3.0, 5.0], Some(4), true));
    dump("vander_dec", &vander(&[1.0, 2.0, 3.0, 5.0], None, false));
    dump("helmert5", &helmert(5));
    dump("helmert_full5", &helmert_full(5));
    dump("fiedler", &fiedler(&[1.0, 4.0, 12.0, 45.0]));
    dump(
        "fiedler_companion",
        &fiedler_companion(&[1.0, -3.0, 2.0, -5.0, 7.0]),
    );
    if let Ok(m) = hadamard(8) {
        dump("hadamard8", &m);
    }
    if let Ok(m) = companion(&[1.0, -10.0, 31.0, -30.0]) {
        dump("companion", &m);
    }
    if let Ok(m) = leslie(&[0.1, 2.0, 1.0, 0.1], &[0.2, 0.8, 0.7]) {
        dump("leslie", &m);
    }
}
