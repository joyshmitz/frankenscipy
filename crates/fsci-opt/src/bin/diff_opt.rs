//! Differential oracle probe: opt root-finders / LSA / isotonic / nnls vs scipy (gitignored).
//! Lines: `name,key,value`. Inputs match the python comparator.
use fsci_opt::RootOptions;
use fsci_opt::{
    bisect, brenth, brentq, isotonic_regression, linear_sum_assignment, nnls, ridder, toms748,
};

type RootProb = (&'static str, fn(f64) -> f64, (f64, f64));
type IsoCase = (&'static str, Vec<f64>, Option<Vec<f64>>);

fn main() {
    // ---- root finders on several functions/brackets ----
    let probs: Vec<RootProb> = vec![
        ("cubic", |x| x * x * x - 2.0 * x - 5.0, (2.0, 3.0)),
        ("cos_x", |x| x.cos() - x, (0.0, 1.0)),
        ("exp", |x| (-x).exp() - x, (0.0, 1.0)),
        ("poly", |x| (x - 1.3) * (x - 1.3) * (x + 0.7), (0.5, 2.0)),
        ("trig", |x| (3.0 * x).sin() + 0.5, (0.5, 1.5)),
    ];
    for (name, f, br) in &probs {
        let o = RootOptions::default();
        if let Ok(r) = brentq(f, *br, o) {
            println!("brentq,{name},{:.17e}", r.root);
        }
        if let Ok(r) = brenth(f, *br, o) {
            println!("brenth,{name},{:.17e}", r.root);
        }
        if let Ok(r) = ridder(f, *br, o) {
            println!("ridder,{name},{:.17e}", r.root);
        }
        if let Ok(r) = toms748(f, *br, o) {
            println!("toms748,{name},{:.17e}", r.root);
        }
        if let Ok(r) = bisect(f, *br, o) {
            println!("bisect,{name},{:.17e}", r.root);
        }
    }

    // ---- linear_sum_assignment: compare optimal cost ----
    let costs: Vec<(&str, Vec<Vec<f64>>)> = vec![
        (
            "sq3",
            vec![
                vec![4.0, 1.0, 3.0],
                vec![2.0, 0.0, 5.0],
                vec![3.0, 2.0, 2.0],
            ],
        ),
        (
            "sq4",
            vec![
                vec![9.0, 2.0, 7.0, 8.0],
                vec![6.0, 4.0, 3.0, 7.0],
                vec![5.0, 8.0, 1.0, 8.0],
                vec![7.0, 6.0, 9.0, 4.0],
            ],
        ),
        (
            "rect2x4",
            vec![vec![1.5, 2.0, 3.0, 0.5], vec![4.0, 1.0, 2.5, 3.5]],
        ),
    ];
    for (name, c) in &costs {
        if let Ok((ri, ci)) = linear_sum_assignment(c) {
            let cost: f64 = ri.iter().zip(&ci).map(|(&r, &cc)| c[r][cc]).sum();
            println!("lsa,{name},{:.17e}", cost);
        }
    }

    // ---- isotonic_regression ----
    let ys: Vec<IsoCase> = vec![
        ("iso1", vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], None),
        (
            "iso_w",
            vec![1.0, 0.0, 2.0, 1.0, 3.0],
            Some(vec![1.0, 2.0, 1.0, 3.0, 1.0]),
        ),
    ];
    for (name, y, w) in &ys {
        let r = isotonic_regression(y, w.as_deref());
        for (i, &v) in r.iter().enumerate() {
            println!("iso,{name}_{i},{v:.17e}");
        }
    }

    // ---- nnls ----
    let a = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![2.0, 1.0],
    ];
    let b = vec![1.0, 2.0, 1.5, 3.0];
    if let Ok((x, rnorm)) = nnls(&a, &b) {
        for (i, &v) in x.iter().enumerate() {
            println!("nnls,x{i},{v:.17e}");
        }
        println!("nnls,rnorm,{rnorm:.17e}");
    }
}
