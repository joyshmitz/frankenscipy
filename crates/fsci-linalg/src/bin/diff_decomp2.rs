//! Differential probe: pinv/lstsq/svdvals/null_space/orth/subspace_angles vs scipy.linalg
use fsci_linalg as la;
fn m(name: &str, mat: &[Vec<f64>]) {
    let r = mat.len();
    let c = if r > 0 { mat[0].len() } else { 0 };
    let mut s = Vec::new();
    for row in mat {
        for &v in row {
            s.push(format!("{v:.17e}"));
        }
    }
    println!("{name}|{r}|{c}|{}", s.join(";"));
}
fn v(name: &str, vec: &[f64]) {
    let s: Vec<String> = vec.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|1|{}|{}", vec.len(), s.join(";"));
}
fn main() {
    // matrices: square full, tall, wide, rank-deficient square, near-singular
    let sq: Vec<Vec<f64>> = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let tall: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 9.0],
    ];
    let wide: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 5.0, 1.0]];
    let rankdef: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![1.0, 1.0, 1.0],
    ];
    let do_dec = |name: &str, a: &Vec<Vec<f64>>| {
        if let Ok(p) = la::pinv(a, la::PinvOptions::default()) {
            m(&format!("pinv_{name}"), &p.pseudo_inverse);
        }
        if let Ok(s) = la::svdvals(a, la::DecompOptions::default()) {
            v(&format!("svdvals_{name}"), &s);
        }
        if let Ok(ns) = la::null_space(a, None, la::DecompOptions::default()) {
            let r = ns.len();
            let c = if r > 0 { ns[0].len() } else { 0 };
            println!("nulldim_{name}|1|1|{}", c as f64);
        }
        if let Ok(o) = la::orth(a, None, la::DecompOptions::default()) {
            let r = o.len();
            let c = if r > 0 { o[0].len() } else { 0 };
            println!("orthdim_{name}|1|1|{}", c as f64);
        }
    };
    do_dec("sq", &sq);
    do_dec("tall", &tall);
    do_dec("wide", &wide);
    do_dec("rankdef", &rankdef);
    // lstsq: solve A x = b
    let b_sq = vec![1.0, 2.0, 3.0];
    if let Ok(r) = la::lstsq(&sq, &b_sq, la::LstsqOptions::default()) {
        v("lstsq_sq_x", &r.x);
        println!("lstsq_sq_rank|1|1|{}", r.rank as f64);
    }
    let b_tall = vec![1.0, 2.0, 2.0, 3.0];
    if let Ok(r) = la::lstsq(&tall, &b_tall, la::LstsqOptions::default()) {
        v("lstsq_tall_x", &r.x);
        v("lstsq_tall_res", &r.residuals);
        println!("lstsq_tall_rank|1|1|{}", r.rank as f64);
    }
    let b_rd = vec![1.0, 2.0, 1.0];
    if let Ok(r) = la::lstsq(&rankdef, &b_rd, la::LstsqOptions::default()) {
        v("lstsq_rd_x", &r.x);
        println!("lstsq_rd_rank|1|1|{}", r.rank as f64);
    }
    // subspace_angles
    let p = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    if let Ok(ang) = la::subspace_angles(&p, &q, la::DecompOptions::default()) {
        v("subspace_angles", &ang);
    }
}
