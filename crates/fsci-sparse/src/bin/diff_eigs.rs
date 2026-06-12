use fsci_sparse::ops::FormatConvertible;
use fsci_sparse::{CooMatrix, EigsOptions, Shape2D, eigs};
fn main() {
    // 4x4: 2x2 block [[3,-4],[4,3]] (eig 3±4i, |.|=5) + diag 2,1
    let (r, c, v) = (
        vec![0, 0, 1, 1, 2, 3],
        vec![0, 1, 0, 1, 2, 3],
        vec![3.0, -4.0, 4.0, 3.0, 2.0, 1.0],
    );
    let a = CooMatrix::from_triplets(Shape2D::new(4, 4), v, r, c, false)
        .unwrap()
        .to_csr()
        .unwrap();
    match eigs(&a, 2, EigsOptions::default()) {
        Ok(res) => {
            // (re, im) pairs, sorted by complex magnitude desc — scipy returns 3±4i.
            let mut e: Vec<(f64, f64)> = res
                .eigenvalues
                .iter()
                .copied()
                .zip(res.eigenvalues_im.iter().copied())
                .collect();
            e.sort_by(|x, y| (y.0 * y.0 + y.1 * y.1).total_cmp(&(x.0 * x.0 + x.1 * x.1)));
            for (i, &(re, im)) in e.iter().enumerate() {
                println!("eigs_realpart,{i},{re:.10e}");
                println!("eigs_imagpart,{i},{im:.10e}");
                println!("eigs_abs,{i},{:.10e}", (re * re + im * im).sqrt());
            }
            println!("converged,{}", res.converged);
        }
        Err(e) => println!("ERR,{e:?}"),
    }
}
