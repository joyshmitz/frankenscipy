fn main() {
    use fsci_stats::{ContinuousDistribution, ExponPow};
    let d = ExponPow::new(0.5);
    for x in [1e-12_f64, 1e-9, 1e-6, 1e-3, 0.1, 0.5, 1.0, 5.0, 10.0] {
        let p = d.pdf(x);
        println!("x={x}: pdf={p:.6e}, -p ln(p) = {:.6e}", -p * p.ln());
    }
}
