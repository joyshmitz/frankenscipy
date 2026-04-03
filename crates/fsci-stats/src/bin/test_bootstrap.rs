use fsci_stats::bootstrap_ci;

fn main() {
    let data = vec![1.0, 2.0, 3.0];
    let (lo, hi) = bootstrap_ci(&data, |d| d.iter().sum::<f64>(), 100, 2.0, 42);
    println!("lo: {}, hi: {}", lo, hi);
}
