fn main() {
    use fsci_stats::pearsonr_alternative;
    let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = (1..=10).rev().map(|i| i as f64).collect();
    for alt in ["two-sided", "less", "greater"] {
        let r = pearsonr_alternative(&x, &y, alt);
        println!("anti(r=-1) {}: stat={} p={}", alt, r.statistic, r.pvalue);
    }
    let y2: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    for alt in ["two-sided", "less", "greater"] {
        let r = pearsonr_alternative(&x, &y2, alt);
        println!("perfect(r=1) {}: stat={} p={}", alt, r.statistic, r.pvalue);
    }
}
