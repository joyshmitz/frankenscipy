fn main() {
    let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let r = fsci_stats::bayes_mvs(&data, 0.95);
    println!("mean.statistic = {} (scipy: 5.5)", r.mean.statistic);
    println!("mean.low/high = {}/{} (scipy: 3.334/7.666)", r.mean.low, r.mean.high);
    println!("var.statistic = {} (scipy: 11.7857)", r.variance.statistic);
    println!("var.low/high = {}/{} (scipy: 4.337/30.551)", r.variance.low, r.variance.high);
    println!("std.statistic = {} (scipy: 3.3130)", r.std.statistic);
    println!("std.low/high = {}/{} (scipy: 2.083/5.527)", r.std.low, r.std.high);
}
