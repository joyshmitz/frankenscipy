fn zscore(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }
    let n = data.len() as f64;
    let mean_val = data.iter().sum::<f64>() / n;
    let std_val = (data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std_val == 0.0 {
        return vec![0.0; data.len()];
    }
    data.iter().map(|&x| (x - mean_val) / std_val).collect()
}
fn main() {
    println!("{:?}", zscore(&[1.0, 2.0, 3.0]));
}
