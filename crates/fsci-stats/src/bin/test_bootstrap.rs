use fsci_stats::bootstrap_mean;

fn main() {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("symmetric_5pt", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        (
            "asymmetric_8pt",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
        ),
        (
            "spread_15pt",
            vec![
                -2.0, -1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5,
            ],
        ),
    ];
    let n_bootstrap = 5000;
    let confidence = 0.95;
    let seed = 42u64;
    for (name, data) in &datasets {
        let (lo, hi) = bootstrap_mean(data, n_bootstrap, confidence, seed);
        let original_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        println!(
            "{}: mean={:.10} BCa_lo={:.10} BCa_hi={:.10}",
            name, original_mean, lo, hi
        );
    }
}
