// Manual scipy-anchor verification for cramervonmises one-sample
// (frankenscipy-4cv67). Exercises the public `cramervonmises` API on
// uniformly-distributed fixtures and compares the resulting pvalue to
// scipy.stats.cramervonmises(data, 'uniform') reference values.
//
// The internal cvm_cdf is private; the outer cramervonmises call is what
// downstream consumers see, so we anchor that.

fn main() {
    use fsci_stats::cramervonmises;
    // (data, scipy.stats.cramervonmises(data, 'uniform').pvalue)
    let cases: [(&[f64], f64); 4] = [
        (&[0.1, 0.2, 0.3, 0.4, 0.5], 0.11944716780950626),
        (&[0.05, 0.15, 0.25, 0.4, 0.55, 0.7], 0.33174208310577924),
        (
            &[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95],
            0.9860137589521115,
        ),
        (
            &[
                0.02, 0.07, 0.13, 0.21, 0.29, 0.34, 0.41, 0.47, 0.53, 0.6, 0.66, 0.72, 0.79, 0.88,
            ],
            0.7547845391130281,
        ),
    ];
    let mut max_diff: f64 = 0.0;
    for (data, expected_p) in cases {
        let r = cramervonmises(data, |x| x.clamp(0.0, 1.0));
        let diff = (r.pvalue - expected_p).abs();
        max_diff = max_diff.max(diff);
        println!(
            "n={:2}: stat={:.10} p={:.10} scipy={:.10} diff={:.2e}",
            data.len(),
            r.statistic,
            r.pvalue,
            expected_p,
            diff
        );
    }
    println!("max_diff = {:.2e}", max_diff);
}
