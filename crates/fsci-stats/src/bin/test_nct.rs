// Quick verification harness for NoncentralT
use fsci_stats::{ContinuousDistribution, NoncentralT};

fn main() {
    let cases = [
        (
            "nct(10, 1.0).ppf(0.5)",
            NoncentralT::new(10.0, 1.0).ppf(0.5),
            1.0257209228280022,
        ),
        (
            "nct(20, 2.0).ppf(0.5)",
            NoncentralT::new(20.0, 2.0).ppf(0.5),
            2.025952753998231,
        ),
        (
            "nct(15, -2.5).cdf(1.5)",
            NoncentralT::new(15.0, -2.5).cdf(1.5),
            0.9999395205428545,
        ),
        (
            "nct(15, -2.5).cdf(3.0)",
            NoncentralT::new(15.0, -2.5).cdf(3.0),
            0.9999994362738605,
        ),
        (
            "nct(15, -2.5).pdf(3.0)",
            NoncentralT::new(15.0, -2.5).pdf(3.0),
            1.6570469892035188e-06,
        ),
        (
            "nct(5, 0.5).cdf(0.5)",
            NoncentralT::new(5.0, 0.5).cdf(0.5),
            0.49036270180675584,
        ),
        (
            "nct(5, 0.5).cdf(1.5)",
            NoncentralT::new(5.0, 0.5).cdf(1.5),
            0.7997685641492409,
        ),
        (
            "nct(5, 0.5).pdf(0.0)",
            NoncentralT::new(5.0, 0.5).pdf(0.0),
            0.33500172796874267,
        ),
        (
            "nct(3, -1).cdf(-1.5)",
            NoncentralT::new(3.0, -1.0).cdf(-1.5),
            0.37424409594356617,
        ),
    ];
    for (name, ours, scipy) in cases.iter() {
        let abs_diff = (ours - scipy).abs();
        let pass = if scipy.abs() > 1e-3 {
            abs_diff < 1e-3 * scipy.abs().max(1.0)
        } else {
            abs_diff < 1e-6
        };
        println!(
            "{}: ours={:.10}, scipy={:.10}, diff={:.3e} {}",
            name,
            ours,
            scipy,
            abs_diff,
            if pass { "OK" } else { "FAIL" }
        );
    }
}
