#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{kendalltau, linregress, pearsonr, spearmanr};
use libfuzzer_sys::fuzz_target;

const MAX_LEN: usize = 128;
const TOL: f64 = 1e-10;

#[derive(Debug, Arbitrary)]
struct CorrelationInput {
    raw_x: Vec<f64>,
    raw_y: Vec<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn reference_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn reference_std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let mean = reference_mean(data);
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn reference_pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let mean_x = reference_mean(x);
    let mean_y = reference_mean(y);
    let std_x = reference_std(x);
    let std_y = reference_std(y);
    if std_x == 0.0 || std_y == 0.0 || !std_x.is_finite() || !std_y.is_finite() {
        return f64::NAN;
    }
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / (x.len() - 1) as f64;
    cov / (std_x * std_y)
}

fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn reference_spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let rank_x = rank(x);
    let rank_y = rank(y);
    reference_pearson(&rank_x, &rank_y)
}

fn reference_linregress(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    if x.len() != y.len() || x.len() < 2 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    let r = reference_pearson(x, y);

    (slope, intercept, r)
}

fuzz_target!(|input: CorrelationInput| {
    if input.raw_x.is_empty() || input.raw_y.is_empty() {
        return;
    }

    let len = input.raw_x.len().min(input.raw_y.len()).min(MAX_LEN);
    if len < 2 {
        return;
    }

    let x: Vec<f64> = input.raw_x.iter().take(len).map(|&v| sanitize(v)).collect();
    let y: Vec<f64> = input.raw_y.iter().take(len).map(|&v| sanitize(v)).collect();

    // Oracle 1: pearsonr statistic in [-1, 1]
    let pr = pearsonr(&x, &y);
    if pr.statistic.is_finite() {
        assert!(
            pr.statistic >= -1.0 - TOL && pr.statistic <= 1.0 + TOL,
            "pearsonr statistic {} out of [-1, 1]",
            pr.statistic
        );
    }

    // Oracle 2: pearsonr pvalue in [0, 1]
    if pr.pvalue.is_finite() {
        assert!(
            pr.pvalue >= -TOL && pr.pvalue <= 1.0 + TOL,
            "pearsonr pvalue {} out of [0, 1]",
            pr.pvalue
        );
    }

    // Oracle 3: pearsonr matches reference implementation
    let ref_pr = reference_pearson(&x, &y);
    if pr.statistic.is_finite() && ref_pr.is_finite() {
        let diff = (pr.statistic - ref_pr).abs();
        assert!(
            diff < 1e-8,
            "pearsonr {} != reference {} (diff {})",
            pr.statistic,
            ref_pr,
            diff
        );
    }

    // Oracle 4: spearmanr statistic in [-1, 1]
    let sr = spearmanr(&x, &y);
    if sr.statistic.is_finite() {
        assert!(
            sr.statistic >= -1.0 - TOL && sr.statistic <= 1.0 + TOL,
            "spearmanr statistic {} out of [-1, 1]",
            sr.statistic
        );
    }

    // Oracle 5: spearmanr pvalue in [0, 1]
    if sr.pvalue.is_finite() {
        assert!(
            sr.pvalue >= -TOL && sr.pvalue <= 1.0 + TOL,
            "spearmanr pvalue {} out of [0, 1]",
            sr.pvalue
        );
    }

    // Oracle 6: spearmanr matches reference implementation
    let ref_sr = reference_spearman(&x, &y);
    if sr.statistic.is_finite() && ref_sr.is_finite() {
        let diff = (sr.statistic - ref_sr).abs();
        assert!(
            diff < 1e-8,
            "spearmanr {} != reference {} (diff {})",
            sr.statistic,
            ref_sr,
            diff
        );
    }

    // Oracle 7: kendalltau statistic in [-1, 1]
    let kt = kendalltau(&x, &y);
    if kt.statistic.is_finite() {
        assert!(
            kt.statistic >= -1.0 - TOL && kt.statistic <= 1.0 + TOL,
            "kendalltau statistic {} out of [-1, 1]",
            kt.statistic
        );
    }

    // Oracle 8: kendalltau pvalue in [0, 1]
    if kt.pvalue.is_finite() {
        assert!(
            kt.pvalue >= -TOL && kt.pvalue <= 1.0 + TOL,
            "kendalltau pvalue {} out of [0, 1]",
            kt.pvalue
        );
    }

    // Oracle 9: linregress rvalue in [-1, 1]
    let lr = linregress(&x, &y);
    if lr.rvalue.is_finite() {
        assert!(
            lr.rvalue >= -1.0 - TOL && lr.rvalue <= 1.0 + TOL,
            "linregress rvalue {} out of [-1, 1]",
            lr.rvalue
        );
    }

    // Oracle 10: linregress pvalue in [0, 1]
    if lr.pvalue.is_finite() {
        assert!(
            lr.pvalue >= -TOL && lr.pvalue <= 1.0 + TOL,
            "linregress pvalue {} out of [0, 1]",
            lr.pvalue
        );
    }

    // Oracle 11: linregress matches reference slope/intercept
    let (ref_slope, ref_intercept, ref_r) = reference_linregress(&x, &y);
    if lr.slope.is_finite() && ref_slope.is_finite() {
        let rel_err = (lr.slope - ref_slope).abs() / ref_slope.abs().max(1.0);
        assert!(
            rel_err < 1e-6,
            "linregress slope {} != reference {} (rel_err {})",
            lr.slope,
            ref_slope,
            rel_err
        );
    }
    if lr.intercept.is_finite() && ref_intercept.is_finite() {
        let rel_err = (lr.intercept - ref_intercept).abs() / ref_intercept.abs().max(1.0);
        assert!(
            rel_err < 1e-6,
            "linregress intercept {} != reference {} (rel_err {})",
            lr.intercept,
            ref_intercept,
            rel_err
        );
    }
    if lr.rvalue.is_finite() && ref_r.is_finite() {
        let diff = (lr.rvalue - ref_r).abs();
        assert!(
            diff < 1e-8,
            "linregress rvalue {} != reference {} (diff {})",
            lr.rvalue,
            ref_r,
            diff
        );
    }

    // Oracle 12: linregress rvalue equals pearsonr statistic
    if lr.rvalue.is_finite() && pr.statistic.is_finite() {
        let diff = (lr.rvalue - pr.statistic).abs();
        assert!(
            diff < 1e-10,
            "linregress rvalue {} != pearsonr {} (diff {})",
            lr.rvalue,
            pr.statistic,
            diff
        );
    }

    // Oracle 13: pearsonr(x, x) = 1.0 for non-constant x
    let std_x = reference_std(&x);
    if std_x.is_finite() && std_x > 1e-10 {
        let pr_xx = pearsonr(&x, &x);
        if pr_xx.statistic.is_finite() {
            assert!(
                (pr_xx.statistic - 1.0).abs() < TOL,
                "pearsonr(x, x) = {} should be 1.0",
                pr_xx.statistic
            );
        }
    }

    // Oracle 14: spearmanr(x, x) = 1.0 for non-constant x
    if std_x.is_finite() && std_x > 1e-10 {
        let sr_xx = spearmanr(&x, &x);
        if sr_xx.statistic.is_finite() {
            assert!(
                (sr_xx.statistic - 1.0).abs() < TOL,
                "spearmanr(x, x) = {} should be 1.0",
                sr_xx.statistic
            );
        }
    }

    // Oracle 15: kendalltau(x, x) = 1.0 for non-constant x
    if std_x.is_finite() && std_x > 1e-10 {
        let kt_xx = kendalltau(&x, &x);
        if kt_xx.statistic.is_finite() {
            assert!(
                (kt_xx.statistic - 1.0).abs() < TOL,
                "kendalltau(x, x) = {} should be 1.0",
                kt_xx.statistic
            );
        }
    }

    // Oracle 16: linregress(x, x) has slope=1, intercept=0
    if std_x.is_finite() && std_x > 1e-10 {
        let lr_xx = linregress(&x, &x);
        if lr_xx.slope.is_finite() && lr_xx.intercept.is_finite() {
            assert!(
                (lr_xx.slope - 1.0).abs() < 1e-8,
                "linregress(x, x) slope = {} should be 1.0",
                lr_xx.slope
            );
            assert!(
                lr_xx.intercept.abs() < 1e-8,
                "linregress(x, x) intercept = {} should be 0.0",
                lr_xx.intercept
            );
        }
    }
});
