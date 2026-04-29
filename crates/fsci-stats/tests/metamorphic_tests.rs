//! Metamorphic tests for `fsci-stats`.
//!
//! Each test checks an oracle-free relation: distribution round-trips,
//! correlation symmetries, t-test sign invariance, etc.
//!
//! Run with: `cargo test -p fsci-stats --test metamorphic_tests`

use fsci_stats::{
    BetaDist, ChiSquared, ContinuousDistribution, Exponential, GammaDist, Normal, StudentT,
    Uniform, ks_2samp, pearsonr, spearmanr, ttest_1samp, ttest_ind,
};

const ATOL: f64 = 1e-8;
const RTOL: f64 = 1e-7;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(close(a, b), "{msg}: actual={a:.16e} expected={b:.16e}");
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — cdf-ppf round-trip: ppf(cdf(x)) ≈ x for x in the support.
// ─────────────────────────────────────────────────────────────────────

fn cdf_ppf_roundtrip<D: ContinuousDistribution>(d: &D, xs: &[f64], label: &str) {
    for &x in xs {
        let p = d.cdf(x);
        // Skip endpoints where ppf returns ±inf.
        if !(p > 1e-12 && p < 1.0 - 1e-12) {
            continue;
        }
        let x_back = d.ppf(p);
        // Use a slightly looser bound near the tails because ppf has
        // steep slope there.
        let tol = 1e-6 * (1.0 + x.abs());
        assert!(
            (x_back - x).abs() <= tol,
            "{label} ppf(cdf(x)) round-trip: x={x} cdf={p:.16e} ppf={x_back} diff={}",
            (x_back - x).abs()
        );
    }
}

#[test]
fn mr_normal_cdf_ppf_roundtrip() {
    let d = Normal {
        loc: 0.5,
        scale: 1.5,
    };
    let xs: Vec<f64> = (-30..=30).map(|i| 0.2 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Normal");
}

#[test]
fn mr_uniform_cdf_ppf_roundtrip() {
    let d = Uniform {
        loc: -2.0,
        scale: 5.0,
    };
    let xs: Vec<f64> = (1..=49).map(|i| -2.0 + 5.0 * i as f64 / 50.0).collect();
    cdf_ppf_roundtrip(&d, &xs, "Uniform");
}

#[test]
fn mr_exponential_cdf_ppf_roundtrip() {
    let d = Exponential::new(2.0);
    let xs: Vec<f64> = (1..=40).map(|i| 0.05 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Exponential");
}

#[test]
fn mr_chisquared_cdf_ppf_roundtrip() {
    let d = ChiSquared::new(5.0);
    let xs: Vec<f64> = (1..=40).map(|i| 0.5 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "ChiSquared df=5");
}

#[test]
fn mr_studentt_cdf_ppf_roundtrip() {
    let d = StudentT::new(3.0);
    let xs: Vec<f64> = (-30..=30).map(|i| 0.25 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "StudentT df=3");
}

#[test]
fn mr_gamma_cdf_ppf_roundtrip() {
    let d = GammaDist {
        a: 2.5,
        scale: 1.0,
    };
    let xs: Vec<f64> = (1..=40).map(|i| 0.2 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Gamma a=2.5");
}

#[test]
fn mr_beta_cdf_ppf_roundtrip() {
    let d = BetaDist { a: 2.0, b: 5.0 };
    let xs: Vec<f64> = (1..=49).map(|i| i as f64 / 50.0).collect();
    cdf_ppf_roundtrip(&d, &xs, "Beta a=2,b=5");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — cdf monotonicity: x1 < x2 ⇒ cdf(x1) ≤ cdf(x2)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdf_monotonic() {
    let d = Normal {
        loc: 0.0,
        scale: 1.0,
    };
    let xs: Vec<f64> = (-50..=50).map(|i| 0.1 * i as f64).collect();
    let cdfs: Vec<f64> = xs.iter().map(|&x| d.cdf(x)).collect();
    for i in 1..cdfs.len() {
        assert!(
            cdfs[i] >= cdfs[i - 1] - 1e-15,
            "Normal cdf not monotone at i={i}: cdfs[{}]={} cdfs[{}]={}",
            i - 1,
            cdfs[i - 1],
            i,
            cdfs[i]
        );
    }
    // Bounds.
    assert!(cdfs[0] >= 0.0 && cdfs[0] < 1e-3);
    assert!(cdfs[cdfs.len() - 1] > 1.0 - 1e-3 && cdfs[cdfs.len() - 1] <= 1.0);
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — cdf + sf = 1 (identity)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdf_plus_sf_equals_one() {
    let d = Normal {
        loc: 0.0,
        scale: 1.0,
    };
    for x in (-50..=50).map(|i| 0.1 * i as f64) {
        let c = d.cdf(x);
        let s = d.sf(x);
        // Loosened tolerance for the deep tail: SciPy's default sf
        // implementation can drop precision when cdf saturates near 1.
        assert!(
            (c + s - 1.0).abs() <= 1e-9 * (1.0 + c.abs() + s.abs()),
            "MR3 cdf+sf != 1 at x={x}: cdf={c} sf={s} sum={}",
            c + s
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — Pearson correlation symmetry: pearsonr(x, y) == pearsonr(y, x)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pearsonr_symmetric() {
    let x = vec![1.0, 2.0, 3.5, 4.2, 6.0, 7.5, 9.0, 10.0];
    let y = vec![2.1, 4.0, 5.8, 6.9, 9.5, 11.2, 13.0, 14.7];
    let rxy = pearsonr(&x, &y);
    let ryx = pearsonr(&y, &x);
    assert_close(rxy.statistic, ryx.statistic, "MR4 pearsonr symmetric stat");
    assert_close(rxy.pvalue, ryx.pvalue, "MR4 pearsonr symmetric pvalue");
}

#[test]
fn mr_spearmanr_symmetric() {
    let x = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let rxy = spearmanr(&x, &y);
    let ryx = spearmanr(&y, &x);
    assert_close(rxy.statistic, ryx.statistic, "MR4 spearmanr symmetric stat");
    assert_close(rxy.pvalue, ryx.pvalue, "MR4 spearmanr symmetric pvalue");
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — Pearson on identical sequences = +1; on negated sequence = -1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pearsonr_identical_and_anti() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let r_self = pearsonr(&x, &x);
    assert!(
        (r_self.statistic - 1.0).abs() < 1e-12,
        "MR5 pearsonr(x, x) != 1: got {}",
        r_self.statistic
    );
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    let r_neg = pearsonr(&x, &neg);
    assert!(
        (r_neg.statistic + 1.0).abs() < 1e-12,
        "MR5 pearsonr(x, -x) != -1: got {}",
        r_neg.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — t-test sign symmetry: ttest_ind(b, a).statistic = -ttest_ind(a, b).statistic
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ttest_ind_sign_symmetry() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let ab = ttest_ind(&a, &b);
    let ba = ttest_ind(&b, &a);
    assert_close(ab.statistic, -ba.statistic, "MR6 ttest_ind sign symmetry");
    assert_close(ab.pvalue, ba.pvalue, "MR6 ttest_ind pvalue symmetry");
    assert_close(ab.df, ba.df, "MR6 ttest_ind df symmetry");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — KS-2samp swap symmetry: ks_2samp(a, b) ≈ ks_2samp(b, a) (statistic
//        and pvalue identical because the test is symmetric).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ks_2samp_swap_symmetric() {
    let a = vec![1.0, 2.0, 2.5, 3.0, 4.0, 5.5, 7.0, 8.5, 10.0, 12.0];
    let b = vec![1.5, 3.5, 4.5, 5.0, 6.0, 7.5, 9.0, 10.5, 11.0, 13.5];
    let ab = ks_2samp(&a, &b);
    let ba = ks_2samp(&b, &a);
    assert_close(ab.statistic, ba.statistic, "MR7 ks_2samp swap stat");
    assert_close(ab.pvalue, ba.pvalue, "MR7 ks_2samp swap pvalue");
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — ttest_1samp shift invariance: subtracting the popmean from
//        every observation moves the popmean to 0 and yields the same
//        t-statistic.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ttest_1samp_shift_invariance() {
    let data = vec![5.1, 5.4, 4.9, 5.5, 5.0, 5.6, 5.3, 5.2];
    let popmean = 5.0;
    let r1 = ttest_1samp(&data, popmean);
    let shifted: Vec<f64> = data.iter().map(|&v| v - popmean).collect();
    let r2 = ttest_1samp(&shifted, 0.0);
    assert_close(
        r1.statistic,
        r2.statistic,
        "MR8 ttest_1samp shift invariance stat",
    );
    assert_close(
        r1.pvalue,
        r2.pvalue,
        "MR8 ttest_1samp shift invariance pvalue",
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — pdf is non-negative and cdf is in [0, 1] across the support.
// ─────────────────────────────────────────────────────────────────────

fn check_pdf_cdf<D: ContinuousDistribution>(d: &D, label: &str) {
    for x in (-50..=50).map(|i| 0.2 * i as f64) {
        let p = d.pdf(x);
        let c = d.cdf(x);
        assert!(p >= 0.0 || p.is_nan(), "MR9 {label} pdf({x}) < 0: {p}");
        assert!(
            (0.0..=1.0).contains(&c) || c.is_nan(),
            "MR9 {label} cdf({x}) outside [0,1]: {c}"
        );
    }
}

#[test]
fn mr_pdf_nonneg_cdf_in_unit_interval() {
    check_pdf_cdf(&Normal::default(), "Normal(0,1)");
    check_pdf_cdf(&Uniform::default(), "Uniform(0,1)");
    check_pdf_cdf(&Exponential::new(1.0), "Exponential(1)");
    check_pdf_cdf(&ChiSquared::new(3.0), "ChiSquared(3)");
    check_pdf_cdf(&StudentT::new(5.0), "StudentT(5)");
}
