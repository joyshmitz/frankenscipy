#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, GumbelLeft, Laplace, Logistic, Lognormal, Lomax, Maxwell, Normal, Pareto, Rayleigh,
    StudentT, Triangular, Uniform, Weibull,
};
use libfuzzer_sys::fuzz_target;
use p2c007_stats_common::{EdgeF64, approx_eq_prob};

#[derive(Debug, Arbitrary)]
struct DistributionInput {
    selector: u8,
    q: EdgeF64,
    a: EdgeF64,
    b: EdgeF64,
    c: EdgeF64,
    xs: Vec<EdgeF64>,
}

fn sanitized_xs(edges: &[EdgeF64]) -> Vec<f64> {
    let mut xs: Vec<f64> = edges
        .iter()
        .map(|edge| edge.finite(-1.0e4, 1.0e4, 0.0))
        .collect();
    if xs.is_empty() {
        xs.extend_from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    }
    xs.sort_by(f64::total_cmp);
    xs
}

fn check_distribution<D: ContinuousDistribution>(name: &str, dist: &D, q: f64, xs: &[f64]) {
    let ppf = dist.ppf(q);
    if ppf.is_finite() {
        let cdf_at_ppf = dist.cdf(ppf);
        if cdf_at_ppf.is_finite() {
            assert!(
                approx_eq_prob(cdf_at_ppf, q),
                "{name}: cdf(ppf(q)) drifted for q={q}, ppf={ppf}, cdf={cdf_at_ppf}"
            );
        }
    }

    let mut prev_cdf = None;
    for &x in xs {
        let pdf = dist.pdf(x);
        assert!(
            pdf.is_nan() || pdf >= 0.0,
            "{name}: expected non-negative pdf at x={x}, got {pdf}"
        );

        let cdf = dist.cdf(x);
        if cdf.is_finite() {
            assert!(
                (-1.0e-6..=1.0 + 1.0e-6).contains(&cdf),
                "{name}: expected cdf in [0,1] at x={x}, got {cdf}"
            );
            if let Some(prev) = prev_cdf {
                assert!(
                    cdf + 1.0e-6 >= prev,
                    "{name}: cdf not monotone between {prev} and {cdf} at x={x}"
                );
            }
            prev_cdf = Some(cdf);
        }
    }
}

fuzz_target!(|input: DistributionInput| {
    let q = input.q.probability();
    let xs = sanitized_xs(&input.xs);

    match input.selector % 20 {
        0 => {
            let dist = Normal::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Normal", &dist, q, &xs);
        }
        1 => {
            let dist = StudentT::new(input.a.positive(1.0e-6, 1.0e3, 5.0));
            check_distribution("StudentT", &dist, q, &xs);
        }
        2 => {
            let dist = ChiSquared::new(input.a.positive(1.0e-6, 1.0e3, 2.0));
            check_distribution("ChiSquared", &dist, q, &xs);
        }
        3 => {
            let dist = Uniform::new(
                input.a.finite(-1.0e3, 1.0e3, -1.0),
                input.b.positive(1.0e-6, 1.0e3, 2.0),
            );
            check_distribution("Uniform", &dist, q, &xs);
        }
        4 => {
            let dist = Exponential::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            check_distribution("Exponential", &dist, q, &xs);
        }
        5 => {
            let dist = FDistribution::new(
                input.a.positive(1.0e-6, 1.0e3, 4.0),
                input.b.positive(1.0e-6, 1.0e3, 7.0),
            );
            check_distribution("FDistribution", &dist, q, &xs);
        }
        6 => {
            let dist = BetaDist::new(
                input.a.positive(1.0e-6, 1.0e3, 2.0),
                input.b.positive(1.0e-6, 1.0e3, 3.0),
            );
            check_distribution("BetaDist", &dist, q, &xs);
        }
        7 => {
            let dist = GammaDist::new(
                input.a.positive(1.0e-6, 1.0e3, 2.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("GammaDist", &dist, q, &xs);
        }
        8 => {
            let dist = Weibull::new(
                input.a.positive(1.0e-6, 1.0e3, 1.5),
                input.b.positive(1.0e-6, 1.0e3, 2.0),
            );
            check_distribution("Weibull", &dist, q, &xs);
        }
        9 => {
            let dist = Lognormal::new(
                input.a.positive(1.0e-6, 1.0e3, 0.75),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Lognormal", &dist, q, &xs);
        }
        10 => {
            let dist = Pareto::new(
                input.a.positive(1.0e-6, 1.0e3, 3.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Pareto", &dist, q, &xs);
        }
        11 => {
            let dist = Lomax::new(input.a.positive(1.0e-6, 1.0e3, 2.0));
            check_distribution("Lomax", &dist, q, &xs);
        }
        12 => {
            let dist = Rayleigh::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            check_distribution("Rayleigh", &dist, q, &xs);
        }
        13 => {
            let dist = Gumbel::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Gumbel", &dist, q, &xs);
        }
        14 => {
            let dist = GumbelLeft::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("GumbelLeft", &dist, q, &xs);
        }
        15 => {
            let dist = Logistic::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Logistic", &dist, q, &xs);
        }
        16 => {
            let dist = Maxwell::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            check_distribution("Maxwell", &dist, q, &xs);
        }
        17 => {
            let dist = Cauchy::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Cauchy", &dist, q, &xs);
        }
        18 => {
            let dist = Laplace::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            check_distribution("Laplace", &dist, q, &xs);
        }
        _ => {
            let left = input.a.finite(-1.0e3, 1.0e3, -1.0);
            let width = input.b.positive(1.0e-6, 1.0e3, 2.0);
            let right = left + width;
            let mode = (left + input.c.probability() * width).clamp(left + 1.0e-9, right);
            let dist = Triangular { left, mode, right };
            check_distribution("Triangular", &dist, q, &xs);
        }
    }
});
