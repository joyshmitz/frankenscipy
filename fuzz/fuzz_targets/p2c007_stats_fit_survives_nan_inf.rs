#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, GumbelLeft, Laplace, Logistic, Lognormal, Lomax, Maxwell, Normal, Pareto, Rayleigh,
    StudentT, Triangular, Uniform, Weibull,
};
use libfuzzer_sys::fuzz_target;
use p2c007_stats_common::all_finite_or_all_nan;

#[derive(Debug, Arbitrary)]
struct FitInput {
    data: Vec<f64>,
}

fn assert_params(name: &str, params: &[f64]) {
    assert!(
        all_finite_or_all_nan(params),
        "{name}: expected all-finite or all-NaN fit params, got {params:?}"
    );
}

fuzz_target!(|input: FitInput| {
    if input.data.len() > 1_000_000 {
        return;
    }
    let data = input.data.as_slice();

    let normal = Normal::fit(data);
    assert_params("Normal", &[normal.loc, normal.scale]);

    let student = StudentT::fit(data);
    assert_params("StudentT", &[student.df]);

    let chi_squared = ChiSquared::fit(data);
    assert_params("ChiSquared", &[chi_squared.df]);

    let uniform = Uniform::fit(data);
    assert_params("Uniform", &[uniform.loc, uniform.scale]);

    let exponential = Exponential::fit(data);
    assert_params("Exponential", &[exponential.lambda]);

    let f_dist = FDistribution::fit(data);
    assert_params("FDistribution", &[f_dist.dfn, f_dist.dfd]);

    let beta = BetaDist::fit(data);
    assert_params("BetaDist", &[beta.a, beta.b]);

    let gamma = GammaDist::fit(data);
    assert_params("GammaDist", &[gamma.a, gamma.scale]);

    let weibull = Weibull::fit(data);
    assert_params("Weibull", &[weibull.c, weibull.scale]);

    let lognormal = Lognormal::fit(data);
    assert_params("Lognormal", &[lognormal.s, lognormal.scale]);

    let pareto = Pareto::fit(data);
    assert_params("Pareto", &[pareto.b, pareto.scale]);

    let lomax = Lomax::fit(data);
    assert_params("Lomax", &[lomax.c]);

    let rayleigh = Rayleigh::fit(data);
    assert_params("Rayleigh", &[rayleigh.scale]);

    let gumbel = Gumbel::fit(data);
    assert_params("Gumbel", &[gumbel.loc, gumbel.scale]);

    let gumbel_left = GumbelLeft::fit(data);
    assert_params("GumbelLeft", &[gumbel_left.loc, gumbel_left.scale]);

    let logistic = Logistic::fit(data);
    assert_params("Logistic", &[logistic.loc, logistic.scale]);

    let maxwell = Maxwell::fit(data);
    assert_params("Maxwell", &[maxwell.scale]);

    let cauchy = Cauchy::fit(data);
    assert_params("Cauchy", &[cauchy.loc, cauchy.scale]);

    let laplace = Laplace::fit(data);
    assert_params("Laplace", &[laplace.loc, laplace.scale]);

    let triangular = Triangular::fit(data);
    assert_params(
        "Triangular",
        &[triangular.left, triangular.mode, triangular.right],
    );
});
