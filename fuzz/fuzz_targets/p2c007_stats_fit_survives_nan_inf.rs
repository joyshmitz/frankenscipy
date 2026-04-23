#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, GumbelLeft, Laplace, Logistic, Lognormal, Lomax, Maxwell, Normal, Pareto, Rayleigh,
    StudentT, Triangular, Uniform, Weibull,
};
use libfuzzer_sys::fuzz_target;

/// Structural invariant for a distribution fit output: every parameter
/// must be finite (legitimate data), OR every parameter must be NaN
/// (degenerate / invalid data), OR every parameter must be +Inf (well-
/// defined degenerate cases like Exponential::fit(all-zeros) = +Inf).
/// The mixed case (finite + NaN or finite + Inf) signals a bug: fit()
/// returned a partially-degenerate distribution that is neither valid
/// nor uniformly signaling failure.
///
/// Per frankenscipy-m4hu: the prior `wo5w` fix replaced assert_params
/// with bare black_box(), downgrading oracle strength from metamorphic
/// (4) to crash-only (5). This restores per-distribution structural
/// invariants that accept each distribution's legitimate degenerate-
/// output classes.
#[track_caller]
fn assert_fit_structural(name: &str, params: &[f64]) {
    let all_finite = params.iter().all(|p| p.is_finite());
    let all_nan = params.iter().all(|p| p.is_nan());
    let all_pos_inf = params.iter().all(|p| p.is_infinite() && p.is_sign_positive());
    let all_neg_inf = params.iter().all(|p| p.is_infinite() && p.is_sign_negative());
    assert!(
        all_finite || all_nan || all_pos_inf || all_neg_inf,
        "{name}::fit returned partly-degenerate params: {:?} — expected all-finite OR all-NaN OR all-Inf-same-sign",
        params
    );
}

#[derive(Debug, Arbitrary)]
struct FitInput {
    data: Vec<f64>,
}

fuzz_target!(|input: FitInput| {
    if input.data.len() > 1_000_000 {
        return;
    }
    let data = input.data.as_slice();

    let normal = Normal::fit(data);
    assert_fit_structural("Normal", &[normal.loc, normal.scale]);

    let student = StudentT::fit(data);
    assert_fit_structural("StudentT", &[student.df]);

    let chi2 = ChiSquared::fit(data);
    assert_fit_structural("ChiSquared", &[chi2.df]);

    let uniform = Uniform::fit(data);
    assert_fit_structural("Uniform", &[uniform.loc, uniform.scale]);

    let exponential = Exponential::fit(data);
    assert_fit_structural("Exponential", &[exponential.lambda]);

    let f = FDistribution::fit(data);
    assert_fit_structural("FDistribution", &[f.dfn, f.dfd]);

    let beta = BetaDist::fit(data);
    assert_fit_structural("BetaDist", &[beta.a, beta.b]);

    let gamma = GammaDist::fit(data);
    assert_fit_structural("GammaDist", &[gamma.a, gamma.scale]);

    let weibull = Weibull::fit(data);
    assert_fit_structural("Weibull", &[weibull.c, weibull.scale]);

    let lognorm = Lognormal::fit(data);
    assert_fit_structural("Lognormal", &[lognorm.s, lognorm.scale]);

    let pareto = Pareto::fit(data);
    assert_fit_structural("Pareto", &[pareto.b, pareto.scale]);

    let lomax = Lomax::fit(data);
    assert_fit_structural("Lomax", &[lomax.c]);

    let rayleigh = Rayleigh::fit(data);
    assert_fit_structural("Rayleigh", &[rayleigh.scale]);

    let gumbel = Gumbel::fit(data);
    assert_fit_structural("Gumbel", &[gumbel.loc, gumbel.scale]);

    let gumbel_l = GumbelLeft::fit(data);
    assert_fit_structural("GumbelLeft", &[gumbel_l.loc, gumbel_l.scale]);

    let logistic = Logistic::fit(data);
    assert_fit_structural("Logistic", &[logistic.loc, logistic.scale]);

    let maxwell = Maxwell::fit(data);
    assert_fit_structural("Maxwell", &[maxwell.scale]);

    let cauchy = Cauchy::fit(data);
    assert_fit_structural("Cauchy", &[cauchy.loc, cauchy.scale]);

    let laplace = Laplace::fit(data);
    assert_fit_structural("Laplace", &[laplace.loc, laplace.scale]);

    let triangular = Triangular::fit(data);
    assert_fit_structural(
        "Triangular",
        &[triangular.left, triangular.mode, triangular.right],
    );
});

