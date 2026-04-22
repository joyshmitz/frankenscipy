#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, GumbelLeft, Laplace, Logistic, Lognormal, Lomax, Maxwell, Normal, Pareto, Rayleigh,
    StudentT, Triangular, Uniform, Weibull,
};
use libfuzzer_sys::fuzz_target;
use std::hint::black_box;

#[derive(Debug, Arbitrary)]
struct FitInput {
    data: Vec<f64>,
}

fuzz_target!(|input: FitInput| {
    if input.data.len() > 1_000_000 {
        return;
    }
    let data = input.data.as_slice();

    // This harness is about panic-survival on hostile inputs, not enforcing a
    // single output-class convention across heterogeneous fit APIs.
    black_box(Normal::fit(data));
    black_box(StudentT::fit(data));
    black_box(ChiSquared::fit(data));
    black_box(Uniform::fit(data));
    black_box(Exponential::fit(data));
    black_box(FDistribution::fit(data));
    black_box(BetaDist::fit(data));
    black_box(GammaDist::fit(data));
    black_box(Weibull::fit(data));
    black_box(Lognormal::fit(data));
    black_box(Pareto::fit(data));
    black_box(Lomax::fit(data));
    black_box(Rayleigh::fit(data));
    black_box(Gumbel::fit(data));
    black_box(GumbelLeft::fit(data));
    black_box(Logistic::fit(data));
    black_box(Maxwell::fit(data));
    black_box(Cauchy::fit(data));
    black_box(Laplace::fit(data));
    black_box(Triangular::fit(data));
});
