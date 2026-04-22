#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, GumbelLeft, Laplace, Logistic, Lognormal, Lomax, Maxwell, Normal, Pareto, Rayleigh,
    StudentT, Triangular, Uniform, Weibull,
};
use libfuzzer_sys::fuzz_target;
use p2c007_stats_common::EdgeF64;
use rand::{SeedableRng, rngs::StdRng};

#[derive(Debug, Arbitrary)]
struct SampleInput {
    selector: u8,
    seed: u64,
    count: u16,
    a: EdgeF64,
    b: EdgeF64,
    c: EdgeF64,
}

fn assert_deterministic<D: ContinuousDistribution>(name: &str, dist: &D, seed: u64, count: usize) {
    let mut first_rng = StdRng::seed_from_u64(seed);
    let mut second_rng = StdRng::seed_from_u64(seed);

    let first = dist.rvs(count, &mut first_rng);
    let second = dist.rvs(count, &mut second_rng);

    assert_eq!(first.len(), count, "{name}: first sample length mismatch");
    assert_eq!(second.len(), count, "{name}: second sample length mismatch");

    for (index, (lhs, rhs)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{name}: nondeterministic sample at index {index}: {lhs:?} vs {rhs:?}"
        );
    }
}

fuzz_target!(|input: SampleInput| {
    let count = usize::from(input.count % 257);

    match input.selector % 20 {
        0 => {
            let dist = Normal::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Normal", &dist, input.seed, count);
        }
        1 => {
            let dist = StudentT::new(input.a.positive(1.0e-6, 1.0e3, 5.0));
            assert_deterministic("StudentT", &dist, input.seed, count);
        }
        2 => {
            let dist = ChiSquared::new(input.a.positive(1.0e-6, 1.0e3, 2.0));
            assert_deterministic("ChiSquared", &dist, input.seed, count);
        }
        3 => {
            let dist = Uniform::new(
                input.a.finite(-1.0e3, 1.0e3, -1.0),
                input.b.positive(1.0e-6, 1.0e3, 2.0),
            );
            assert_deterministic("Uniform", &dist, input.seed, count);
        }
        4 => {
            let dist = Exponential::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            assert_deterministic("Exponential", &dist, input.seed, count);
        }
        5 => {
            let dist = FDistribution::new(
                input.a.positive(1.0e-6, 1.0e3, 4.0),
                input.b.positive(1.0e-6, 1.0e3, 7.0),
            );
            assert_deterministic("FDistribution", &dist, input.seed, count);
        }
        6 => {
            let dist = BetaDist::new(
                input.a.positive(1.0e-6, 1.0e3, 2.0),
                input.b.positive(1.0e-6, 1.0e3, 3.0),
            );
            assert_deterministic("BetaDist", &dist, input.seed, count);
        }
        7 => {
            let dist = GammaDist::new(
                input.a.positive(1.0e-6, 1.0e3, 2.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("GammaDist", &dist, input.seed, count);
        }
        8 => {
            let dist = Weibull::new(
                input.a.positive(1.0e-6, 1.0e3, 1.5),
                input.b.positive(1.0e-6, 1.0e3, 2.0),
            );
            assert_deterministic("Weibull", &dist, input.seed, count);
        }
        9 => {
            let dist = Lognormal::new(
                input.a.positive(1.0e-6, 1.0e3, 0.75),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Lognormal", &dist, input.seed, count);
        }
        10 => {
            let dist = Pareto::new(
                input.a.positive(1.0e-6, 1.0e3, 3.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Pareto", &dist, input.seed, count);
        }
        11 => {
            let dist = Lomax::new(input.a.positive(1.0e-6, 1.0e3, 2.0));
            assert_deterministic("Lomax", &dist, input.seed, count);
        }
        12 => {
            let dist = Rayleigh::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            assert_deterministic("Rayleigh", &dist, input.seed, count);
        }
        13 => {
            let dist = Gumbel::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Gumbel", &dist, input.seed, count);
        }
        14 => {
            let dist = GumbelLeft::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("GumbelLeft", &dist, input.seed, count);
        }
        15 => {
            let dist = Logistic::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Logistic", &dist, input.seed, count);
        }
        16 => {
            let dist = Maxwell::new(input.a.positive(1.0e-6, 1.0e3, 1.0));
            assert_deterministic("Maxwell", &dist, input.seed, count);
        }
        17 => {
            let dist = Cauchy::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Cauchy", &dist, input.seed, count);
        }
        18 => {
            let dist = Laplace::new(
                input.a.finite(-1.0e3, 1.0e3, 0.0),
                input.b.positive(1.0e-6, 1.0e3, 1.0),
            );
            assert_deterministic("Laplace", &dist, input.seed, count);
        }
        _ => {
            let left = input.a.finite(-1.0e3, 1.0e3, -1.0);
            let width = input.b.positive(1.0e-6, 1.0e3, 2.0);
            let right = left + width;
            let mode = (left + input.c.probability() * width).clamp(left + 1.0e-9, right);
            let dist = Triangular { left, mode, right };
            assert_deterministic("Triangular", &dist, input.seed, count);
        }
    }
});
