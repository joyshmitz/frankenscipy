#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    digamma, erfcinv, erfinv, exp1, expi, gamma, gammaln, lambertw, rgamma, wrightomega,
};
use libfuzzer_sys::fuzz_target;

type UnarySpecialFn = fn(&SpecialTensor, RuntimeMode) -> SpecialResult;

const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum EdgeF64 {
    Finite(f64),
    Zero,
    NegZero,
    One,
    NegOne,
    Half,
    NegHalf,
    Tiny,
    NegTiny,
    PosInf,
    NegInf,
    Nan,
    LambertBranch,
    GammaPole,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e3, 1.0e3),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
            Self::LambertBranch => -1.0 / std::f64::consts::E,
            Self::GammaPole => -1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct ComplexBranchInput {
    re: EdgeF64,
    im: EdgeF64,
    hardened: bool,
    vectorize: bool,
}

fn mode_from_flag(hardened: bool) -> RuntimeMode {
    if hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    }
}

fn approx_eq_scalar(lhs: f64, rhs: f64) -> bool {
    if !(lhs.is_finite() && rhs.is_finite()) {
        return false;
    }
    let scale = lhs.abs().max(rhs.abs());
    (lhs - rhs).abs() <= ABS_TOL + REL_TOL * scale
}

fn approx_eq_complex(lhs: Complex64, rhs: Complex64) -> bool {
    approx_eq_scalar(lhs.re, rhs.re) && approx_eq_scalar(lhs.im, rhs.im)
}

fn complex_from_result(result: SpecialResult) -> Option<Complex64> {
    match result.ok()? {
        RealScalar(value) => Some(Complex64::from_real(value)),
        ComplexScalar(value) => Some(value),
        _ => None,
    }
}

fn complex_vec_from_result(result: SpecialResult) -> Option<Vec<Complex64>> {
    match result.ok()? {
        RealVec(values) => Some(values.into_iter().map(Complex64::from_real).collect()),
        ComplexVec(values) => Some(values),
        _ => None,
    }
}

fn check_unary(name: &str, func: UnarySpecialFn, z: Complex64, mode: RuntimeMode, vectorize: bool) {
    let _ = func(&ComplexScalar(z), mode);

    if z.is_finite() && z.im != 0.0 {
        let lhs = complex_from_result(func(&ComplexScalar(z), mode));
        let rhs = complex_from_result(func(&ComplexScalar(z.conj()), mode));
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: conjugation mismatch for {z:?} -> {left:?} vs {right:?}"
            );
        }
    }

    if vectorize {
        let vec_result = complex_vec_from_result(func(&ComplexVec(vec![z, z.conj()]), mode));
        if let Some(values) = vec_result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            if values[0].is_finite() && values[1].is_finite() && z.is_finite() && z.im != 0.0 {
                assert!(
                    approx_eq_complex(values[1], values[0].conj()),
                    "{name}: vector conjugation mismatch for {z:?}"
                );
            }
        }
    }

    if z.im == 0.0 {
        let real = z.re;
        let real_result = func(&RealScalar(real), mode);
        let complex_result =
            complex_from_result(func(&ComplexScalar(Complex64::from_real(real)), mode));
        if let (Ok(RealScalar(real_value)), Some(complex_value)) = (real_result, complex_result)
            && complex_value.is_finite()
            && real_value.is_finite()
        {
            assert!(
                approx_eq_scalar(complex_value.re, real_value),
                "{name}: real-axis reduction mismatch for {real}"
            );
            assert!(
                complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                "{name}: expected near-real output on real axis for {real}, got {complex_value:?}"
            );
        }
    }
}

fn check_gamma_family(z: Complex64, mode: RuntimeMode, vectorize: bool) {
    for (name, func) in [
        ("gamma", gamma as UnarySpecialFn),
        ("gammaln", gammaln as UnarySpecialFn),
        ("rgamma", rgamma as UnarySpecialFn),
        ("digamma", digamma as UnarySpecialFn),
    ] {
        check_unary(name, func, z, mode, vectorize);
    }

    let gamma_value = complex_from_result(gamma(&ComplexScalar(z), mode));
    let gammaln_value = complex_from_result(gammaln(&ComplexScalar(z), mode));
    if let (Some(gamma_z), Some(gammaln_z)) = (gamma_value, gammaln_value)
        && gamma_z.is_finite()
        && gammaln_z.is_finite()
    {
        let expected = gammaln_z.exp();
        assert!(
            approx_eq_complex(gamma_z, expected),
            "gamma/gammaln mismatch for {z:?}: {gamma_z:?} vs {expected:?}"
        );
    }

    let rgamma_value = complex_from_result(rgamma(&ComplexScalar(z), mode));
    if let (Some(gamma_z), Some(rgamma_z)) = (gamma_value, rgamma_value)
        && gamma_z.is_finite()
        && rgamma_z.is_finite()
        && gamma_z.abs() > 1.0e-8
    {
        let product = gamma_z * rgamma_z;
        assert!(
            approx_eq_complex(product, Complex64::from_real(1.0)),
            "gamma*rgamma should equal 1 for {z:?}, got {product:?}"
        );
    }

    if z.im == 0.0
        && z.re <= 0.0
        && z.re.fract() == 0.0
        && let Some(rgamma_z) = rgamma_value
        && rgamma_z.is_finite()
    {
        assert!(
            approx_eq_complex(rgamma_z, Complex64::from_real(0.0)),
            "rgamma should vanish at negative-integer poles for {z:?}, got {rgamma_z:?}"
        );
    }
}

fuzz_target!(|input: ComplexBranchInput| {
    let z = Complex64::new(input.re.raw(), input.im.raw());
    let mode = mode_from_flag(input.hardened);

    for (name, func) in [
        ("lambertw", lambertw as UnarySpecialFn),
        ("wrightomega", wrightomega as UnarySpecialFn),
        ("exp1", exp1 as UnarySpecialFn),
        ("expi", expi as UnarySpecialFn),
        ("erfinv", erfinv as UnarySpecialFn),
        ("erfcinv", erfcinv as UnarySpecialFn),
    ] {
        check_unary(name, func, z, mode, input.vectorize);
    }

    check_gamma_family(z, mode, input.vectorize);
});
