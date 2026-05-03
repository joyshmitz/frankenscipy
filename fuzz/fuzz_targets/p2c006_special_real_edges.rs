#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    hyp1f1, hyp2f1, lambertw,
};
use libfuzzer_sys::fuzz_target;

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
    Two,
    NegTwo,
    PosInf,
    NegInf,
    Nan,
    LambertBranch,
    LambertBranchAbove,
    LambertBranchBelow,
    HyperPole1,
    HyperPole2,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e6, 1.0e6),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::Two => 2.0,
            Self::NegTwo => -2.0,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
            Self::LambertBranch => -1.0 / std::f64::consts::E,
            Self::LambertBranchAbove => -1.0 / std::f64::consts::E + 1.0e-12,
            Self::LambertBranchBelow => -1.0 / std::f64::consts::E - 1.0e-12,
            Self::HyperPole1 => -1.0,
            Self::HyperPole2 => -2.0,
        }
    }

    fn hyper(self) -> f64 {
        let value = self.raw();
        if value.is_finite() {
            value.clamp(-24.0, 24.0)
        } else {
            value
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct SpecialEdgeInput {
    x: EdgeF64,
    a: EdgeF64,
    b: EdgeF64,
    c: EdgeF64,
    z: EdgeF64,
    alt: EdgeF64,
    hardened: bool,
    vectorize: bool,
    complex_probe: bool,
}

fn mode_from_flag(hardened: bool) -> RuntimeMode {
    if hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    }
}

fn approx_eq(lhs: f64, rhs: f64) -> bool {
    if !(lhs.is_finite() && rhs.is_finite()) {
        return false;
    }
    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    (lhs - rhs).abs() <= ABS_TOL + REL_TOL * scale
}

fn scalar_from_result(result: SpecialResult) -> Option<f64> {
    match result.ok()? {
        RealScalar(value) => Some(value),
        ComplexScalar(value) if value.im == 0.0 => Some(value.re),
        _ => None,
    }
}

fn complex_from_result(result: SpecialResult) -> Option<Complex64> {
    match result.ok()? {
        RealScalar(value) => Some(Complex64::from_real(value)),
        ComplexScalar(value) => Some(value),
        _ => None,
    }
}

fn check_vector_len(name: &str, result: SpecialResult, expected_len: usize) {
    match result {
        Ok(RealVec(values)) => assert_eq!(values.len(), expected_len, "{name}: RealVec length"),
        Ok(ComplexVec(values)) => {
            assert_eq!(values.len(), expected_len, "{name}: ComplexVec length")
        }
        _ => {}
    }
}

fn check_lambertw(input: &SpecialEdgeInput, mode: RuntimeMode) {
    let x = input.x.raw();
    let real_result = lambertw(&RealScalar(x), mode);
    let _ = lambertw(&ComplexScalar(Complex64::from_real(x)), mode);

    if x == f64::INFINITY
        && let Some(value) = scalar_from_result(real_result.clone())
    {
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "lambertw(+inf) must return +inf, got {value}"
        );
    }

    let min_x = -1.0 / std::f64::consts::E;
    if x.is_finite()
        && (min_x..=32.0).contains(&x)
        && let Some(w) = scalar_from_result(real_result)
        && w.is_finite()
    {
        let reconstructed = w * w.exp();
        assert!(
            approx_eq(reconstructed, x),
            "lambertw identity failed: x={x}, w={w}, w*exp(w)={reconstructed}"
        );
    }

    if input.vectorize {
        check_vector_len(
            "lambertw",
            lambertw(&RealVec(vec![x, input.alt.raw(), f64::INFINITY]), mode),
            3,
        );
    }
}

fn check_hyper(input: &SpecialEdgeInput, mode: RuntimeMode) {
    let a = input.a.hyper();
    let b = input.b.hyper();
    let c = input.c.hyper();
    let z = input.z.hyper();

    let _ = hyp1f1(&RealScalar(a), &RealScalar(b), &RealScalar(z), mode);
    let _ = hyp2f1(
        &RealScalar(a),
        &RealScalar(b),
        &RealScalar(c),
        &RealScalar(z),
        mode,
    );

    if let Some(value) = scalar_from_result(hyp1f1(
        &RealScalar(a),
        &RealScalar(b),
        &RealScalar(0.0),
        mode,
    )) && value.is_finite()
    {
        assert!(approx_eq(value, 1.0), "hyp1f1(a,b,0) got {value}");
    }

    if let Some(value) = scalar_from_result(hyp2f1(
        &RealScalar(a),
        &RealScalar(b),
        &RealScalar(c),
        &RealScalar(0.0),
        mode,
    )) && value.is_finite()
    {
        assert!(approx_eq(value, 1.0), "hyp2f1(a,b,c,0) got {value}");
    }

    if input.complex_probe {
        let zc = Complex64::new(z, input.alt.hyper());
        let _ = hyp1f1(
            &ComplexScalar(Complex64::from_real(a)),
            &ComplexScalar(Complex64::from_real(b)),
            &ComplexScalar(zc),
            mode,
        );
        let _ = hyp2f1(
            &ComplexScalar(Complex64::from_real(a)),
            &ComplexScalar(Complex64::from_real(b)),
            &ComplexScalar(Complex64::from_real(c)),
            &ComplexScalar(zc),
            mode,
        );

        let zero = Complex64::from_real(0.0);
        if let Some(value) = complex_from_result(hyp2f1(
            &ComplexScalar(Complex64::from_real(a)),
            &ComplexScalar(Complex64::from_real(b)),
            &ComplexScalar(Complex64::from_real(c)),
            &ComplexScalar(zero),
            mode,
        )) && value.is_finite()
        {
            assert!(
                approx_eq(value.re, 1.0) && approx_eq(value.im, 0.0),
                "complex hyp2f1(a,b,c,0) got {value:?}"
            );
        }
    }

    if input.vectorize {
        check_vector_len(
            "hyp1f1",
            hyp1f1(
                &RealScalar(a),
                &RealScalar(b),
                &RealVec(vec![z, 0.0, input.alt.hyper()]),
                mode,
            ),
            3,
        );
        check_vector_len(
            "hyp2f1",
            hyp2f1(
                &RealScalar(a),
                &RealScalar(b),
                &RealScalar(c),
                &RealVec(vec![z, 0.0, input.alt.hyper()]),
                mode,
            ),
            3,
        );
    }
}

fuzz_target!(|input: SpecialEdgeInput| {
    let mode = mode_from_flag(input.hardened);
    check_lambertw(&input, mode);
    check_hyper(&input, mode);
});
