#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    iv, ivp, jv, jvp, kv, kvp, yv, yvp,
};
use libfuzzer_sys::fuzz_target;

type BaseFn = fn(&SpecialTensor, &SpecialTensor, RuntimeMode) -> SpecialResult;
type DerivativeFn = fn(&SpecialTensor, &SpecialTensor, usize, RuntimeMode) -> SpecialResult;

const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-6;
const WIDE_EDGE_LIMIT: f64 = 1.0e300;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum EdgeF64 {
    Finite(f64),
    Zero,
    NegZero,
    One,
    NegOne,
    Tiny,
    NegTiny,
    Underflow,
    Huge,
    NegHuge,
    Overflow,
    PosInf,
    NegInf,
    Nan,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => {
                value.clamp(-WIDE_EDGE_LIMIT, WIDE_EDGE_LIMIT)
            }
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::Underflow => f64::MIN_POSITIVE * 0.5,
            Self::Huge => 1.0e300,
            Self::NegHuge => -1.0e300,
            Self::Overflow => f64::MAX,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct BesselDerivativeInput {
    order: EdgeF64,
    re: EdgeF64,
    im: EdgeF64,
    derivative_order: u8,
    hardened: bool,
    vectorize: bool,
}

#[derive(Clone, Copy)]
struct DerivativeCase {
    v: f64,
    z: Complex64,
    derivative_order: usize,
    mode: RuntimeMode,
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

fn check_bessel_derivative(
    name: &str,
    base: BaseFn,
    derivative: DerivativeFn,
    case: DerivativeCase,
) {
    let v_tensor = RealScalar(case.v);
    let _ = derivative(
        &v_tensor,
        &ComplexScalar(case.z),
        case.derivative_order,
        case.mode,
    );

    if case.derivative_order == 0 {
        let derived = complex_from_result(derivative(
            &v_tensor,
            &ComplexScalar(case.z),
            case.derivative_order,
            case.mode,
        ));
        let base_value = complex_from_result(base(&v_tensor, &ComplexScalar(case.z), case.mode));
        if let (Some(lhs), Some(rhs)) = (derived, base_value)
            && lhs.is_finite()
            && rhs.is_finite()
        {
            assert!(
                approx_eq_complex(lhs, rhs),
                "{name}: zero-order derivative should match base function for v={:?}, z={:?}",
                case.v,
                case.z
            );
        }
    }

    if case.z.is_finite() && case.z.im != 0.0 {
        let lhs = complex_from_result(derivative(
            &v_tensor,
            &ComplexScalar(case.z),
            case.derivative_order,
            case.mode,
        ));
        let rhs = complex_from_result(derivative(
            &v_tensor,
            &ComplexScalar(case.z.conj()),
            case.derivative_order,
            case.mode,
        ));
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: conjugation mismatch for v={:?}, z={:?}",
                case.v,
                case.z
            );
        }
    }

    if case.vectorize {
        let vec_result = complex_vec_from_result(derivative(
            &v_tensor,
            &ComplexVec(vec![case.z, case.z.conj()]),
            case.derivative_order,
            case.mode,
        ));
        if let Some(values) = vec_result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            if values[0].is_finite()
                && values[1].is_finite()
                && case.z.is_finite()
                && case.z.im != 0.0
            {
                assert!(
                    approx_eq_complex(values[1], values[0].conj()),
                    "{name}: vector conjugation mismatch for v={:?}, z={:?}",
                    case.v,
                    case.z
                );
            }
        }
    }

    if case.z.im == 0.0 && case.z.re.is_finite() && case.z.re > 0.0 {
        let real_result = derivative(
            &v_tensor,
            &RealScalar(case.z.re),
            case.derivative_order,
            case.mode,
        );
        let complex_result = complex_from_result(derivative(
            &v_tensor,
            &ComplexScalar(Complex64::from_real(case.z.re)),
            case.derivative_order,
            case.mode,
        ));
        if let (Ok(RealScalar(real_value)), Some(complex_value)) = (real_result, complex_result)
            && real_value.is_finite()
            && complex_value.is_finite()
        {
            assert!(
                approx_eq_scalar(real_value, complex_value.re),
                "{name}: real-axis reduction mismatch for v={:?}, z={:?}",
                case.v,
                case.z
            );
            assert!(
                complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                "{name}: expected near-real derivative on positive real axis for v={:?}, z={:?}",
                case.v,
                case.z
            );
        }
    }
}

fuzz_target!(|input: BesselDerivativeInput| {
    let v = input.order.raw();
    let z = Complex64::new(input.re.raw(), input.im.raw());
    let case = DerivativeCase {
        v,
        z,
        derivative_order: usize::from(input.derivative_order % 5),
        mode: mode_from_flag(input.hardened),
        vectorize: input.vectorize,
    };

    for (name, base, derivative) in [
        ("jvp", jv as BaseFn, jvp as DerivativeFn),
        ("yvp", yv as BaseFn, yvp as DerivativeFn),
        ("ivp", iv as BaseFn, ivp as DerivativeFn),
        ("kvp", kv as BaseFn, kvp as DerivativeFn),
    ] {
        check_bessel_derivative(name, base, derivative, case);
    }
});
