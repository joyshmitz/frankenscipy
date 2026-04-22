#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    i0, i1, iv, j0, j1, jn, jv, kv, y0, y1, yn, yv,
};
use libfuzzer_sys::fuzz_target;

type UnarySpecialFn = fn(&SpecialTensor, RuntimeMode) -> SpecialResult;
type BinarySpecialFn = fn(&SpecialTensor, &SpecialTensor, RuntimeMode) -> SpecialResult;

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
        }
    }

    fn bounded(self) -> f64 {
        let raw = self.raw();
        if raw.is_finite() {
            raw.clamp(-16.0, 16.0)
        } else if raw.is_sign_positive() {
            16.0
        } else if raw.is_sign_negative() {
            -16.0
        } else {
            0.0
        }
    }

    fn positive_real(self) -> f64 {
        self.bounded().abs().clamp(1.0e-6, 16.0)
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct BesselBaseInput {
    order: EdgeF64,
    re: EdgeF64,
    im: EdgeF64,
    hardened: bool,
    vectorize: bool,
    mismatch_lengths: bool,
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

fn real_from_result(result: SpecialResult) -> Option<f64> {
    match result.ok()? {
        RealScalar(value) => Some(value),
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

fn complex_vec_from_result(result: SpecialResult) -> Option<Vec<Complex64>> {
    match result.ok()? {
        RealVec(values) => Some(values.into_iter().map(Complex64::from_real).collect()),
        ComplexVec(values) => Some(values),
        _ => None,
    }
}

fn check_complex_base(
    name: &str,
    func: BinarySpecialFn,
    v: f64,
    z: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
    mismatch_lengths: bool,
) {
    let v_tensor = RealScalar(v);
    let _ = func(&v_tensor, &ComplexScalar(z), mode);

    if z.is_finite() && z.im != 0.0 {
        let lhs = complex_from_result(func(&v_tensor, &ComplexScalar(z), mode));
        let rhs = complex_from_result(func(&v_tensor, &ComplexScalar(z.conj()), mode));
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: conjugation mismatch for v={v:?}, z={z:?}"
            );
        }
    }

    if vectorize {
        let vec_result =
            complex_vec_from_result(func(&v_tensor, &ComplexVec(vec![z, z.conj()]), mode));
        if let Some(values) = vec_result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            if values[0].is_finite() && values[1].is_finite() && z.is_finite() && z.im != 0.0 {
                assert!(
                    approx_eq_complex(values[1], values[0].conj()),
                    "{name}: vector conjugation mismatch for v={v:?}, z={z:?}"
                );
            }
        }
    }

    if mismatch_lengths {
        assert!(
            func(&RealVec(vec![v, v + 1.0]), &ComplexVec(vec![z]), mode).is_err(),
            "{name}: mismatched vector lengths should fail closed"
        );
    }

    if z.im == 0.0 && z.re > 0.0 {
        let real_result = real_from_result(func(&v_tensor, &RealScalar(z.re), mode));
        let complex_result = complex_from_result(func(
            &v_tensor,
            &ComplexScalar(Complex64::from_real(z.re)),
            mode,
        ));
        if let (Some(real_value), Some(complex_value)) = (real_result, complex_result)
            && real_value.is_finite()
            && complex_value.is_finite()
        {
            assert!(
                approx_eq_scalar(real_value, complex_value.re),
                "{name}: real-axis reduction mismatch for v={v:?}, z={z:?}"
            );
            assert!(
                complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                "{name}: expected near-real output on positive real axis for v={v:?}, z={z:?}"
            );
        }
    }
}

fn check_wrapper_matches_base(
    name: &str,
    wrapper: UnarySpecialFn,
    base: BinarySpecialFn,
    order: f64,
    x: f64,
    mode: RuntimeMode,
) {
    let wrapper_value = real_from_result(wrapper(&RealScalar(x), mode));
    let base_real = real_from_result(base(&RealScalar(order), &RealScalar(x), mode));
    let base_complex = complex_from_result(base(
        &RealScalar(order),
        &ComplexScalar(Complex64::from_real(x)),
        mode,
    ));
    if let (Some(wrapped), Some(real_base), Some(complex_base)) =
        (wrapper_value, base_real, base_complex)
        && wrapped.is_finite()
        && real_base.is_finite()
        && complex_base.is_finite()
    {
        assert!(
            approx_eq_scalar(wrapped, real_base),
            "{name}: real wrapper mismatch at x={x}"
        );
        assert!(
            approx_eq_scalar(wrapped, complex_base.re),
            "{name}: complex reduction mismatch at x={x}"
        );
        assert!(
            complex_base.im.abs() <= ABS_TOL + REL_TOL * complex_base.re.abs(),
            "{name}: expected near-real complex reduction at x={x}"
        );
    }
}

fn check_integer_wrappers(order: usize, x: f64, mode: RuntimeMode) {
    check_wrapper_matches_base(
        "j0",
        j0 as UnarySpecialFn,
        jv as BinarySpecialFn,
        0.0,
        x,
        mode,
    );
    check_wrapper_matches_base(
        "j1",
        j1 as UnarySpecialFn,
        jv as BinarySpecialFn,
        1.0,
        x,
        mode,
    );
    check_wrapper_matches_base(
        "y0",
        y0 as UnarySpecialFn,
        yv as BinarySpecialFn,
        0.0,
        x,
        mode,
    );
    check_wrapper_matches_base(
        "y1",
        y1 as UnarySpecialFn,
        yv as BinarySpecialFn,
        1.0,
        x,
        mode,
    );
    check_wrapper_matches_base(
        "i0",
        i0 as UnarySpecialFn,
        iv as BinarySpecialFn,
        0.0,
        x,
        mode,
    );
    check_wrapper_matches_base(
        "i1",
        i1 as UnarySpecialFn,
        iv as BinarySpecialFn,
        1.0,
        x,
        mode,
    );

    let order_tensor = RealScalar(order as f64);
    let jn_wrapper = real_from_result(jn(&order_tensor, &RealScalar(x), mode));
    let yn_wrapper = real_from_result(yn(&order_tensor, &RealScalar(x), mode));
    let jv_base = real_from_result(jv(&order_tensor, &RealScalar(x), mode));
    let yv_base = real_from_result(yv(&order_tensor, &RealScalar(x), mode));
    if let (Some(jn_value), Some(jv_value)) = (jn_wrapper, jv_base)
        && jn_value.is_finite()
        && jv_value.is_finite()
    {
        assert!(
            approx_eq_scalar(jn_value, jv_value),
            "jn/jv mismatch for n={order}, x={x}"
        );
    }
    if let (Some(yn_value), Some(yv_value)) = (yn_wrapper, yv_base)
        && yn_value.is_finite()
        && yv_value.is_finite()
    {
        assert!(
            approx_eq_scalar(yn_value, yv_value),
            "yn/yv mismatch for n={order}, x={x}"
        );
    }
}

fuzz_target!(|input: BesselBaseInput| {
    let mode = mode_from_flag(input.hardened);
    let v = input.order.bounded();
    let z = Complex64::new(input.re.raw(), input.im.raw());
    let positive_real = input.re.positive_real();
    let integer_order = input.order.bounded().abs().round() as usize;

    for (name, func) in [
        ("jv", jv as BinarySpecialFn),
        ("yv", yv as BinarySpecialFn),
        ("iv", iv as BinarySpecialFn),
        ("kv", kv as BinarySpecialFn),
    ] {
        check_complex_base(
            name,
            func,
            v,
            z,
            mode,
            input.vectorize,
            input.mismatch_lengths,
        );
    }

    check_integer_wrappers(integer_order, positive_real, mode);
});
