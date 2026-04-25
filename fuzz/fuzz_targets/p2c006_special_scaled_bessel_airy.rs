#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    airy, iv, jv, kv, yv,
};
use libfuzzer_sys::fuzz_target;

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
    Fifty,
    NegFifty,
    UnderflowBoundaryLo,
    UnderflowBoundary,
    UnderflowBoundaryHi,
    PosInf,
    NegInf,
    Nan,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-720.0, 720.0),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::Fifty => 50.0,
            Self::NegFifty => -50.0,
            Self::UnderflowBoundaryLo => 699.0,
            Self::UnderflowBoundary => 700.0,
            Self::UnderflowBoundaryHi => 701.0,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
        }
    }

    fn finite(self) -> f64 {
        let raw = self.raw();
        if raw.is_finite() {
            raw
        } else if raw.is_sign_positive() {
            720.0
        } else if raw.is_sign_negative() {
            -720.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ScaledBesselCase {
    Jve,
    Yve,
    Ive,
    Kve,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct ScaledSpecialInput {
    order: EdgeF64,
    re: EdgeF64,
    im: EdgeF64,
    function: ScaledBesselCase,
    hardened: bool,
    vectorize: bool,
}

#[derive(Clone, Copy, Debug)]
enum BesselScale {
    Imaginary,
    Real,
    Argument,
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

fn complex_sqrt(z: Complex64) -> Complex64 {
    if z.im == 0.0 {
        if z.re >= 0.0 {
            return Complex64::new(z.re.sqrt(), 0.0);
        }
        return Complex64::new(0.0, (-z.re).sqrt().copysign(z.im));
    }

    let radius = z.abs();
    let re = ((radius + z.re) * 0.5).sqrt();
    let im = ((radius - z.re) * 0.5).sqrt().copysign(z.im);
    Complex64::new(re, im)
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

fn complex_scalar_from_tensor(tensor: &SpecialTensor) -> Option<Complex64> {
    match tensor {
        RealScalar(value) => Some(Complex64::from_real(*value)),
        ComplexScalar(value) => Some(*value),
        _ => None,
    }
}

fn complex_vec_from_tensor(tensor: &SpecialTensor) -> Option<Vec<Complex64>> {
    match tensor {
        RealVec(values) => Some(values.iter().copied().map(Complex64::from_real).collect()),
        ComplexVec(values) => Some(values.clone()),
        _ => None,
    }
}

fn airy_scalar_bundle(
    result: Result<Vec<SpecialTensor>, fsci_special::SpecialError>,
) -> Option<[Complex64; 4]> {
    let values = result.ok()?;
    if values.len() != 4 {
        return None;
    }
    Some([
        complex_scalar_from_tensor(&values[0])?,
        complex_scalar_from_tensor(&values[1])?,
        complex_scalar_from_tensor(&values[2])?,
        complex_scalar_from_tensor(&values[3])?,
    ])
}

fn airy_vec_bundle(
    result: Result<Vec<SpecialTensor>, fsci_special::SpecialError>,
) -> Option<[Vec<Complex64>; 4]> {
    let values = result.ok()?;
    if values.len() != 4 {
        return None;
    }
    Some([
        complex_vec_from_tensor(&values[0])?,
        complex_vec_from_tensor(&values[1])?,
        complex_vec_from_tensor(&values[2])?,
        complex_vec_from_tensor(&values[3])?,
    ])
}

fn scaled_bessel_factor(z: Complex64, scale: BesselScale) -> Complex64 {
    match scale {
        BesselScale::Imaginary => Complex64::from_real((-z.im.abs()).exp()),
        BesselScale::Real => Complex64::from_real((-z.re.abs()).exp()),
        BesselScale::Argument => z.exp(),
    }
}

fn scaled_bessel_value(
    func: BinarySpecialFn,
    scale: BesselScale,
    v: f64,
    z: Complex64,
    mode: RuntimeMode,
) -> Option<Complex64> {
    let base = complex_from_result(func(&RealScalar(v), &ComplexScalar(z), mode))?;
    Some(base * scaled_bessel_factor(z, scale))
}

fn check_scaled_bessel(
    name: &str,
    func: BinarySpecialFn,
    scale: BesselScale,
    v: f64,
    z: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
) {
    let scaled = scaled_bessel_value(func, scale, v, z, mode);

    if z.im == 0.0 {
        let real = complex_from_result(func(&RealScalar(v), &RealScalar(z.re), mode));
        if let (Some(real_base), Some(complex_scaled)) = (real, scaled)
            && real_base.is_finite()
            && complex_scaled.is_finite()
        {
            let expected = real_base * scaled_bessel_factor(z, scale);
            assert!(
                approx_eq_complex(complex_scaled, expected),
                "{name}: real-axis scaled relation mismatch for v={v:?}, z={z:?}"
            );
        }
    }

    if z.is_finite() && z.im != 0.0 {
        let lhs = scaled;
        let rhs = scaled_bessel_value(func, scale, v, z.conj(), mode);
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: scaled conjugation mismatch for v={v:?}, z={z:?}"
            );
        }
    }

    if vectorize {
        let result =
            complex_vec_from_result(func(&RealScalar(v), &ComplexVec(vec![z, z.conj()]), mode));
        if let Some(values) = result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            let first_scaled = values[0] * scaled_bessel_factor(z, scale);
            let second_scaled = values[1] * scaled_bessel_factor(z.conj(), scale);
            if first_scaled.is_finite() && second_scaled.is_finite() && z.is_finite() && z.im != 0.0
            {
                assert!(
                    approx_eq_complex(second_scaled, first_scaled.conj()),
                    "{name}: scaled vector conjugation mismatch for v={v:?}, z={z:?}"
                );
            }
        }
    }
}

fn scaled_airy_bundle(bundle: [Complex64; 4], z: Complex64) -> [Complex64; 4] {
    let eta = z * complex_sqrt(z) * (2.0 / 3.0);
    let ai_scale = eta.exp();
    let bi_scale = Complex64::from_real((-eta.re.abs()).exp());
    [
        bundle[0] * ai_scale,
        bundle[1] * ai_scale,
        bundle[2] * bi_scale,
        bundle[3] * bi_scale,
    ]
}

fn check_scaled_airy(z: Complex64, mode: RuntimeMode, vectorize: bool) {
    let scalar = airy_scalar_bundle(airy(&ComplexScalar(z), mode));
    if let Some(bundle) = scalar {
        let scaled = scaled_airy_bundle(bundle, z);
        if z.is_finite() && z.im != 0.0 {
            let conj_bundle = airy_scalar_bundle(airy(&ComplexScalar(z.conj()), mode));
            if let Some(conj_values) = conj_bundle {
                let scaled_conj = scaled_airy_bundle(conj_values, z.conj());
                for (index, (left, right)) in scaled.iter().zip(scaled_conj.iter()).enumerate() {
                    if left.is_finite() && right.is_finite() {
                        assert!(
                            approx_eq_complex(*right, left.conj()),
                            "airye: scaled conjugation mismatch in component {index} for {z:?}"
                        );
                    }
                }
            }
        }

        if z == Complex64::from_real(0.0) {
            for (index, (unscaled, scaled_value)) in bundle.iter().zip(scaled.iter()).enumerate() {
                assert!(
                    approx_eq_complex(*unscaled, *scaled_value),
                    "airye: zero scaling should preserve component {index}"
                );
            }
        }
    }

    if vectorize {
        let result = airy_vec_bundle(airy(&ComplexVec(vec![z, z.conj()]), mode));
        if let Some(bundle) = result {
            for (index, component) in bundle.iter().enumerate() {
                assert_eq!(
                    component.len(),
                    2,
                    "airye component {index} length mismatch"
                );
            }
        }
    }

    if z.im == 0.0 {
        let real_bundle = airy_scalar_bundle(airy(&RealScalar(z.re), mode));
        let complex_bundle = airy_scalar_bundle(airy(&ComplexScalar(z), mode));
        if let (Some(real_values), Some(complex_values)) = (real_bundle, complex_bundle) {
            let real_scaled = scaled_airy_bundle(real_values, z);
            let complex_scaled = scaled_airy_bundle(complex_values, z);
            for (index, (real_value, complex_value)) in
                real_scaled.iter().zip(complex_scaled.iter()).enumerate()
            {
                if real_value.is_finite() && complex_value.is_finite() {
                    assert!(
                        approx_eq_complex(*real_value, *complex_value),
                        "airye: real-axis reduction mismatch in component {index} for {z:?}"
                    );
                }
            }
        }
    }
}

fuzz_target!(|input: ScaledSpecialInput| {
    let mode = mode_from_flag(input.hardened);
    let v = input.order.finite().clamp(-32.0, 32.0);
    let z = Complex64::new(input.re.finite(), input.im.finite());

    match input.function {
        ScaledBesselCase::Jve => check_scaled_bessel(
            "jve",
            jv as BinarySpecialFn,
            BesselScale::Imaginary,
            v,
            z,
            mode,
            input.vectorize,
        ),
        ScaledBesselCase::Yve => check_scaled_bessel(
            "yve",
            yv as BinarySpecialFn,
            BesselScale::Imaginary,
            v,
            z,
            mode,
            input.vectorize,
        ),
        ScaledBesselCase::Ive => check_scaled_bessel(
            "ive",
            iv as BinarySpecialFn,
            BesselScale::Real,
            v,
            z,
            mode,
            input.vectorize,
        ),
        ScaledBesselCase::Kve => check_scaled_bessel(
            "kve",
            kv as BinarySpecialFn,
            BesselScale::Argument,
            v,
            z,
            mode,
            input.vectorize,
        ),
    }

    check_scaled_airy(z, mode, input.vectorize);
});
