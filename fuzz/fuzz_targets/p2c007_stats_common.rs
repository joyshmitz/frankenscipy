#![allow(dead_code)]

use arbitrary::Arbitrary;

pub const CDF_ABS_TOL: f64 = 1.0e-4;
pub const CDF_REL_TOL: f64 = 2.0e-4;

#[derive(Clone, Copy, Debug, Arbitrary)]
pub enum EdgeF64 {
    Finite(f64),
    Zero,
    NegZero,
    One,
    NegOne,
    Tiny,
    NegTiny,
    Two,
    NegTwo,
    PosInf,
    NegInf,
    Nan,
}

impl EdgeF64 {
    pub fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e6, 1.0e6),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::Two => 2.0,
            Self::NegTwo => -2.0,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
        }
    }

    pub fn finite(self, lo: f64, hi: f64, default: f64) -> f64 {
        let raw = self.raw();
        if raw.is_finite() {
            raw.clamp(lo, hi)
        } else {
            default
        }
    }

    pub fn positive(self, lo: f64, hi: f64, default: f64) -> f64 {
        let raw = self.raw();
        if raw.is_finite() {
            raw.abs().clamp(lo, hi)
        } else {
            default
        }
    }

    pub fn probability(self) -> f64 {
        let raw = self.raw();
        let unit = if raw.is_finite() {
            raw.abs().fract()
        } else {
            0.5
        };
        unit.clamp(1.0e-9, 1.0 - 1.0e-9)
    }
}

pub fn all_finite_or_all_nan(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite()) || values.iter().all(|value| value.is_nan())
}

pub fn approx_eq_prob(lhs: f64, rhs: f64) -> bool {
    if !(lhs.is_finite() && rhs.is_finite()) {
        return false;
    }
    let scale = lhs.abs().max(rhs.abs());
    (lhs - rhs).abs() <= CDF_ABS_TOL + CDF_REL_TOL * scale
}
