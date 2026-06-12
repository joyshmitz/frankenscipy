#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::gamma;
use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const BETA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "beta",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "use gamma/gammaln composition in stable space",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter regime uses logspace stabilization",
            },
        ],
        notes: "Symmetry beta(a,b)=beta(b,a) must hold in strict mode and hardened mode.",
    },
    DispatchPlan {
        function: "betaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "direct logspace composition",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "a+b sufficiently large",
            },
        ],
        notes: "Primary path for underflow-prone beta regions.",
    },
    DispatchPlan {
        function: "betainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x in lower-tail region",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x in upper-tail region",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifts for stability",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
    DispatchPlan {
        function: "betaincc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "evaluate the complementary tail as I_(1-x)(b, a)",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "delegates to the stable betainc tail path after swapping parameters",
            },
        ],
        notes: "Strict mode preserves SciPy endpoint behavior at x=0 and x=1.",
    },
];

const DISTRIBUTION_INVERSE_ITERS: usize = 160;
const DISTRIBUTION_INVERSE_UPPER_SENTINEL: f64 = 1.0e100;

pub fn beta(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    beta_dispatch(a, b, mode)
}

pub fn betaln(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    betaln_dispatch(a, b, mode)
}

fn beta_dispatch(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if matches!(
        (a, b),
        (
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_),
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_)
        )
    ) {
        map_real_binary("beta", a, b, mode, |av, bv| beta_scalar(av, bv, mode))
    } else {
        map_complex_binary("beta", a, b, mode, complex_beta_scalar)
    }
}

fn betaln_dispatch(a: &SpecialTensor, b: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if matches!(
        (a, b),
        (
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_),
            SpecialTensor::RealScalar(_) | SpecialTensor::RealVec(_)
        )
    ) {
        map_real_binary("betaln", a, b, mode, |av, bv| betaln_scalar(av, bv, mode))
    } else {
        map_complex_binary("betaln", a, b, mode, complex_betaln_scalar)
    }
}

fn map_complex_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(Complex64, Complex64) -> Complex64 + Sync,
{
    match (a, b) {
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            Ok(SpecialTensor::ComplexScalar(kernel(*lhs, *rhs)))
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::ComplexVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, rhs[i]))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::ComplexVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs[i])))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(Complex64::from_real(*lhs), *rhs)),
        ),
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealScalar(rhs)) => Ok(
            SpecialTensor::ComplexScalar(kernel(*lhs, Complex64::from_real(*rhs))),
        ),
        (SpecialTensor::RealVec(lhs), SpecialTensor::ComplexScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices(lhs.len(), |i| Ok(kernel(Complex64::from_real(lhs[i]), rhs)))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(lhs), SpecialTensor::RealVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, Complex64::from_real(rhs[i]))))
                .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::ComplexVec(rhs)) => {
            let lhs = Complex64::from_real(*lhs);
            par_map_indices(rhs.len(), |i| Ok(kernel(lhs, rhs[i]))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::RealScalar(rhs)) => {
            let rhs = Complex64::from_real(*rhs);
            par_map_indices(lhs.len(), |i| Ok(kernel(lhs[i], rhs))).map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::ComplexVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(lhs.len(), |i| {
                Ok(kernel(Complex64::from_real(lhs[i]), rhs[i]))
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(lhs), SpecialTensor::RealVec(rhs)) => {
            if lhs.len() != rhs.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(lhs.len(), |i| {
                Ok(kernel(lhs[i], Complex64::from_real(rhs[i])))
            })
            .map(SpecialTensor::ComplexVec)
        }
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported tensor combination",
        }),
    }
}

pub fn betainc(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    betainc_dispatch(a, b, x, mode)
}

fn betainc_dispatch(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match (a, b, x) {
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(xv),
        ) => betainc_scalar(*av, *bv, *xv, mode).map(SpecialTensor::RealScalar),
        (
            SpecialTensor::ComplexScalar(av),
            SpecialTensor::ComplexScalar(bv),
            SpecialTensor::ComplexScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            *av, *bv, *xv,
        ))),
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::ComplexScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            Complex64::from_real(*av),
            Complex64::from_real(*bv),
            *xv,
        ))),
        (
            SpecialTensor::ComplexScalar(av),
            SpecialTensor::ComplexScalar(bv),
            SpecialTensor::RealScalar(xv),
        ) => Ok(SpecialTensor::ComplexScalar(complex_betainc_scalar(
            *av,
            *bv,
            Complex64::from_real(*xv),
        ))),
        _ => map_real_ternary("betainc", a, b, x, mode, |av, bv, xv| {
            betainc_scalar(av, bv, xv, mode)
        }),
    }
}

pub fn betaincc(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("betaincc", a, b, x, mode, |av, bv, xv| {
        betaincc_scalar(av, bv, xv, mode)
    })
}

pub fn betainccinv(
    a: &SpecialTensor,
    b: &SpecialTensor,
    y: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary("betainccinv", a, b, y, mode, |av, bv, yv| {
        Ok(betainccinv_scalar(av, bv, yv))
    })
}

/// Beta distribution CDF.
///
/// Matches `scipy.special.btdtr(a, b, x)`.
#[must_use]
pub fn btdtr(a: f64, b: f64, x: f64) -> f64 {
    betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Beta distribution survival function.
///
/// Returns P(X > x) = 1 - btdtr(a, b, x).
///
/// Matches `scipy.special.btdtrc(a, b, x)`.
#[must_use]
pub fn btdtrc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }
    // btdtrc(a, b, x) = 1 - betainc(a, b, x) = betainc(b, a, 1-x)
    betainc_scalar(b, a, 1.0 - x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse beta distribution CDF.
///
/// Matches `scipy.special.btdtri(a, b, y)`.
#[must_use]
pub fn btdtri(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }

    // btdtr(a, b, x) == betainc(a, b, x), so the inverse is exactly betaincinv.
    // The dedicated inverse carries a small-x asymptotic seed
    // (x ~ (y·a·B(a,b))^{1/a}) + relative-tolerance Newton, which resolves the
    // deep a<1 tail (e.g. y=1e-8) instead of stalling at the ~1e-16 floor that
    // the generic bisection bottomed out on. frankenscipy-8urrz.
    crate::convenience::betaincinv_scalar(a, b, y)
}

/// Inverse beta distribution CDF with respect to shape parameter `a`.
///
/// Returns `a` such that `betainc(a, b, x) = p`.
///
/// Matches `scipy.special.btdtria(p, b, x)`.
#[must_use]
pub fn btdtria(p: f64, b: f64, x: f64) -> f64 {
    if p.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) || x <= 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return f64::MIN_POSITIVE;
    }
    if x >= 1.0 || !x.is_finite() {
        return f64::NAN;
    }
    // I_x(a, b) = 1 - I_{1-x}(b, a); use the smaller tail when p is near 1.
    if p > 0.5 {
        return btdtrib(b, 1.0 - p, 1.0 - x);
    }

    invert_monotone_positive(|a| btdtr(a, b, x), p, false)
}

/// Inverse beta distribution CDF with respect to shape parameter `b`.
///
/// Returns `b` such that `betainc(a, b, x) = p`.
///
/// Matches `scipy.special.btdtrib(a, p, x)`.
#[must_use]
pub fn btdtrib(a: f64, p: f64, x: f64) -> f64 {
    if a.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || !(0.0..=1.0).contains(&p) || x <= 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::MIN_POSITIVE;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if x >= 1.0 || !x.is_finite() {
        return f64::NAN;
    }
    // I_x(a, b) = 1 - I_{1-x}(b, a); use the smaller tail when p is near 1.
    if p > 0.5 {
        return btdtria(1.0 - p, a, 1.0 - x);
    }

    invert_monotone_positive(|b| btdtr(a, b, x), p, true)
}

/// F-distribution CDF.
///
/// Matches `scipy.special.fdtr(dfn, dfd, x)`.
#[must_use]
pub fn fdtr(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = dfn * x / (dfn * x + dfd);
    btdtr(0.5 * dfn, 0.5 * dfd, z)
}

/// F-distribution survival function.
///
/// Returns P(X > x) = 1 - fdtr(dfn, dfd, x).
///
/// Matches `scipy.special.fdtrc(dfn, dfd, x)`.
#[must_use]
pub fn fdtrc(dfn: f64, dfd: f64, x: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    // fdtrc(dfn, dfd, x) = 1 - fdtr(dfn, dfd, x)
    // = 1 - btdtr(dfn/2, dfd/2, z) = btdtr(dfd/2, dfn/2, 1-z)
    let z = dfn * x / (dfn * x + dfd);
    btdtr(0.5 * dfd, 0.5 * dfn, 1.0 - z)
}

/// Inverse F-distribution CDF.
///
/// Matches `scipy.special.fdtri(dfn, dfd, y)`.
#[must_use]
pub fn fdtri(dfn: f64, dfd: f64, y: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }

    let z = btdtri(0.5 * dfn, 0.5 * dfd, y);
    dfd * z / (dfn * (1.0 - z))
}

/// Inverse F-distribution CDF with respect to denominator degrees of freedom.
///
/// Returns dfd such that P(F <= x) = p for an F distribution with dfn and dfd
/// degrees of freedom.
///
/// Matches `scipy.special.fdtridfd(dfn, p, x)`, including CDFlib sentinel
/// values for no-solution boundary cases.
#[must_use]
pub fn fdtridfd(dfn: f64, p: f64, x: f64) -> f64 {
    const LOWER_SENTINEL: f64 = 1.0e-100;
    const UPPER_SENTINEL: f64 = 1.0e100;

    if dfn.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || !(0.0..=1.0).contains(&p) || x < 0.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if p == 0.0 {
            5.0
        } else if p <= 0.5 {
            LOWER_SENTINEL
        } else {
            UPPER_SENTINEL
        };
    }
    if x.is_infinite() {
        return UPPER_SENTINEL;
    }
    if p == 0.0 {
        return LOWER_SENTINEL;
    }

    let upper_limit = gamma::chdtr(dfn, dfn * x);
    if !upper_limit.is_finite() {
        return f64::NAN;
    }
    if p >= upper_limit {
        return UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while fdtr(dfn, hi, x) < p {
        hi *= 2.0;
        if hi >= UPPER_SENTINEL {
            return UPPER_SENTINEL;
        }
    }

    let mut lo = 0.0;
    for _ in 0..240 {
        let mid = lo + (hi - lo) * 0.5;
        let value = fdtr(dfn, mid, x);
        if !value.is_finite() || value < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let dfd = lo + (hi - lo) * 0.5;
    if dfd == 0.0 { LOWER_SENTINEL } else { dfd }
}

/// Non-central F cumulative distribution function.
///
/// Matches `scipy.special.ncfdtr(dfn, dfd, nc, f)`: the CDF at `f` of a
/// non-central F variable with `dfn`/`dfd` degrees of freedom and
/// non-centrality `nc`.
///
/// Computed as the Poisson(nc/2)-weighted mixture of central regularized
/// incomplete beta values
///
/// ```text
///   ncfdtr = Σ_{j≥0} e^{−λ} λ^j/j! · I_y(dfn/2 + j, dfd/2),
///   λ = nc/2,  y = dfn·f / (dfn·f + dfd)
/// ```
///
/// The sum is accumulated outward from the Poisson mode `j₀ = ⌊λ⌋` (mode weight
/// formed in log space) so large `nc` neither underflows `e^{−λ}` nor loses
/// precision.
#[must_use]
pub fn ncfdtr(dfn: f64, dfd: f64, nc: f64, f: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || nc.is_nan() || f.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 0.0;
    }
    let y = dfn * f / (dfn * f + dfd);
    if nc == 0.0 {
        return btdtr(0.5 * dfn, 0.5 * dfd, y);
    }
    let lam = nc / 2.0;
    let j0 = lam.floor();
    let logw0 =
        -lam + j0 * lam.ln() - gammaln_scalar(j0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let w0 = logw0.exp();

    let mut total = 0.0_f64;
    let mut w = w0;
    let mut j = j0;
    let mut steps = 0;
    while steps < 100_000 {
        total += w * btdtr(0.5 * dfn + j, 0.5 * dfd, y);
        j += 1.0;
        w *= lam / j;
        if w < 1e-300 || (w < 1e-17 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1;
    }
    w = w0;
    j = j0;
    while j > 0.0 {
        w *= j / lam;
        j -= 1.0;
        total += w * btdtr(0.5 * dfn + j, 0.5 * dfd, y);
        if w < 1e-17 * total.max(1e-300) {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Inverse of [`ncfdtr`] in the argument `f`.
///
/// Returns `f` such that `ncfdtr(dfn, dfd, nc, f) = p`, matching
/// `scipy.special.ncfdtri(dfn, dfd, nc, p)`. Monotone increasing in `f`, solved
/// by bracket-and-bisect. `p = 0 → 0`, `p = 1 → +∞`, `p ∉ [0, 1]` → NaN.
#[must_use]
pub fn ncfdtri(dfn: f64, dfd: f64, nc: f64, p: f64) -> f64 {
    if dfn.is_nan() || dfd.is_nan() || nc.is_nan() || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 || nc < 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut hi = 1.0_f64;
    while ncfdtr(dfn, dfd, nc, hi) < p {
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    let mut lo = 0.0_f64;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if ncfdtr(dfn, dfd, nc, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Inverse of [`nctdtr`] in the argument `t`.
///
/// Returns `t` such that `nctdtr(df, nc, t) = p`, matching
/// `scipy.special.nctdtrit(df, nc, p)`. Monotone increasing in `t`, solved by
/// bracket-and-bisect over the whole real line. Following scipy's cdflib, the
/// exact boundaries `p ≤ 0` and `p ≥ 1` return `+∞`.
#[must_use]
pub fn nctdtrit(df: f64, nc: f64, p: f64) -> f64 {
    if df.is_nan() || nc.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 || p >= 1.0 {
        return f64::INFINITY;
    }
    // Bracket the root.
    let mut lo = -1.0_f64;
    while nctdtr(df, nc, lo) > p {
        lo *= 2.0;
        if lo < -1e300 {
            return f64::NEG_INFINITY;
        }
    }
    let mut hi = 1.0_f64;
    while nctdtr(df, nc, hi) < p {
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if nctdtr(df, nc, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Student's t distribution CDF.
///
/// Returns P(T <= t) where T follows a Student's t distribution
/// with v degrees of freedom.
///
/// Matches `scipy.special.stdtr(v, t)`.
#[must_use]
/// Non-central Student's t cumulative distribution function.
///
/// Matches `scipy.special.nctdtr(df, nc, t)`: the CDF at `t` of a non-central
/// t variable with `df` degrees of freedom and non-centrality `nc`.
///
/// Uses Lenth's (1989, AS 243) series. For `t ≥ 0`,
///
/// ```text
///   P(T ≤ t) = Φ(−δ) + ½ Σ_{j≥0} [ p_j·I_x(j+½, df/2) + q_j·I_x(j+1, df/2) ],
///   x = t²/(t²+df),  λ = δ²/2,
///   p_j = e^{−λ} λ^j / j!,  q_j = (δ/√2)·e^{−λ} λ^j / Γ(j+3/2)
/// ```
///
/// with `nctdtr(df, nc, t) = 1 − nctdtr(df, −nc, −t)` for `t < 0`. The series is
/// summed outward from the Poisson mode `j₀=⌊λ⌋` (mode weights formed in log
/// space, with `q`'s sign carried separately) so large `nc` stays stable.
#[must_use]
pub fn nctdtr(df: f64, nc: f64, t: f64) -> f64 {
    if df.is_nan() || nc.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if t < 0.0 {
        return 1.0 - nctdtr(df, -nc, -t);
    }
    let phi = crate::convenience::ndtr_scalar(-nc);
    if t == 0.0 {
        return phi;
    }
    let x = t * t / (t * t + df);
    let half_df = 0.5 * df;
    let lam = 0.5 * nc * nc;
    if lam == 0.0 {
        return phi + 0.5 * btdtr(0.5, half_df, x);
    }

    let j0 = lam.floor();
    let lg = |z: f64| gammaln_scalar(z, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let p0 = (-lam + j0 * lam.ln() - lg(j0 + 1.0)).exp();
    let q_sign = if nc >= 0.0 { 1.0 } else { -1.0 };
    let q0 = q_sign
        * ((nc.abs() / std::f64::consts::SQRT_2).ln() - lam + j0 * lam.ln() - lg(j0 + 1.5)).exp();

    let mut s = 0.0_f64;
    // Upward from the mode.
    let (mut p, mut q, mut j) = (p0, q0, j0);
    let mut steps = 0;
    while steps < 100_000 {
        s += p * btdtr(j + 0.5, half_df, x) + q * btdtr(j + 1.0, half_df, x);
        j += 1.0;
        p *= lam / j;
        q *= lam / (j + 0.5);
        let m = p.abs().max(q.abs());
        if (p.abs() < 1e-300 && q.abs() < 1e-300) || (m < 1e-17 * s.abs().max(1e-300) && j > lam) {
            break;
        }
        steps += 1;
    }
    // Downward from the mode.
    p = p0;
    q = q0;
    j = j0;
    while j > 0.0 {
        p *= j / lam;
        q *= (j + 0.5) / lam;
        j -= 1.0;
        s += p * btdtr(j + 0.5, half_df, x) + q * btdtr(j + 1.0, half_df, x);
        if p.abs().max(q.abs()) < 1e-17 * s.abs().max(1e-300) {
            break;
        }
    }
    (phi + 0.5 * s).clamp(0.0, 1.0)
}

pub fn stdtr(v: f64, t: f64) -> f64 {
    if v.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }

    // Use the relation with incomplete beta:
    // For t >= 0: stdtr(v, t) = 1 - 0.5 * I(v/(v+t²); v/2, 1/2)
    // For t < 0:  stdtr(v, t) = 0.5 * I(v/(v+t²); v/2, 1/2)
    let x = v / (v + t * t);
    let half_beta = 0.5 * btdtr(0.5 * v, 0.5, x);

    if t >= 0.0 { 1.0 - half_beta } else { half_beta }
}

/// Student's t distribution survival function.
///
/// Returns P(T > t) = 1 - stdtr(v, t) where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtrc(v, t)`.
#[must_use]
pub fn stdtrc(v: f64, t: f64) -> f64 {
    if v.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }

    // Use symmetry: stdtrc(v, t) = 1 - stdtr(v, t) = stdtr(v, -t)
    let x = v / (v + t * t);
    let half_beta = 0.5 * btdtr(0.5 * v, 0.5, x);

    if t >= 0.0 { half_beta } else { 1.0 - half_beta }
}

/// Inverse Student's t distribution CDF.
///
/// Returns t such that P(T <= t) = p where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtrit(v, p)`.
#[must_use]
pub fn stdtrit(v: f64, p: f64) -> f64 {
    if v.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // v == 1 is the standard Cauchy distribution, whose quantile is
    // tan(π(p − 1/2)) = −cos(πp)/sin(πp). Evaluating it directly keeps full
    // precision in the tails, where the general inverse-beta path below loses
    // ~1e-6 (e.g. stdtrit(1, 1e-6) was off by 2.8e-6 vs scipy).
    if v == 1.0 {
        let pi_p = std::f64::consts::PI * p;
        return -pi_p.cos() / pi_p.sin();
    }

    // Use the inverse beta to find z = v/(v+t²)
    // For p > 0.5: z = btdtri(v/2, 1/2, 2*(1-p))
    // For p < 0.5: z = btdtri(v/2, 1/2, 2*p)
    let (z, sign) = if p > 0.5 {
        (btdtri(0.5 * v, 0.5, 2.0 * (1.0 - p)), 1.0)
    } else {
        (btdtri(0.5 * v, 0.5, 2.0 * p), -1.0)
    };

    // z = v/(v+t²) => t² = v*(1-z)/z => t = sign * sqrt(v*(1-z)/z)
    if z <= 0.0 {
        return sign * f64::INFINITY;
    }
    if z >= 1.0 {
        return 0.0;
    }

    sign * (v * (1.0 - z) / z).sqrt()
}

/// Inverse Student's t distribution CDF with respect to degrees of freedom.
///
/// Returns v such that P(T <= t) = p where T follows a Student's t
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.stdtridf(p, t)`, including CDFlib sentinel
/// values for no-solution boundary cases.
#[must_use]
pub fn stdtridf(p: f64, t: f64) -> f64 {
    const LOWER_SENTINEL: f64 = -1.0e100;
    const UPPER_SENTINEL: f64 = 1.0e10;
    const MIN_DF_SENTINEL: f64 = 5.0e-51;

    if p.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if t == 0.0 {
        return if p == 0.5 { 5.0 } else { UPPER_SENTINEL };
    }
    if t.is_infinite() {
        return if t.is_sign_positive() {
            if p == 1.0 {
                5.0
            } else if p > 0.5 {
                LOWER_SENTINEL
            } else {
                UPPER_SENTINEL
            }
        } else if p == 0.0 {
            5.0
        } else if p < 0.5 {
            LOWER_SENTINEL
        } else {
            UPPER_SENTINEL
        };
    }
    if p == 0.5 {
        return MIN_DF_SENTINEL;
    }
    if t < 0.0 {
        return stdtridf(1.0 - p, -t);
    }
    if p < 0.5 {
        return LOWER_SENTINEL;
    }

    let upper_value = stdtr(UPPER_SENTINEL, t);
    if !upper_value.is_finite() {
        return f64::NAN;
    }
    if p >= upper_value {
        return UPPER_SENTINEL;
    }

    let mut lo = 0.0;
    let mut hi = UPPER_SENTINEL;

    for _ in 0..240 {
        let mid = lo + (hi - lo) * 0.5;
        let value = stdtr(mid, t);
        if !value.is_finite() || value < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let df = lo + (hi - lo) * 0.5;
    if df == 0.0 { MIN_DF_SENTINEL } else { df }
}

/// Binomial distribution CDF.
///
/// Returns P(X <= k) where X follows a binomial distribution
/// with n trials and success probability p.
///
/// Matches `scipy.special.bdtr(k, n, p)`.
#[must_use]
pub fn bdtr(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }

    // bdtr(k, n, p) = I(1-p; n-k, k+1) = betainc(n-k, k+1, 1-p)
    // Or equivalently: 1 - betainc(k+1, n-k, p)
    btdtr(n - k, k + 1.0, 1.0 - p)
}

/// Binomial distribution survival function.
///
/// Returns P(X > k) where X follows a binomial distribution
/// with n trials and success probability p.
///
/// Matches `scipy.special.bdtrc(k, n, p)`.
#[must_use]
pub fn bdtrc(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k < 0.0 {
        return 1.0;
    }
    if k >= n {
        return 0.0;
    }

    // bdtrc(k, n, p) = betainc(k+1, n-k, p)
    btdtr(k + 1.0, n - k, p)
}

/// Inverse binomial distribution CDF.
///
/// Returns p such that P(X <= k) = y where X follows a binomial distribution
/// with n trials.
///
/// Matches `scipy.special.bdtri(k, n, y)`.
#[must_use]
pub fn bdtri(k: f64, n: f64, y: f64) -> f64 {
    if k.is_nan() || n.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 || !(0.0..=1.0).contains(&y) || k < 0.0 || k > n {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 {
        return 0.0;
    }
    if k >= n {
        return 0.0;
    }

    // bdtr(k, n, p) = betainc(n-k, k+1, 1-p) = y
    // So 1-p = btdtri(n-k, k+1, y)
    // Thus p = 1 - btdtri(n-k, k+1, y)
    1.0 - btdtri(n - k, k + 1.0, y)
}

/// Inverse binomial CDF with respect to `k`.
///
/// Returns `k` such that `bdtr(k, n, p) = y`.
/// Matches `scipy.special.bdtrik(y, n, p)` for the regular interior domain.
#[must_use]
pub fn bdtrik(y: f64, n: f64, p: f64) -> f64 {
    if y.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 || p == 1.0 {
        return n;
    }

    bisect_increasing(0.0, n, y, |k| bdtr(k, n, p))
}

/// Inverse binomial CDF with respect to `n`.
///
/// Returns `n` such that `bdtr(k, n, p) = y`.
/// Matches `scipy.special.bdtrin(k, y, p)` for the regular interior domain.
#[must_use]
pub fn bdtrin(k: f64, y: f64, p: f64) -> f64 {
    if k.is_nan() || y.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }
    if y == 1.0 || p == 1.0 {
        return k;
    }

    let lo = k;
    let mut hi = (k + 1.0).max(1.0);
    while bdtr(k, hi, p) > y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_decreasing(lo, hi, y, |n| bdtr(k, n, p))
}

/// Negative binomial distribution CDF.
///
/// Returns P(X <= k) where X is the number of failures before n successes
/// in a sequence of Bernoulli trials with success probability p.
///
/// Matches `scipy.special.nbdtr(k, n, p)`.
#[must_use]
pub fn nbdtr(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    if k < 0.0 {
        return 0.0;
    }

    // nbdtr(k, n, p) = I_p(n, k+1) = betainc(n, k+1, p)
    btdtr(n, k + 1.0, p)
}

/// Negative binomial distribution survival function.
///
/// Returns P(X > k) where X is the number of failures before n successes.
///
/// Matches `scipy.special.nbdtrc(k, n, p)`.
#[must_use]
pub fn nbdtrc(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p == 1.0 {
        return 0.0;
    }
    if k < 0.0 {
        return 1.0;
    }

    // nbdtrc(k, n, p) = 1 - betainc(n, k+1, p) = betainc(k+1, n, 1-p)
    btdtr(k + 1.0, n, 1.0 - p)
}

/// Inverse negative binomial distribution CDF.
///
/// Returns p such that nbdtr(k, n, p) = y.
///
/// Matches `scipy.special.nbdtri(k, n, y)`.
#[must_use]
pub fn nbdtri(k: f64, n: f64, y: f64) -> f64 {
    if k.is_nan() || n.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || k < 0.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }

    // nbdtr(k, n, p) = betainc(n, k+1, p) = y
    // So p = btdtri(n, k+1, y)
    btdtri(n, k + 1.0, y)
}

/// Inverse negative-binomial CDF with respect to `k`.
///
/// Returns `k` such that `nbdtr(k, n, p) = y`.
/// Matches `scipy.special.nbdtrik(y, n, p)` for the regular interior domain.
#[must_use]
pub fn nbdtrik(y: f64, n: f64, p: f64) -> f64 {
    if y.is_nan() || n.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if n <= 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if y == 0.0 || p == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while nbdtr(hi, n, p) < y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_increasing(0.0, hi, y, |k| nbdtr(k, n, p))
}

/// Inverse negative-binomial CDF with respect to `n`.
///
/// Returns `n` such that `nbdtr(k, n, p) = y`.
/// Matches `scipy.special.nbdtrin(k, y, p)` for the regular interior domain.
#[must_use]
pub fn nbdtrin(k: f64, y: f64, p: f64) -> f64 {
    if k.is_nan() || y.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&y) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if y == 0.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }
    if y == 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
    }

    let mut hi = 1.0;
    while nbdtr(k, hi, p) > y {
        hi *= 2.0;
        if hi >= DISTRIBUTION_INVERSE_UPPER_SENTINEL {
            return DISTRIBUTION_INVERSE_UPPER_SENTINEL;
        }
    }

    bisect_decreasing(0.0, hi, y, |n| nbdtr(k, n, p))
}

fn bisect_increasing<F>(mut lo: f64, mut hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..DISTRIBUTION_INVERSE_ITERS {
        let mid = lo + (hi - lo) * 0.5;
        let value = f(mid);
        if !value.is_finite() || value < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * 0.5
}

fn bisect_decreasing<F>(mut lo: f64, mut hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    for _ in 0..DISTRIBUTION_INVERSE_ITERS {
        let mid = lo + (hi - lo) * 0.5;
        let value = f(mid);
        if !value.is_finite() || value > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * 0.5
}

/// Evaluate `f(0..n)` into a `Vec`, parallel over index chunks for large `n`.
/// Beta-family kernels (incomplete-beta continued fractions) are expensive per element
/// and each index writes its own slot, so chunking across cores and concatenating in
/// index order is bit-identical to `(0..n).map(f).collect()` — including returning the
/// first failing index's error in index order. Used by the array arms below.
fn par_map_indices<T, G>(n: usize, f: G) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    G: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 128)
            .max(1)
    };
    if nthreads <= 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("beta array worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

fn map_real_binary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    match (a, b) {
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealScalar(rhs)) => {
            kernel(*lhs, *rhs).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealScalar(rhs)) => {
            let rhs = *rhs;
            par_map_indices(lhs.len(), |i| kernel(lhs[i], rhs)).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(lhs), SpecialTensor::RealVec(rhs)) => {
            let lhs = *lhs;
            par_map_indices(rhs.len(), |i| kernel(lhs, rhs[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(lhs), SpecialTensor::RealVec(rhs)) => {
            if lhs.len() != rhs.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", lhs.len(), rhs.len()),
                    "fail_closed",
                    "vector inputs must have matching lengths",
                    false,
                );
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(lhs.len(), |i| kernel(lhs[i], rhs[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _)
        | (SpecialTensor::ComplexVec(_), _)
        | (_, SpecialTensor::ComplexScalar(_))
        | (_, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid special-function input",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid special-function input",
            })
        }
    }
}

fn map_real_ternary<F>(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError> + Sync,
{
    match (a, b, c) {
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => kernel(*av, *bv, *cv).map(SpecialTensor::RealScalar),
        (
            SpecialTensor::RealVec(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealScalar(cv),
        ) => {
            let (bv, cv) = (*bv, *cv);
            par_map_indices(av.len(), |i| kernel(av[i], bv, cv)).map(SpecialTensor::RealVec)
        }
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealVec(bv),
            SpecialTensor::RealScalar(cv),
        ) => {
            let (av, cv) = (*av, *cv);
            par_map_indices(bv.len(), |i| kernel(av, bv[i], cv)).map(SpecialTensor::RealVec)
        }
        (
            SpecialTensor::RealScalar(av),
            SpecialTensor::RealScalar(bv),
            SpecialTensor::RealVec(cv),
        ) => {
            let (av, bv) = (*av, *bv);
            par_map_indices(cv.len(), |i| kernel(av, bv, cv[i])).map(SpecialTensor::RealVec)
        }
        (SpecialTensor::ComplexScalar(_), _, _)
        | (SpecialTensor::ComplexVec(_), _, _)
        | (_, SpecialTensor::ComplexScalar(_), _)
        | (_, SpecialTensor::ComplexVec(_), _)
        | (_, _, SpecialTensor::ComplexScalar(_))
        | (_, _, SpecialTensor::ComplexVec(_)) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_broadcast_pattern",
                "fail_closed",
                "unsupported broadcast pattern for ternary inputs",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported broadcast pattern for ternary inputs",
            })
        }
    }
}

/// Sign of Γ(x) on the real line. Γ > 0 for x > 0; for x < 0 it alternates on
/// each interval between consecutive negative integers. (Nonpositive-integer
/// poles are carried by the ±inf log path, so the value here is harmless there.)
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if (-x).ceil() as i64 % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// scipy/cephes `beta` at a nonpositive-integer argument when the OTHER argument
/// is strictly positive. The straight `Γ(a)Γ(b)/Γ(a+b)` route gives `inf − inf`
/// in the log domain (→ NaN) and the gamma-sign product picks the wrong side of
/// the pole, so handle these directly:
///   * positive integer `m` with the pole at `-k` (`k ≥ m`): the Γ(b) and Γ(a+b)
///     poles cancel, leaving the finite rational `B(m,−k) = (−1)^m (m−1)!(k−m)!/k!`
///     (computed as `(−1)^m (m−1)! / (k·(k−1)···(k−m+1))` to avoid factorial
///     overflow);
///   * otherwise `+inf` (scipy's one-signed value at the simple pole).
/// Returns `None` when neither arg is a nonpositive integer paired with a strictly
/// positive other arg (the both-nonpositive-integer cases are cephes-specific and
/// asymmetric — e.g. `beta(-2,-1)=-inf` but `beta(-1,-2)=+inf` — so we leave the
/// existing path to handle them rather than guess).
fn beta_nonpos_integer_special(a: f64, b: f64) -> Option<f64> {
    let is_nonpos_int = |x: f64| x <= 0.0 && x.is_finite() && x.fract() == 0.0;
    let (pole, other) = if is_nonpos_int(a) && b > 0.0 {
        (a, b)
    } else if is_nonpos_int(b) && a > 0.0 {
        (b, a)
    } else {
        return None;
    };
    let k = (-pole) as i64; // nonnegative integer magnitude of the pole
    if other.fract() == 0.0 {
        let m = other as i64; // positive integer
        if m >= 1 && m <= k {
            let mut val = if m % 2 == 0 { 1.0 } else { -1.0 };
            for i in 1..m {
                val *= i as f64; // (m-1)!
            }
            for j in 0..m {
                val /= (k - j) as f64; // k·(k-1)···(k-m+1)
            }
            return Some(val);
        }
    }
    Some(f64::INFINITY)
}

pub(crate) fn beta_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if let Some(v) = beta_nonpos_integer_special(a, b) {
        return Ok(v);
    }
    // Symmetry beta(a,b)=beta(b,a)
    let (a, b) = if a < b { (b, a) } else { (a, b) };

    let log_value = betaln_scalar(a, b, mode)?;
    // B(a,b) = Γ(a)Γ(b)/Γ(a+b) is signed: betaln gives ln|B|, so restore the sign
    // from the gamma factors (scipy.special.beta(-2.5,3)=-1.0667). For positive
    // a,b every gamma is positive => sign = +1 (unchanged).
    let sign = gamma_sign(a) * gamma_sign(b) * gamma_sign(a + b);
    const LN_MAX: f64 = 709.782_712_893_384;
    const LN_MIN: f64 = -745.133_219_101_941_1;

    if log_value > LN_MAX {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "beta",
                mode,
                "overflow_risk",
                format!("a={a},b={b},log_beta={log_value}"),
                "fail_closed",
                "beta overflow risk",
                true,
            );
            return Err(SpecialError {
                function: "beta",
                kind: SpecialErrorKind::OverflowRisk,
                mode,
                detail: "beta overflow risk",
            });
        }
        record_special_trace(
            "beta",
            mode,
            "overflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_inf",
            "strict overflow fallback",
            true,
        );
        return Ok(sign * f64::INFINITY);
    }
    if log_value < LN_MIN {
        record_special_trace(
            "beta",
            mode,
            "underflow_risk",
            format!("a={a},b={b},log_beta={log_value}"),
            "returned_zero",
            "underflow-safe clamp to zero",
            true,
        );
        return Ok(sign * 0.0);
    }

    Ok(sign * log_value.exp())
}

pub fn betaln_scalar(a: f64, b: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() {
        return Ok(f64::NAN);
    }
    if matches!(mode, RuntimeMode::Hardened) && (a <= 0.0 || b <= 0.0) {
        record_special_trace(
            "betaln",
            mode,
            "domain_error",
            format!("a={a},b={b}"),
            "fail_closed",
            "betaln principal domain requires positive parameters",
            false,
        );
        return Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "betaln principal domain requires positive parameters",
        });
    }
    // Nonpositive-integer argument with a positive partner: the gammaln sum below
    // is inf − inf = NaN, so use the closed-form beta value and take its log.
    // frankenscipy-dwd3d
    if let Some(v) = beta_nonpos_integer_special(a, b) {
        return Ok(if v.is_infinite() {
            f64::INFINITY
        } else {
            v.abs().ln()
        });
    }
    // `betaln(a,b) = ln|B(a,b)| = gammaln(a) + gammaln(b) - gammaln(a+b)` is valid
    // for ALL real a, b, not just positives: SciPy returns finite values for
    // negative non-integer arguments (e.g. betaln(-2.5, 3.0) = 0.0645) and the
    // pole limits (+/-inf) elsewhere. `gammaln_scalar(_, Strict)` returns the
    // reflection-formula value for negative non-integers and +inf at nonpositive-
    // integer poles, so the sum reproduces SciPy across the real line. (Only the
    // rare both-nonpositive-integer pole-cancellation, e.g. betaln(-3, 2), is left
    // as NaN; SciPy resolves it via a finite gamma-ratio, see frankenscipy notes.)
    let lg_a = gammaln_scalar(a, RuntimeMode::Strict)?;
    let lg_b = gammaln_scalar(b, RuntimeMode::Strict)?;
    let lg_ab = gammaln_scalar(a + b, RuntimeMode::Strict)?;
    Ok(lg_a + lg_b - lg_ab)
}

pub fn complex_betaln_scalar(a: Complex64, b: Complex64) -> Complex64 {
    let lg_a = gamma::complex_gammaln(a);
    let lg_b = gamma::complex_gammaln(b);
    let lg_ab = gamma::complex_gammaln(a + b);
    lg_a + lg_b - lg_ab
}

pub fn complex_beta_scalar(a: Complex64, b: Complex64) -> Complex64 {
    complex_betaln_scalar(a, b).exp()
}

pub fn complex_betainc_scalar(a: Complex64, b: Complex64, x: Complex64) -> Complex64 {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    if x.re == 0.0 && x.im == 0.0 {
        return zero;
    }
    if x.re == 1.0 && x.im == 0.0 {
        return one;
    }

    let ln_beta = complex_betaln_scalar(a, b);
    let ln_front = a * x.ln() + b * (one - x).ln() - ln_beta;
    let front = ln_front.exp();

    let threshold = (a + one) / (a + b + Complex64::new(2.0, 0.0));
    if x.re < threshold.re {
        front * complex_betacf(a, b, x) / a
    } else {
        one - front * complex_betacf(b, a, one - x) / b
    }
}

fn complex_betacf(a: Complex64, b: Complex64, x: Complex64) -> Complex64 {
    const MAX_ITERS: usize = 200;
    const EPS: f64 = 3.0e-14;
    const MIN_NUM: f64 = 1.0e-300;

    let one = Complex64::new(1.0, 0.0);
    let qab = a + b;
    let qap = a + one;
    let qam = a - one;
    let mut c = one;
    let mut d = one - qab * x / qap;
    if d.abs() < MIN_NUM {
        d = Complex64::new(MIN_NUM, 0.0);
    }
    d = d.recip();
    let mut h = d;

    for m in 1..=MAX_ITERS {
        let m_f = Complex64::new(m as f64, 0.0);
        let m2 = m_f * Complex64::new(2.0, 0.0);
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = one + aa * d;
        if d.abs() < MIN_NUM {
            d = Complex64::new(MIN_NUM, 0.0);
        }
        c = one + aa / c;
        if c.abs() < MIN_NUM {
            c = Complex64::new(MIN_NUM, 0.0);
        }
        d = d.recip();
        h = h * d * c;

        let aa2 = (a + m_f) * (qab + m_f) * x * Complex64::new(-1.0, 0.0) / ((a + m2) * (qap + m2));
        d = one + aa2 * d;
        if d.abs() < MIN_NUM {
            d = Complex64::new(MIN_NUM, 0.0);
        }
        c = one + aa2 / c;
        if c.abs() < MIN_NUM {
            c = Complex64::new(MIN_NUM, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta - one).abs() <= EPS {
            break;
        }
    }

    h
}

pub fn betainc_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc domain requires x in [0, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc domain requires x in [0, 1]",
                })
            }
        };
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    if a <= 0.0 || b <= 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betainc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betainc requires positive shape parameters",
                    false,
                );
                Err(SpecialError {
                    function: "betainc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betainc requires positive shape parameters",
                })
            }
        };
    }

    let ln_beta = betaln_scalar(a, b, RuntimeMode::Strict)?;
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(front * betacf(a, b, x) / a)
    } else {
        Ok(1.0 - front * betacf(b, a, 1.0 - x) / b)
    }
}

/// Natural log of the regularized incomplete beta function `I_x(a, b)`.
///
/// `ln I_x(a, b)` stays finite deep in the tail where `I` itself underflows to
/// 0 (so `betainc_scalar(a, b, x).ln()` would be `-inf`). The shared front
/// factor `front = exp(a*ln x + b*ln(1-x) - lnB(a,b))` underflows but the
/// Lentz continued fraction `betacf` is `O(1)`. In the small-`I` region
/// (`x < (a+1)/(a+b+2)`) `I = front * betacf(a,b,x)/a`, so
/// `ln I = (a*ln x + b*ln(1-x) - lnB(a,b)) + ln(betacf(a,b,x)/a)` keeps full
/// precision; in the large-`I` region `I = 1 - complement` and
/// `ln I = ln1p(-complement)`. Matches `betainc_scalar(a, b, x).ln()` wherever
/// the latter is representable.
///
/// For the complementary log use the reflection `ln(1 - I_x(a,b)) =
/// log_betainc_scalar(b, a, 1 - x)`.
///
/// Domain: `a > 0`, `b > 0`, `x in [0, 1]`. Returns `NaN` for invalid inputs,
/// `-inf` at `x = 0` (`I = 0`), and `0.0` at `x = 1` (`I = 1`).
#[must_use]
pub fn log_betainc_scalar(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        return 0.0;
    }

    let ln_beta = match betaln_scalar(a, b, RuntimeMode::Strict) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let log_front = a * x.ln() + b * (1.0 - x).ln() - ln_beta;

    if x < (a + 1.0) / (a + b + 2.0) {
        // Small-I region: log form is finite even when front underflows.
        log_front + (betacf(a, b, x) / a).ln()
    } else {
        // Large-I region: I = 1 - complement; complement is small & representable.
        let complement = log_front.exp() * betacf(b, a, 1.0 - x) / b;
        (-complement).ln_1p()
    }
}

pub fn betaincc_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betaincc domain requires x in [0, 1]",
                    false,
                );
                Err(SpecialError {
                    function: "betaincc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betaincc domain requires x in [0, 1]",
                })
            }
        };
    }
    if x == 0.0 {
        return Ok(1.0);
    }
    if x == 1.0 {
        return Ok(0.0);
    }
    if a <= 0.0 || b <= 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(f64::NAN)
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "betaincc",
                    mode,
                    "domain_error",
                    format!("a={a},b={b},x={x}"),
                    "fail_closed",
                    "betaincc requires positive shape parameters",
                    false,
                );
                Err(SpecialError {
                    function: "betaincc",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "betaincc requires positive shape parameters",
                })
            }
        };
    }

    betainc_scalar(b, a, 1.0 - x, mode)
}

#[must_use]
pub fn betainccinv_scalar(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 {
        return 0.0;
    }

    1.0 - crate::convenience::betaincinv_scalar(b, a, y)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITERS: usize = 200;
    const EPS: f64 = 3.0e-14;
    const MIN_NUM: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MIN_NUM {
        d = MIN_NUM;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < MIN_NUM {
            d = MIN_NUM;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < MIN_NUM {
            c = MIN_NUM;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
    }

    h
}

fn invert_monotone_positive(cdf: impl Fn(f64) -> f64, target: f64, increasing: bool) -> f64 {
    let mut lo = f64::MIN_POSITIVE;
    let mut hi = 1.0;
    let mut hi_value = cdf(hi);

    if !hi_value.is_finite() {
        return f64::NAN;
    }

    if increasing {
        while hi_value < target {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
            hi_value = cdf(hi);
            if !hi_value.is_finite() {
                return f64::NAN;
            }
        }
    } else {
        while hi_value > target {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
            hi_value = cdf(hi);
            if !hi_value.is_finite() {
                return f64::NAN;
            }
        }
    }

    for _ in 0..180 {
        let mid = 0.5 * (lo + hi);
        let value = cdf(mid);
        if !value.is_finite() {
            return f64::NAN;
        }
        if increasing {
            if value < target {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if value > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1.0e-12 * hi.abs().max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi)
}

fn gammaln_scalar(value: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let tensor = SpecialTensor::RealScalar(value);
    let result = gamma::gammaln(&tensor, mode)?;
    match result {
        SpecialTensor::RealScalar(v) => Ok(v),
        _ => Err(SpecialError {
            function: "betaln",
            kind: SpecialErrorKind::NotYetImplemented,
            mode,
            detail: "unexpected non-scalar gammaln output",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdtrit_v1_is_exact_cauchy_in_the_tails() {
        // v == 1 is the standard Cauchy. Verify via the independent round-trip
        // through the Cauchy CDF F(t) = 1/2 + atan(t)/π, which must recover p.
        // Regression: the general inverse-beta path lost ~3e-6 at p = 1e-6.
        for &p in &[
            1e-8_f64, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.75, 0.9, 0.99, 0.999_999,
        ] {
            let t = stdtrit(1.0, p);
            let cdf = 0.5 + t.atan() / std::f64::consts::PI;
            assert!(
                (cdf - p).abs() < 1e-12,
                "stdtrit(1, {p}) = {t}: CDF round-trip {cdf} != {p}"
            );
        }
    }

    type DistributionInverseCase = (fn(f64, f64, f64) -> f64, f64, f64, f64, f64);

    #[test]
    fn log_betainc_matches_betainc_where_representable() {
        // exp(log I) == I wherever I is representable (betainc_scalar is the
        // scipy-validated reference). Covers both branches and the symmetry.
        let cases = [
            (0.5, 0.5, 0.2),
            (2.0, 3.0, 0.25),
            (2.0, 3.0, 0.75),
            (5.0, 1.0, 0.5),
            (1.0, 5.0, 0.5),
            (10.0, 10.0, 0.3),
            (0.5, 2.0, 0.001),
        ];
        for (a, b, x) in cases {
            let i = betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap();
            let li = log_betainc_scalar(a, b, x);
            assert!(i > 0.0, "precondition: I representable for ({a},{b},{x})");
            let rel = (li - i.ln()).abs() / i.ln().abs().max(1.0);
            assert!(
                rel <= 1e-11,
                "log_betainc({a},{b},{x})={li} vs ln(I)={}",
                i.ln()
            );
        }
    }

    #[test]
    fn log_betainc_finite_in_underflowed_tail() {
        // Deep tail: I underflows to 0 (ln I < ~-745) but log I stays finite and
        // matches the leading a*ln x - ln(a) - lnB(a,b) for tiny x.
        for (a, b, x) in [(2.0, 3.0, 1e-170), (5.0, 2.0, 1e-70), (3.0, 3.0, 1e-130)] {
            assert_eq!(
                betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap(),
                0.0,
                "precondition: I underflows for ({a},{b},{x})"
            );
            let li = log_betainc_scalar(a, b, x);
            assert!(
                li.is_finite() && li < -100.0,
                "log I({a},{b},{x})={li} not finite"
            );
            // I ≈ x^a/(a·B(a,b)) for tiny x ⇒ ln I ≈ a·ln x − ln a − lnB(a,b).
            let asymp = a * x.ln() - a.ln() - betaln_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (li - asymp).abs() / asymp.abs() < 1e-2,
                "log I({a},{b},{x})={li} vs asymptotic {asymp}"
            );
        }
    }

    #[test]
    fn log_betainc_edge_and_reflection() {
        assert_eq!(log_betainc_scalar(2.0, 3.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(log_betainc_scalar(2.0, 3.0, 1.0), 0.0);
        assert!(log_betainc_scalar(-1.0, 2.0, 0.5).is_nan());
        assert!(log_betainc_scalar(2.0, 3.0, 1.5).is_nan());
        // ln(1 - I_x(a,b)) == log_betainc(b, a, 1-x).
        for (a, b, x) in [(2.0, 3.0, 0.7), (5.0, 1.5, 0.9)] {
            let icc = betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap();
            let lcc = log_betainc_scalar(b, a, 1.0 - x);
            assert!((lcc - icc.ln()).abs() / icc.ln().abs().max(1.0) <= 1e-11);
        }
    }

    #[test]
    fn betaincc_complements_betainc_and_preserves_endpoints() {
        for &(a, b, x) in &[
            (0.5_f64, 0.5, 0.2),
            (2.0, 3.0, 0.25),
            (5.0, 2.0, 0.75),
            (10.0, 10.0, 0.5),
        ] {
            let lower = betainc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let upper = betaincc_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            assert!(
                (lower + upper - 1.0).abs() < 1.0e-12,
                "betainc + betaincc must sum to 1 for ({a}, {b}, {x})"
            );
        }

        assert_eq!(
            betaincc_scalar(2.0, 3.0, 0.0, RuntimeMode::Strict).unwrap_or(f64::NAN),
            1.0
        );
        assert_eq!(
            betaincc_scalar(2.0, 3.0, 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN),
            0.0
        );
        assert!(betaincc_scalar(2.0, 3.0, -0.1, RuntimeMode::Strict).is_ok_and(f64::is_nan));
    }

    #[test]
    fn betainccinv_matches_scipy_reference_and_inverts_tail() {
        let reference = betainccinv_scalar(2.0, 3.0, 0.25);
        assert!(
            (reference - 0.543_678_285_419_080_3).abs() < 1.0e-12,
            "betainccinv(2, 3, 0.25) = {reference}"
        );

        for &y in &[0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
            let x = betainccinv_scalar(2.0, 3.0, y);
            let tail = betaincc_scalar(2.0, 3.0, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            assert!(
                (tail - y).abs() < 1.0e-9,
                "betaincc(2, 3, betainccinv(..., {y})) = {tail}"
            );
        }

        assert_eq!(betainccinv_scalar(2.0, 3.0, 0.0), 1.0);
        assert_eq!(betainccinv_scalar(2.0, 3.0, 1.0), 0.0);
        assert!(betainccinv_scalar(2.0, 3.0, -0.1).is_nan());
        assert!(betainccinv_scalar(-1.0, 3.0, 0.5).is_nan());
    }

    #[test]
    fn betaincc_tensor_dispatch_broadcasts_real_vectors() {
        let result = betaincc(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(3.0),
            &SpecialTensor::RealVec(vec![0.0, 0.25, 0.5, 1.0]),
            RuntimeMode::Strict,
        )
        .expect("betaincc vector");
        let values = match result {
            SpecialTensor::RealVec(values) => values,
            _ => Vec::new(),
        };
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 1.0);
        assert!((values[1] - 0.738_281_25).abs() < 1.0e-12);
        assert!((values[2] - 0.312_5).abs() < 1.0e-12);
        assert_eq!(values[3], 0.0);
    }

    #[test]
    fn btdtr_closed_form_reference_values() {
        // /testing-golden-artifacts for [frankenscipy-1ulgv]:
        // btdtr (regularized incomplete beta I_x(a, b)) has multiple
        // analytic closed-form arms:
        //
        //   btdtr(a, b, 0) = 0
        //   btdtr(a, b, 1) = 1
        //   btdtr(1, 1, x) = x                    (uniform CDF)
        //   btdtr(a, 1, x) = x^a                  (power-law)
        //   btdtr(1, b, x) = 1 - (1 - x)^b
        //
        // Catches subtle sign or exponent errors in the underlying
        // incomplete-beta core.
        for &a in &[0.5_f64, 1.0, 2.5, 7.0] {
            for &b in &[0.5_f64, 1.0, 2.5, 7.0] {
                assert_eq!(btdtr(a, b, 0.0), 0.0, "btdtr({a}, {b}, 0) must be 0");
                assert!(
                    (btdtr(a, b, 1.0) - 1.0).abs() < 1e-12,
                    "btdtr({a}, {b}, 1) = {} != 1",
                    btdtr(a, b, 1.0)
                );
            }
        }

        for &x in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
            assert!(
                (btdtr(1.0, 1.0, x) - x).abs() < 1e-12,
                "btdtr(1, 1, {x}) = {} != {x}",
                btdtr(1.0, 1.0, x)
            );
        }

        for &a in &[0.5_f64, 2.0, 5.0] {
            for &x in &[0.1_f64, 0.5, 0.9] {
                let expected = x.powf(a);
                assert!(
                    (btdtr(a, 1.0, x) - expected).abs() < 1e-12,
                    "btdtr({a}, 1, {x}) = {} != x^a = {expected}",
                    btdtr(a, 1.0, x)
                );
            }
        }

        for &b in &[0.5_f64, 2.0, 5.0] {
            for &x in &[0.1_f64, 0.5, 0.9] {
                let expected = 1.0 - (1.0 - x).powf(b);
                assert!(
                    (btdtr(1.0, b, x) - expected).abs() < 1e-12,
                    "btdtr(1, {b}, {x}) = {} != 1 - (1-x)^b = {expected}",
                    btdtr(1.0, b, x)
                );
            }
        }
    }

    #[test]
    fn btdtr_swap_symmetry_identity() {
        // I_x(a, b) = 1 - I_{1-x}(b, a). Catches swapped-argument
        // bugs in the regularized incomplete beta core that the
        // closed-form arms above wouldn't detect.
        for &a in &[0.5_f64, 1.5, 3.0] {
            for &b in &[0.5_f64, 1.5, 3.0] {
                for &x in &[0.1_f64, 0.4, 0.7] {
                    let lhs = btdtr(a, b, x);
                    let rhs = 1.0 - btdtr(b, a, 1.0 - x);
                    assert!(
                        (lhs - rhs).abs() < 1e-10,
                        "btdtr({a},{b},{x}) = {lhs}, but 1 - btdtr({b},{a},{}) = {rhs}",
                        1.0 - x
                    );
                }
            }
        }
    }

    #[test]
    fn btdtrc_complement() {
        // btdtr + btdtrc should equal 1
        for &a in &[0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 1.0, 2.0, 5.0] {
                for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                    let sum = btdtr(a, b, x) + btdtrc(a, b, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "btdtr({a}, {b}, {x}) + btdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtria_inverse() {
        for &a in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 2.0, 5.0] {
                for &x in &[0.2, 0.3, 0.7] {
                    let p = btdtr(a, b, x);
                    let recovered = btdtria(p, b, x);
                    assert!(
                        (recovered - a).abs() <= 1e-8 * a.abs().max(1.0),
                        "btdtria/btdtr failed: a={a}, b={b}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtria_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (0.5, 2.0, 0.3, 1.0249306894715173, 2e-12),
            (0.8, 2.0, 0.3, 0.38229690978762904, 2e-12),
            (0.2, 2.0, 0.3, 2.084034825279176, 2e-12),
            (0.5, 5.0, 0.7, 11.231150488078322, 2e-11),
            (
                0.9999999999999999,
                13.584377534422599,
                0.9654253202433963,
                5.915066688085389,
                2e-12,
            ),
        ];
        for &(p, b, x, expected, tolerance) in cases {
            let actual = btdtria(p, b, x);
            assert!(
                (actual - expected).abs() < tolerance,
                "btdtria({p}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn btdtria_edges_match_scipy() {
        assert!(btdtria(0.0, 2.0, 0.3).is_infinite());
        assert_eq!(btdtria(1.0, 2.0, 0.3), f64::MIN_POSITIVE);
        assert_eq!(btdtria(1.0, 2.0, 1.1), f64::MIN_POSITIVE);
        assert!(btdtria(0.0, 2.0, 1.1).is_infinite());

        assert!(btdtria(0.5, 2.0, 0.0).is_nan());
        assert!(btdtria(0.5, 2.0, 1.0).is_nan());
        assert!(btdtria(0.0, 2.0, 0.0).is_nan());
        assert!(btdtria(1.0, 2.0, 0.0).is_nan());
        assert!(btdtria(0.5, 0.0, 0.3).is_nan());
        assert!(btdtria(0.5, -1.0, 0.3).is_nan());
        assert!(btdtria(-0.1, 2.0, 0.3).is_nan());
        assert!(btdtria(1.1, 2.0, 0.3).is_nan());
        assert!(btdtria(f64::NAN, 2.0, 0.3).is_nan());
        assert!(btdtria(0.5, f64::NAN, 0.3).is_nan());
        assert!(btdtria(0.5, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn btdtrib_inverse() {
        for &a in &[0.25, 0.5, 1.0, 2.0, 5.0] {
            for &b in &[0.5, 2.0, 5.0] {
                for &x in &[0.2, 0.3, 0.7] {
                    let p = btdtr(a, b, x);
                    let recovered = btdtrib(a, p, x);
                    assert!(
                        (recovered - b).abs() <= 1e-8 * b.abs().max(1.0),
                        "btdtrib/btdtr failed: a={a}, b={b}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn btdtrib_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (2.0, 0.5, 0.3, 4.246702175718102, 2e-12),
            (2.0, 0.8, 0.3, 7.924719144223477, 2e-11),
            (2.0, 0.2, 0.3, 1.8790372491805813, 2e-12),
            (0.5, 0.7, 0.2, 2.6396222094025554, 2e-12),
            (
                0.04395757565162919,
                0.9999999999999999,
                0.7611845908830768,
                21.60864146301538,
                2e-11,
            ),
        ];
        for &(a, p, x, expected, tolerance) in cases {
            let actual = btdtrib(a, p, x);
            assert!(
                (actual - expected).abs() < tolerance,
                "btdtrib({a}, {p}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn btdtrib_edges_match_scipy() {
        assert_eq!(btdtrib(2.0, 0.0, 0.3), f64::MIN_POSITIVE);
        assert!(btdtrib(2.0, 1.0, 0.3).is_infinite());
        assert_eq!(btdtrib(2.0, 0.0, 1.1), f64::MIN_POSITIVE);
        assert!(btdtrib(2.0, 1.0, 1.1).is_infinite());

        assert!(btdtrib(2.0, 0.5, 0.0).is_nan());
        assert!(btdtrib(2.0, 0.5, 1.0).is_nan());
        assert!(btdtrib(2.0, 0.0, 0.0).is_nan());
        assert!(btdtrib(2.0, 1.0, 0.0).is_nan());
        assert!(btdtrib(0.0, 0.5, 0.3).is_nan());
        assert!(btdtrib(-1.0, 0.5, 0.3).is_nan());
        assert!(btdtrib(2.0, -0.1, 0.3).is_nan());
        assert!(btdtrib(2.0, 1.1, 0.3).is_nan());
        assert!(btdtrib(f64::NAN, 0.5, 0.3).is_nan());
        assert!(btdtrib(2.0, f64::NAN, 0.3).is_nan());
        assert!(btdtrib(2.0, 0.5, f64::NAN).is_nan());
    }

    #[test]
    fn fdtrc_complement() {
        // fdtr + fdtrc should equal 1
        for &dfn in &[1.0, 5.0, 10.0] {
            for &dfd in &[1.0, 5.0, 10.0] {
                for &x in &[0.5, 1.0, 2.0, 5.0] {
                    let sum = fdtr(dfn, dfd, x) + fdtrc(dfn, dfd, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "fdtr({dfn}, {dfd}, {x}) + fdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn fdtr_metamorphic_boundary_zero_and_monotone() {
        // /testing-metamorphic for [frankenscipy-xybu7]:
        // fdtr at x=0 must be exactly 0; fdtr must be monotonically
        // non-decreasing in x for any (dfn, dfd) > 0.
        for &dfn in &[1.0_f64, 2.0, 5.0, 10.0] {
            for &dfd in &[1.0_f64, 2.0, 5.0, 10.0] {
                assert_eq!(fdtr(dfn, dfd, 0.0), 0.0, "fdtr({dfn},{dfd},0) must be 0");

                let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
                let mut prev = 0.0_f64;
                for &x in &xs {
                    let p = fdtr(dfn, dfd, x);
                    assert!(
                        p >= prev - 1e-12,
                        "fdtr({dfn}, {dfd}, {x}) = {p} < prev {prev} (not monotone)"
                    );
                    assert!(
                        (0.0..=1.0).contains(&p),
                        "fdtr({dfn}, {dfd}, {x}) = {p} outside [0, 1]"
                    );
                    prev = p;
                }
            }
        }
    }

    #[test]
    fn fdtr_dfn_equals_dfd_median_at_x_one() {
        // For F(d, d) the distribution is symmetric around 1 in the
        // sense that 1/F ~ F(d, d). Thus the median is exactly 1:
        //   fdtr(d, d, 1) = 0.5 for any d > 0.
        for &d in &[1.0_f64, 2.0, 5.0, 10.0, 50.0] {
            let p = fdtr(d, d, 1.0);
            assert!(
                (p - 0.5).abs() < 1e-10,
                "fdtr({d}, {d}, 1) = {p}, expected 0.5"
            );
        }
    }

    #[test]
    fn fdtri_roundtrip_recovers_x() {
        // fdtri(dfn, dfd, fdtr(dfn, dfd, x)) ≈ x for any positive x
        // and any (dfn, dfd) > 0.
        for &dfn in &[1.0_f64, 2.0, 5.0] {
            for &dfd in &[1.0_f64, 2.0, 5.0] {
                for &x in &[0.5_f64, 1.0, 2.0, 5.0] {
                    let p = fdtr(dfn, dfd, x);
                    let recovered = fdtri(dfn, dfd, p);
                    assert!(
                        (recovered - x).abs() < 1e-7,
                        "fdtri({dfn}, {dfd}, {p}={fdtr_val}) = {recovered}, expected {x}",
                        fdtr_val = p
                    );
                }
            }
        }
    }

    #[test]
    fn fdtridfd_inverse() {
        for &dfn in &[1.0, 2.0, 5.0, 10.0] {
            for &dfd in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
                for &x in &[1.5, 2.0, 5.0] {
                    let p = fdtr(dfn, dfd, x);
                    let dfd_recovered = fdtridfd(dfn, p, x);
                    assert!(
                        (dfd_recovered - dfd).abs() < 1e-8,
                        "fdtridfd/fdtr failed: dfn={dfn}, dfd={dfd}, x={x}, p={p}, dfd_recovered={dfd_recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn fdtridfd_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (5.0, 0.7, 1.5, 7.1205455518861855, 2e-10),
            (5.0, 0.5537887707581542, 1.5, 2.0, 2e-12),
            (5.0, 0.5, 1.5, 1.3789296276108034, 2e-12),
            (10.0, 0.8, 2.0, 6.2203600193018, 2e-10),
            (2.0, 0.9, 5.0, 3.3085663860076275, 2e-10),
        ];
        for &(dfn, p, x, expected, tolerance) in cases {
            let result = fdtridfd(dfn, p, x);
            assert!(
                (result - expected).abs() < tolerance,
                "fdtridfd({dfn}, {p}, {x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn fdtridfd_scipy_sentinels() {
        assert_eq!(fdtridfd(5.0, 0.0, 1.5), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.0, 0.0), 5.0);
        assert_eq!(fdtridfd(5.0, 0.1, 0.0), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.5, 0.0), 1.0e-100);
        assert_eq!(fdtridfd(5.0, 0.5000000001, 0.0), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.7, 0.0), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.9, 1.5), 1.0e100);
        assert_eq!(fdtridfd(5.0, 0.7, f64::INFINITY), 1.0e100);
        assert!(fdtridfd(5.0, 1.0, 1.5).is_nan());
        assert!(fdtridfd(f64::NAN, 0.7, 1.0).is_nan());
        assert!(fdtridfd(5.0, f64::NAN, 1.0).is_nan());
        assert!(fdtridfd(5.0, 0.7, f64::NAN).is_nan());
        assert!(fdtridfd(-1.0, 0.7, 1.0).is_nan());
        assert!(fdtridfd(5.0, -0.1, 1.0).is_nan());
        assert!(fdtridfd(5.0, 1.1, 1.0).is_nan());
        assert!(fdtridfd(5.0, 0.7, -1.0).is_nan());
    }

    #[test]
    fn stdtr_basic() {
        // stdtr(v, 0) = 0.5 for any v > 0 (symmetric around 0)
        assert!((stdtr(1.0, 0.0) - 0.5).abs() < 1e-10);
        assert!((stdtr(10.0, 0.0) - 0.5).abs() < 1e-10);
        assert!((stdtr(100.0, 0.0) - 0.5).abs() < 1e-10);

        // For large v, approaches normal distribution
        // stdtr(1000, 1.96) ≈ 0.975
        let result = stdtr(1000.0, 1.96);
        assert!(
            (result - 0.975).abs() < 0.01,
            "stdtr(1000, 1.96) = {result}, expected ~0.975"
        );

        // stdtr(1, t) = 0.5 + arctan(t)/π (Cauchy distribution)
        let t = 1.0_f64;
        let expected = 0.5 + t.atan() / std::f64::consts::PI;
        let result = stdtr(1.0, t);
        assert!(
            (result - expected).abs() < 0.001,
            "stdtr(1, 1) = {result}, expected {expected}"
        );
    }

    #[test]
    fn stdtr_symmetry() {
        // stdtr(v, -t) = 1 - stdtr(v, t)
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &t in &[0.5, 1.0, 2.0, 3.0] {
                let left = stdtr(v, -t);
                let right = 1.0 - stdtr(v, t);
                assert!(
                    (left - right).abs() < 1e-10,
                    "stdtr({v}, -{t}) = {left}, expected {right}"
                );
            }
        }
    }

    #[test]
    fn stdtrc_complement() {
        // stdtr + stdtrc should equal 1
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &t in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
                let sum = stdtr(v, t) + stdtrc(v, t);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "stdtr({v}, {t}) + stdtrc = {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn stdtrit_inverse() {
        // stdtrit should be inverse of stdtr
        for &v in &[1.0, 2.0, 5.0, 10.0, 30.0] {
            for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
                let t = stdtrit(v, p);
                let p_recovered = stdtr(v, t);
                assert!(
                    (p_recovered - p).abs() < 1e-8,
                    "stdtrit/stdtr failed: v={v}, p={p}, t={t}, p_recovered={p_recovered}"
                );
            }
        }
    }

    #[test]
    fn stdtrit_endpoints() {
        // SciPy returns +inf at both exact endpoints.
        assert!(stdtrit(5.0, 0.0).is_infinite() && stdtrit(5.0, 0.0).is_sign_positive());
        assert!(stdtrit(5.0, 1.0).is_infinite() && stdtrit(5.0, 1.0).is_sign_positive());
        assert!((stdtrit(5.0, 0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn stdtridf_inverse() {
        for &v in &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
            for &t in &[0.75, 1.0, 1.5, 2.0, 4.0] {
                let p = stdtr(v, t);
                let v_recovered = stdtridf(p, t);
                assert!(
                    (v_recovered - v).abs() < 1e-8,
                    "stdtridf/stdtr failed: v={v}, t={t}, p={p}, v_recovered={v_recovered}"
                );
            }
        }
    }

    #[test]
    fn stdtridf_reference_values() {
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.8, 1.25, 1.2176499408295116, 2e-12),
            (0.6, 1.0, 0.1321140237878431, 2e-12),
            (0.4, -1.0, 0.1321140237878431, 2e-12),
            (0.75, 1.0, 1.0000000000000093, 2e-12),
            (0.95, 2.0, 5.176135682782311, 2e-10),
        ];
        for &(p, t, expected, tolerance) in cases {
            let result = stdtridf(p, t);
            assert!(
                (result - expected).abs() < tolerance,
                "stdtridf({p}, {t}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn stdtridf_scipy_sentinels() {
        assert_eq!(stdtridf(0.4, 1.0), -1.0e100);
        assert_eq!(stdtridf(0.8, -1.0), -1.0e100);
        assert_eq!(stdtridf(0.8, 0.0), 1.0e10);
        assert_eq!(stdtridf(0.5, 0.0), 5.0);
        assert_eq!(stdtridf(0.5, 1.0), 5.0e-51);
        assert_eq!(stdtridf(1.0, 1.0), 1.0e10);
        assert!(stdtridf(f64::NAN, 1.0).is_nan());
        assert!(stdtridf(0.8, f64::NAN).is_nan());
        assert!(stdtridf(-0.1, 1.0).is_nan());
        assert!(stdtridf(1.1, 1.0).is_nan());
    }

    #[test]
    fn bdtr_basic() {
        // bdtr(0, 1, 0.5) = 0.5 (1 trial, k=0: P(X=0) = 1-p)
        let result = bdtr(0.0, 1.0, 0.5);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "bdtr(0, 1, 0.5) = {result}, expected 0.5"
        );

        // bdtr(0, 10, 0.5) = P(X=0) = 0.5^10 = 0.0009765625
        let result = bdtr(0.0, 10.0, 0.5);
        assert!(
            (result - 0.0009765625).abs() < 1e-8,
            "bdtr(0, 10, 0.5) = {result}, expected 0.0009765625"
        );

        // bdtr(5, 10, 0.5) = 0.623046875 (median of symmetric binomial)
        let result = bdtr(5.0, 10.0, 0.5);
        assert!(
            (result - 0.623046875).abs() < 1e-6,
            "bdtr(5, 10, 0.5) = {result}, expected 0.623046875"
        );
    }

    #[test]
    fn bdtr_bdtrc_complement() {
        // bdtr(k, n, p) + bdtrc(k, n, p) = 1
        for &n in &[5.0, 10.0, 20.0] {
            for &p in &[0.2, 0.5, 0.8] {
                for k in 0..=(n as i32) {
                    let kf = k as f64;
                    let sum = bdtr(kf, n, p) + bdtrc(kf, n, p);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "bdtr({kf}, {n}, {p}) + bdtrc = {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn bdtri_inverse() {
        // bdtri should be inverse of bdtr
        for &n in &[5.0, 10.0, 20.0] {
            for &k in &[1.0, 3.0, 5.0] {
                if k >= n {
                    continue;
                }
                for &p in &[0.2, 0.5, 0.8] {
                    let y = bdtr(k, n, p);
                    if y > 0.01 && y < 0.99 {
                        let p_recovered = bdtri(k, n, y);
                        assert!(
                            (p_recovered - p).abs() < 0.01,
                            "bdtri failed: k={k}, n={n}, p={p}, y={y}, p_recovered={p_recovered}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn binomial_inverse_shape_parameters_round_trip() {
        let y = bdtr(4.5, 10.0, 0.5);
        assert!((bdtrik(y, 10.0, 0.5) - 4.5).abs() < 1.0e-10);

        let y = bdtr(5.0, 11.0, 0.5);
        assert!((bdtrin(5.0, y, 0.5) - 11.0).abs() < 1.0e-9);

        assert!(bdtrik(-0.1, 10.0, 0.5).is_nan());
        assert!(bdtrin(5.0, 1.1, 0.5).is_nan());
    }

    #[test]
    fn binomial_inverse_shape_parameters_match_scipy_reference_values() {
        let cases: &[DistributionInverseCase] = &[
            (bdtrik, 0.5, 10.0, 0.5, 4.5),
            (bdtrik, 0.9, 10.0, 0.5, 6.517_443_355_854_331),
            (bdtrik, 0.5, 5.0, 0.2, 0.385_541_504_319_304_9),
            (bdtrin, 5.0, 0.5, 0.5, 10.999_999_999_998_705),
            (bdtrin, 5.0, 0.9, 0.5, 7.507_360_823_906_628),
            (bdtrin, 0.0, 0.000_976_562_5, 0.5, 10.000_000_000_015_802),
        ];
        for &(func, a, b, c, expected) in cases {
            let actual = func(a, b, c);
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "binomial inverse reference mismatch: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn nbdtr_basic() {
        // nbdtr(0, 1, 0.5) = P(0 failures before 1 success) = p = 0.5
        let result = nbdtr(0.0, 1.0, 0.5);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "nbdtr(0, 1, 0.5) = {result}, expected 0.5"
        );

        // nbdtr(0, 1, p) = p for all p (geometric: first trial is success)
        for &p in &[0.1, 0.5, 0.9] {
            let result = nbdtr(0.0, 1.0, p);
            assert!(
                (result - p).abs() < 1e-10,
                "nbdtr(0, 1, {p}) = {result}, expected {p}"
            );
        }

        // As k -> infinity, nbdtr(k, n, p) -> 1
        let result = nbdtr(100.0, 1.0, 0.5);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "nbdtr(100, 1, 0.5) = {result}, expected ~1.0"
        );
    }

    #[test]
    fn nbdtr_nbdtrc_complement() {
        // nbdtr(k, n, p) + nbdtrc(k, n, p) = 1
        for &n in &[1.0, 3.0, 5.0] {
            for &p in &[0.2, 0.5, 0.8] {
                for &k in &[0.0, 1.0, 5.0, 10.0] {
                    let sum = nbdtr(k, n, p) + nbdtrc(k, n, p);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "nbdtr({k}, {n}, {p}) + nbdtrc = {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn nbdtri_inverse() {
        // nbdtri should be inverse of nbdtr
        for &n in &[1.0, 3.0, 5.0] {
            for &k in &[0.0, 2.0, 5.0] {
                for &p in &[0.2, 0.5, 0.8] {
                    let y = nbdtr(k, n, p);
                    if y > 0.01 && y < 0.99 {
                        let p_recovered = nbdtri(k, n, y);
                        assert!(
                            (p_recovered - p).abs() < 0.01,
                            "nbdtri failed: k={k}, n={n}, p={p}, y={y}, p_recovered={p_recovered}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negative_binomial_inverse_shape_parameters_round_trip() {
        let y = nbdtr(9.0, 10.0, 0.5);
        assert!((nbdtrik(y, 10.0, 0.5) - 9.0).abs() < 1.0e-9);

        let y = nbdtr(5.0, 6.0, 0.5);
        assert!((nbdtrin(5.0, y, 0.5) - 6.0).abs() < 1.0e-9);

        assert!(nbdtrik(1.0, 10.0, 0.5).is_nan());
        assert!(nbdtrin(-1.0, 0.5, 0.5).is_nan());
    }

    #[test]
    fn negative_binomial_inverse_shape_parameters_match_scipy_reference_values() {
        let cases: &[DistributionInverseCase] = &[
            (nbdtrik, 0.5, 10.0, 0.5, 9.000_000_000_001_473),
            (nbdtrik, 0.9, 10.0, 0.5, 15.452_848_668_084_998),
            (nbdtrik, 0.5, 5.0, 0.2, 18.017_180_776_964_413),
            (nbdtrin, 5.0, 0.5, 0.5, 5.999_999_999_997_906),
            (nbdtrin, 5.0, 0.9, 0.5, 2.507_360_823_906_642_7),
            (nbdtrin, 0.0, 0.5, 0.5, 1.000_000_000_000_667_7),
        ];
        for &(func, a, b, c, expected) in cases {
            let actual = func(a, b, c);
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "negative-binomial inverse reference mismatch: got {actual}, expected {expected}"
            );
        }
    }

    fn complex_scalar(tensor: &SpecialTensor) -> Complex64 {
        match tensor {
            SpecialTensor::ComplexScalar(c) => *c,
            _ => panic!("expected ComplexScalar"),
        }
    }

    fn complex_vec(tensor: &SpecialTensor) -> &[Complex64] {
        match tensor {
            SpecialTensor::ComplexVec(values) => values,
            _ => panic!("expected ComplexVec"),
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual.re - expected.re).abs() < 1e-10,
            "real parts differ: actual={}, expected={}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < 1e-10,
            "imaginary parts differ: actual={}, expected={}",
            actual.im,
            expected.im
        );
    }

    #[test]
    fn complex_betaln_real_inputs_match_real_path() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_result = betaln(&real_a, &real_b, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-10,
            "complex betaln(2+0i, 3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_beta_real_inputs_match_real_path() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_result = beta(&real_a, &real_b, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-10,
            "complex beta(2+0i, 3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_betaln_with_imaginary_parts() {
        // Verify complex betaln produces finite results for complex inputs
        let a = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        assert!(c.re.is_finite(), "betaln result should be finite");
        assert!(c.im.is_finite(), "betaln result should be finite");
        // Verify symmetry: betaln(a, b) = betaln(b, a)
        let result_sym = betaln(&b, &a, RuntimeMode::Strict).unwrap();
        let c_sym = complex_scalar(&result_sym);
        assert!(
            (c.re - c_sym.re).abs() < 1e-10,
            "betaln should be symmetric"
        );
        assert!(
            (c.im - c_sym.im).abs() < 1e-10,
            "betaln should be symmetric"
        );
    }

    #[test]
    fn complex_beta_with_imaginary_parts() {
        // Verify complex beta produces finite results
        let a = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        assert!(c.re.is_finite(), "beta result should be finite");
        assert!(c.im.is_finite(), "beta result should be finite");
        // beta = exp(betaln)
        let ln_result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let ln_c = complex_scalar(&ln_result);
        let expected = ln_c.exp();
        assert!(
            (c.re - expected.re).abs() < 1e-10,
            "beta should equal exp(betaln)"
        );
        assert!(
            (c.im - expected.im).abs() < 1e-10,
            "beta should equal exp(betaln)"
        );
    }

    #[test]
    fn complex_betaln_vector() {
        let a = SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)]);
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        match result {
            SpecialTensor::ComplexVec(v) => {
                assert_eq!(v.len(), 2);
                // first entry (real inputs) should have near-zero imaginary part
                assert!(v[0].im.abs() < 1e-10, "real inputs should have im~0");
                assert!(v[0].re.is_finite());
                assert!(v[1].re.is_finite());
                assert!(v[1].im.is_finite());
            }
            _ => panic!("expected ComplexVec"),
        }
    }

    #[test]
    fn beta_vector_mismatch_returns_domain_error() {
        let err = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched beta vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn betaln_complex_vector_mismatch_returns_domain_error() {
        let err = betaln(
            &SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]),
            &SpecialTensor::ComplexVec(vec![Complex64::new(1.0, 0.0)]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched betaln vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn complex_beta_mixed_real_complex() {
        let a = SpecialTensor::RealScalar(2.0);
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.5));
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        // Should produce a complex result
        assert!(c.re.is_finite());
        assert!(c.im.is_finite());
    }

    #[test]
    fn complex_betainc_real_inputs_match_real() {
        // Complex betainc with real inputs should match real path
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.0));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);

        let real_a = SpecialTensor::RealScalar(2.0);
        let real_b = SpecialTensor::RealScalar(3.0);
        let real_x = SpecialTensor::RealScalar(0.5);
        let real_result = betainc(&real_a, &real_b, &real_x, RuntimeMode::Strict).unwrap();
        let expected = match real_result {
            SpecialTensor::RealScalar(v) => v,
            _ => panic!("expected RealScalar"),
        };

        assert!(
            (c.re - expected).abs() < 1e-8,
            "complex betainc(2,3,0.5) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(
            c.im.abs() < 1e-10,
            "imaginary part should be ~0, got {}",
            c.im
        );
    }

    #[test]
    fn complex_betainc_endpoints() {
        // betainc(a, b, 0) = 0
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, 0.0));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.0, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        assert!(c.re.abs() < 1e-10, "betainc(a,b,0) should be 0");
        assert!(c.im.abs() < 1e-10);

        // betainc(a, b, 1) = 1
        let x1 = SpecialTensor::ComplexScalar(Complex64::new(1.0, 0.0));
        let result1 = betainc(&a, &b, &x1, RuntimeMode::Strict).unwrap();
        let c1 = complex_scalar(&result1);
        assert!(
            (c1.re - 1.0).abs() < 1e-10,
            "betainc(a,b,1) should be 1, got {}",
            c1.re
        );
        assert!(c1.im.abs() < 1e-10);
    }

    #[test]
    fn complex_betainc_with_complex_args() {
        // Just verify it produces finite results
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.0, 0.5));
        let b = SpecialTensor::ComplexScalar(Complex64::new(3.0, -0.3));
        let x = SpecialTensor::ComplexScalar(Complex64::new(0.5, 0.0));
        let result = betainc(&a, &b, &x, RuntimeMode::Strict).unwrap();
        let c = complex_scalar(&result);
        assert!(c.re.is_finite(), "betainc should produce finite result");
        assert!(c.im.is_finite(), "betainc should produce finite result");
    }

    #[test]
    fn beta_real_scalar_complex_vector_broadcasts() {
        let a = SpecialTensor::RealScalar(2.0);
        let b = SpecialTensor::ComplexVec(vec![Complex64::new(3.0, 0.0), Complex64::new(1.5, 0.5)]);
        let result = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let values = complex_vec(&result);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            complex_beta_scalar(Complex64::from_real(2.0), Complex64::new(3.0, 0.0)),
        );
        assert_complex_close(
            values[1],
            complex_beta_scalar(Complex64::from_real(2.0), Complex64::new(1.5, 0.5)),
        );
    }

    #[test]
    fn betaln_complex_scalar_real_vector_broadcasts() {
        let a = SpecialTensor::ComplexScalar(Complex64::new(2.5, 0.25));
        let b = SpecialTensor::RealVec(vec![1.0, 2.0]);
        let result = betaln(&a, &b, RuntimeMode::Strict).unwrap();
        let values = complex_vec(&result);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            complex_betaln_scalar(Complex64::new(2.5, 0.25), Complex64::from_real(1.0)),
        );
        assert_complex_close(
            values[1],
            complex_betaln_scalar(Complex64::new(2.5, 0.25), Complex64::from_real(2.0)),
        );
    }

    #[test]
    fn beta_mixed_real_complex_vectors_preserve_symmetry() {
        let a = SpecialTensor::RealVec(vec![2.0, 3.0]);
        let b =
            SpecialTensor::ComplexVec(vec![Complex64::new(1.5, 0.5), Complex64::new(2.5, -0.25)]);
        let forward = beta(&a, &b, RuntimeMode::Strict).unwrap();
        let reverse = beta(&b, &a, RuntimeMode::Strict).unwrap();
        let forward_values = complex_vec(&forward);
        let reverse_values = complex_vec(&reverse);

        assert_eq!(forward_values.len(), 2);
        assert_eq!(reverse_values.len(), 2);
        for (forward_value, reverse_value) in forward_values.iter().zip(reverse_values.iter()) {
            assert_complex_close(*forward_value, *reverse_value);
        }
    }

    #[test]
    fn betaln_mixed_vector_mismatch_returns_domain_error() {
        let err = betaln(
            &SpecialTensor::RealVec(vec![2.0, 3.0]),
            &SpecialTensor::ComplexVec(vec![Complex64::new(1.0, 0.0)]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched mixed betaln vectors should fail");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn btdtr_matches_scipy_reference_values() {
        // scipy.special.btdtr(2, 3, 0.5) - beta distribution CDF
        // B(2,3) at x=0.5: I_0.5(2,3) = 0.6875
        let result = btdtr(2.0, 3.0, 0.5);
        assert!(
            (result - 0.6875).abs() < 1e-10,
            "btdtr(2,3,0.5) got {result}, expected 0.6875"
        );
    }

    #[test]
    fn betaln_negative_args_match_scipy() {
        // scipy.special.betaln returns finite values for negative non-integer
        // arguments (= gammaln(a)+gammaln(b)-gammaln(a+b)); we previously
        // fail-closed to NaN for any nonpositive parameter.
        let cases = [
            (-2.5, 3.0, 0.06453852113757116_f64),
            (2.0, -3.5, -2.169053700369523),
            (-4.3, -1.2, 3.814037380330781),
            (0.3, -0.7, 1.2337463436314935),
            (-1.5, 2.5, 1.1447298858494004),
            (5.0, -4.5, -0.2073951943460706),
        ];
        for (a, b, want) in cases {
            let got = betaln_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "betaln({a},{b}) got {got}, want {want}"
            );
        }
        // Pole limits also match SciPy: a+b a nonpositive integer => -inf.
        assert_eq!(
            betaln_scalar(-0.5, 0.5, RuntimeMode::Strict).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            betaln_scalar(-2.5, -2.5, RuntimeMode::Strict).unwrap(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn beta_negative_args_match_scipy_signed() {
        // scipy.special.beta is SIGNED for negative args (B = Γ(a)Γ(b)/Γ(a+b)),
        // not just |B|; the gamma-sign factor restores it.
        let cases = [
            (-2.5, 3.0, -1.0666666666666667_f64),
            (2.0, -3.5, 0.11428571428571427),
            (-4.3, -1.2, -45.33309684186555),
            (-1.5, 2.5, 3.1415926535897936),
            (0.3, -0.7, 3.4340706764177225),
            (3.0, 4.0, 0.016666666666666666), // positive args: sign=+1, unchanged
        ];
        for (a, b, want) in cases {
            let got = beta_scalar(a, b, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "beta({a},{b}) got {got}, want {want}"
            );
        }
    }

    #[test]
    fn beta_nonpositive_integer_args_match_scipy() {
        // Regression (frankenscipy-dwd3d): for a nonpositive-integer argument with
        // a positive partner the Γ poles either cancel to a finite rational
        // (fsci previously got NaN from inf−inf) or give scipy's +inf (fsci
        // previously got −inf from the wrong gamma-sign side). Values from
        // scipy.special.beta / betaln 1.17.1.
        let beta_cases = [
            (2.0_f64, -1.0_f64, f64::INFINITY),
            (-1.0, 2.0, f64::INFINITY),
            (3.0, -2.0, f64::INFINITY),
            (0.5, -1.0, f64::INFINITY),
            (0.0, 2.0, f64::INFINITY),
            (2.0, -2.0, 0.5),
            (2.0, -3.0, 1.0 / 6.0),
            (3.0, -3.0, -1.0 / 3.0),
            (1.0, -1.0, -1.0),
            (1.0, -4.0, -0.25),
        ];
        for (a, b, want) in beta_cases {
            let got = beta_scalar(a, b, RuntimeMode::Strict).unwrap();
            if want.is_infinite() {
                assert!(
                    got.is_infinite() && got.is_sign_positive(),
                    "beta({a},{b}) = {got}, want +inf"
                );
            } else {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "beta({a},{b}) = {got}, want {want}"
                );
            }
        }
        // betaln tracks ln|beta|: betaln(2,-2)=ln(0.5), betaln(2,-1)=+inf.
        assert!(
            (betaln_scalar(2.0, -2.0, RuntimeMode::Strict).unwrap() - (0.5_f64).ln()).abs() < 1e-12
        );
        assert!(
            betaln_scalar(2.0, -1.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
    }

    #[test]
    fn bdtr_matches_scipy_reference_values() {
        // scipy.special.bdtr(5, 10, 0.5) - binomial distribution CDF
        // P(X <= 5) for X ~ Binom(10, 0.5)
        let result = bdtr(5.0, 10.0, 0.5);
        let expected = 0.623046875;
        assert!(
            (result - expected).abs() < 1e-10,
            "bdtr(5,10,0.5) got {result}, expected {expected}"
        );
    }

    #[test]
    fn stdtr_matches_scipy_reference_values() {
        // scipy.special.stdtr(df, t) - Student's t CDF
        // stdtr(5, 0) = 0.5 (symmetric at 0)
        // stdtr(5, 1.0) ≈ 0.8183 (df=5, t=1)
        let result0 = stdtr(5.0, 0.0);
        assert!(
            (result0 - 0.5).abs() < 1e-10,
            "stdtr(5, 0) got {result0}, expected 0.5"
        );

        let result1 = stdtr(5.0, 1.0);
        assert!(
            (result1 - 0.8183).abs() < 1e-3,
            "stdtr(5, 1) got {result1}, expected ~0.8183"
        );
    }

    #[test]
    fn fdtr_matches_scipy_reference_values() {
        // scipy.special.fdtr(dfn, dfd, x) - F distribution CDF
        // fdtr(5, 10, 1.0) ≈ 0.5348
        let result = fdtr(5.0, 10.0, 1.0);
        assert!(
            (result - 0.5349).abs() < 1e-3,
            "fdtr(5, 10, 1) got {result}, expected ~0.5349"
        );

        // fdtr at 0 should be 0
        let result0 = fdtr(5.0, 10.0, 0.0);
        assert!(
            result0.abs() < 1e-10,
            "fdtr(5, 10, 0) got {result0}, expected 0"
        );
    }

    #[test]
    fn nctdtr_matches_scipy_reference_values() {
        // frankenscipy: non-central t CDF was missing. Golden values from
        // scipy.special.nctdtr(df, nc, t) 1.17.1.
        let cases = [
            (10.0_f64, 2.0, 1.5, 0.3047854473760421_f64),
            (5.0, 0.0, 1.0, 0.8183912661754386), // nc=0 → central t
            (20.0, -3.0, -2.0, 0.8358989270421169), // negative nc and t (reflection)
            (8.0, 5.0, 4.0, 0.21027058165197615),
            (15.0, 1.0, 0.0, 0.15865525393145707), // t=0 → Phi(-nc)
        ];
        for (df, nc, t, want) in cases {
            let got = nctdtr(df, nc, t);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "nctdtr({df},{nc},{t}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn noncentral_quantile_inverses_match_scipy() {
        // frankenscipy: golden from scipy.special.ncfdtri / nctdtrit 1.17.1.
        let nc_f = [
            (5.0_f64, 10.0, 3.0, 0.5, 1.5254092911626238_f64),
            (2.0, 4.0, 0.0, 0.7, 1.6514837167011074),
            (10.0, 20.0, 40.0, 0.3, 4.058092482878802),
        ];
        for (dfn, dfd, nc, p, want) in nc_f {
            let got = ncfdtri(dfn, dfd, nc, p);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "ncfdtri = {got}, want {want}"
            );
        }
        let nc_t = [
            (10.0_f64, 2.0, 0.3, 1.4856759815279506_f64),
            (20.0, -3.0, 0.8, -2.138962456180749),
            (5.0, 0.0, 0.5, 0.0),
        ];
        for (df, nc, p, want) in nc_t {
            let got = nctdtrit(df, nc, p);
            assert!(
                (got - want).abs() <= 1e-8 * want.abs().max(1e-6),
                "nctdtrit = {got}, want {want}"
            );
        }
        assert_eq!(ncfdtri(5.0, 10.0, 3.0, 0.0), 0.0);
        assert!(ncfdtri(5.0, 10.0, 3.0, 1.0).is_infinite());
    }

    #[test]
    fn ncfdtr_matches_scipy_reference_values() {
        // frankenscipy: non-central F CDF was missing. Golden values from
        // scipy.special.ncfdtr(dfn, dfd, nc, f) 1.17.1.
        let cases = [
            (5.0_f64, 10.0, 3.0, 2.0, 0.6391470579975839_f64),
            (2.0, 4.0, 0.0, 1.5, 0.673469387755102), // nc=0 → central F
            (10.0, 20.0, 40.0, 3.0, 0.10716411882720595), // large nc
            (3.0, 3.0, 5.0, 0.5, 0.0625950844013485),
        ];
        for (dfn, dfd, nc, f, want) in cases {
            let got = ncfdtr(dfn, dfd, nc, f);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "ncfdtr({dfn},{dfd},{nc},{f}) = {got}, expected {want}"
            );
        }
        assert_eq!(ncfdtr(5.0, 10.0, 3.0, 0.0), 0.0);
    }

    #[test]
    fn fdtri_matches_scipy_reference_values() {
        // scipy.special.fdtri(5, 10, 0.5) ≈ 0.931933160851048
        let result = fdtri(5.0, 10.0, 0.5);
        assert!(
            (result - 0.931933160851048).abs() < 1e-6,
            "fdtri(5, 10, 0.5) got {result}, expected 0.931933160851048"
        );
    }

    #[test]
    fn stdtrit_matches_scipy_reference_values() {
        // scipy.special.stdtrit(10, 0.95) ≈ 1.8124611228116756
        let result = stdtrit(10.0, 0.95);
        assert!(
            (result - 1.8124611228116756).abs() < 1e-6,
            "stdtrit(10, 0.95) got {result}, expected 1.8124611228116756"
        );
    }
}
