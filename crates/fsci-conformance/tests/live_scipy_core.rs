#![forbid(unsafe_code)]
//! Live SciPy differential tests for core FrankenSciPy surfaces.
//!
//! This file intentionally uses a subprocess oracle instead of frozen
//! constants: every case is evaluated by both the Rust implementation and the
//! local `scipy` install from the same serialized inputs. Comparisons use an
//! explicit ULP budget plus a small absolute floor for near-zero values.

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use fsci_linalg::{solve, SolveOptions};
use fsci_runtime::RuntimeMode;
use fsci_signal::lfilter;
use fsci_special::{
    beta, betainc, betaln, gamma, gammainc, gammaincc, gammaln, rgamma, SpecialResult,
    SpecialTensor,
};
use fsci_stats::{ChiSquared, ContinuousDistribution, Normal, StudentT};
use serde::de::DeserializeOwned;
use serde::Serialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Copy)]
struct ULPPolicy {
    max_ulps: u64,
    abs_floor: f64,
}

#[derive(Debug, Clone, Copy)]
struct FloatDiff {
    abs: f64,
    ulps: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VectorOracle {
    case_id: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ScalarOracle {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct LfilterCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SpecialCase {
    case_id: String,
    func: String,
    args: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct StatsCase {
    case_id: String,
    dist: String,
    params: Vec<f64>,
    method: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SolveCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

fn scipy_available_or_skip(test_id: &str) -> Result<bool, String> {
    let available = Command::new("python3")
        .arg("-c")
        .arg("import scipy")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if available {
        return Ok(true);
    }

    if std::env::var_os(REQUIRE_SCIPY_ENV).is_some() {
        return Err(format!(
            "{REQUIRE_SCIPY_ENV}=1 but SciPy is unavailable for {test_id}"
        ));
    }
    Ok(false)
}

fn run_python_oracle<I, O>(script: &str, input: &I) -> Result<O, String>
where
    I: Serialize,
    O: DeserializeOwned,
{
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| format!("spawn Python oracle: {err}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "open Python oracle stdin".to_owned())?;
        let input_json =
            serde_json::to_vec(input).map_err(|err| format!("serialize oracle input: {err}"))?;
        stdin
            .write_all(&input_json)
            .map_err(|err| format!("write oracle input: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for Python oracle: {err}"))?;
    if !output.status.success() {
        return Err(format!("Python oracle exited with {}", output.status));
    }
    serde_json::from_slice(&output.stdout).map_err(|err| format!("parse oracle output: {err}"))
}

fn ordered_float_bits(value: f64) -> i64 {
    let bits = value.to_bits() as i64;
    if bits < 0 {
        i64::MIN - bits
    } else {
        bits
    }
}

fn ulp_distance(left: f64, right: f64) -> u64 {
    if left == right {
        return 0;
    }
    let left_ordered = i128::from(ordered_float_bits(left));
    let right_ordered = i128::from(ordered_float_bits(right));
    left_ordered.abs_diff(right_ordered) as u64
}

fn float_diff(actual: f64, expected: f64) -> FloatDiff {
    FloatDiff {
        abs: (actual - expected).abs(),
        ulps: ulp_distance(actual, expected),
    }
}

fn assert_ulp_close(
    label: &str,
    actual: f64,
    expected: f64,
    policy: ULPPolicy,
) -> Result<FloatDiff, String> {
    if actual == expected {
        return Ok(FloatDiff { abs: 0.0, ulps: 0 });
    }
    if !actual.is_finite() || !expected.is_finite() {
        return Err(format!(
            "{label}: non-finite mismatch, rust={actual:?}, scipy={expected:?}"
        ));
    }
    let diff = float_diff(actual, expected);
    if diff.abs > policy.abs_floor && diff.ulps > policy.max_ulps {
        return Err(format!(
            "{label}: rust={actual:.17e}, scipy={expected:.17e}, abs={:.3e}, ulps={}, max_ulps={}, abs_floor={:.3e}",
            diff.abs, diff.ulps, policy.max_ulps, policy.abs_floor
        ));
    }
    Ok(diff)
}

fn oracle_map<T, F>(
    test_id: &str,
    cases: &[String],
    results: Vec<T>,
    case_id: F,
) -> Result<HashMap<String, T>, String>
where
    F: Fn(&T) -> &str,
{
    let mut map = HashMap::with_capacity(results.len());
    for result in results {
        let previous = map.insert(case_id(&result).to_owned(), result);
        if previous.is_some() {
            return Err(format!("{test_id}: duplicate SciPy oracle case"));
        }
    }

    if map.len() != cases.len() {
        return Err(format!(
            "{test_id}: SciPy oracle returned {} results for {} cases",
            map.len(),
            cases.len()
        ));
    }
    for case in cases {
        if !map.contains_key(case) {
            return Err(format!("{test_id}: missing SciPy oracle case {case}"));
        }
    }
    Ok(map)
}

fn scalar(value: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(value)
}

fn real_scalar(result: SpecialResult, label: &str) -> Result<f64, String> {
    match result {
        Ok(SpecialTensor::RealScalar(value)) => Ok(value),
        Ok(other) => Err(format!("{label}: expected real scalar, got {other:?}")),
        Err(err) => Err(format!("{label}: fsci_special error: {err}")),
    }
}

fn lfilter_cases() -> Vec<LfilterCase> {
    vec![
        LfilterCase {
            case_id: "fir_smoothing".to_owned(),
            b: vec![0.25, 0.5, 0.25],
            a: vec![1.0],
            x: vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        },
        LfilterCase {
            case_id: "iir_first_order".to_owned(),
            b: vec![0.2, 0.1],
            a: vec![1.0, -0.7],
            x: vec![1.0, -1.0, 0.5, 0.25, -0.125, 0.0, 2.0],
        },
        LfilterCase {
            case_id: "normalizes_non_unit_a0".to_owned(),
            b: vec![0.5, -0.25, 0.125],
            a: vec![2.0, -0.5, 0.25],
            x: vec![3.0, 1.0, -2.0, 4.0, 0.5, -0.25],
        },
    ]
}

fn special_cases() -> Vec<SpecialCase> {
    let mut cases = Vec::new();
    for x in [0.5, 1.0, 2.5, 5.0, 8.25] {
        cases.push(SpecialCase {
            case_id: format!("gamma_{x}"),
            func: "gamma".to_owned(),
            args: vec![x],
        });
        cases.push(SpecialCase {
            case_id: format!("gammaln_{x}"),
            func: "gammaln".to_owned(),
            args: vec![x],
        });
        cases.push(SpecialCase {
            case_id: format!("rgamma_{x}"),
            func: "rgamma".to_owned(),
            args: vec![x],
        });
    }

    for (a, b) in [(0.5, 0.5), (2.0, 3.0), (5.5, 1.25), (10.0, 4.0)] {
        cases.push(SpecialCase {
            case_id: format!("beta_{a}_{b}"),
            func: "beta".to_owned(),
            args: vec![a, b],
        });
        cases.push(SpecialCase {
            case_id: format!("betaln_{a}_{b}"),
            func: "betaln".to_owned(),
            args: vec![a, b],
        });
    }

    for (a, b, x) in [(0.5, 0.5, 0.25), (2.0, 3.0, 0.4), (5.0, 2.0, 0.75)] {
        cases.push(SpecialCase {
            case_id: format!("betainc_{a}_{b}_{x}"),
            func: "betainc".to_owned(),
            args: vec![a, b, x],
        });
    }

    for (a, x) in [(0.5, 0.25), (2.0, 1.5), (5.0, 7.0)] {
        cases.push(SpecialCase {
            case_id: format!("gammainc_{a}_{x}"),
            func: "gammainc".to_owned(),
            args: vec![a, x],
        });
        cases.push(SpecialCase {
            case_id: format!("gammaincc_{a}_{x}"),
            func: "gammaincc".to_owned(),
            args: vec![a, x],
        });
    }
    cases
}

fn stats_cases() -> Vec<StatsCase> {
    let mut cases = Vec::new();
    for (loc, scale) in [(0.0, 1.0), (2.0, 0.5)] {
        for x in [-1.0, 0.0, 0.75, 2.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("norm_{loc}_{scale}_{method}_{x}"),
                    dist: "norm".to_owned(),
                    params: vec![loc, scale],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("norm_{loc}_{scale}_ppf_{q}"),
                dist: "norm".to_owned(),
                params: vec![loc, scale],
                method: "ppf".to_owned(),
                value: q,
            });
        }
    }

    for df in [1.0, 2.5, 8.0] {
        for x in [-1.5, 0.0, 2.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("t_{df}_{method}_{x}"),
                    dist: "t".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("t_{df}_ppf_{q}"),
                dist: "t".to_owned(),
                params: vec![df],
                method: "ppf".to_owned(),
                value: q,
            });
        }
    }

    for df in [1.0, 2.0, 5.0] {
        for x in [0.25, 1.0, 4.0, 8.0] {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(StatsCase {
                    case_id: format!("chi2_{df}_{method}_{x}"),
                    dist: "chi2".to_owned(),
                    params: vec![df],
                    method: method.to_owned(),
                    value: x,
                });
            }
        }
        for q in [0.1, 0.5, 0.9] {
            cases.push(StatsCase {
                case_id: format!("chi2_{df}_ppf_{q}"),
                dist: "chi2".to_owned(),
                params: vec![df],
                method: "ppf".to_owned(),
                value: q,
            });
        }
    }
    cases
}

fn solve_cases() -> Vec<SolveCase> {
    vec![
        SolveCase {
            case_id: "solve_2x2_general".to_owned(),
            a: vec![vec![3.0, 1.0], vec![1.0, 2.0]],
            b: vec![9.0, 8.0],
        },
        SolveCase {
            case_id: "solve_3x3_diag_dominant".to_owned(),
            a: vec![
                vec![4.0, 0.5, -0.25],
                vec![0.25, 3.0, 0.75],
                vec![-0.5, 0.25, 2.5],
            ],
            b: vec![1.0, -2.0, 3.0],
        },
        SolveCase {
            case_id: "solve_scaled_4x4".to_owned(),
            a: vec![
                vec![10.0, 2.0, 0.0, 1.0],
                vec![1.0, 8.0, -1.0, 0.5],
                vec![0.5, -2.0, 7.0, 1.5],
                vec![0.0, 1.0, 2.0, 6.0],
            ],
            b: vec![1.0, 0.5, -1.0, 2.0],
        },
    ]
}

#[test]
fn live_scipy_signal_lfilter_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_signal_lfilter_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = lfilter_cases();
    let script = r#"
import json
import sys
from scipy import signal

cases = json.load(sys.stdin)
results = []
for case in cases:
    y = signal.lfilter(case["b"], case["a"], case["x"])
    results.append({"case_id": case["case_id"], "values": [float(v) for v in y]})
print(json.dumps(results))
"#;
    let oracle_results: Vec<VectorOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 64,
        abs_floor: 1.0e-13,
    };

    for case in &cases {
        let rust = lfilter(&case.b, &case.a, &case.x, None)
            .map_err(|err| format!("{}: Rust lfilter failed: {err}", case.case_id))?;
        let scipy = &oracle[&case.case_id].values;
        if rust.len() != scipy.len() {
            return Err(format!("{}: output length diverged", case.case_id));
        }
        for (idx, (&actual, &expected)) in rust.iter().zip(scipy).enumerate() {
            assert_ulp_close(
                &format!("{}[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
    }
    Ok(())
}

#[test]
fn live_scipy_special_gamma_beta_family_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_special_gamma_beta_family_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = special_cases();
    let script = r#"
import json
import sys
from scipy import special

cases = json.load(sys.stdin)
results = []
for case in cases:
    func = case["func"]
    args = [float(v) for v in case["args"]]
    if func == "gamma":
        value = special.gamma(args[0])
    elif func == "gammaln":
        value = special.gammaln(args[0])
    elif func == "rgamma":
        value = special.rgamma(args[0]) if hasattr(special, "rgamma") else 1.0 / special.gamma(args[0])
    elif func == "beta":
        value = special.beta(args[0], args[1])
    elif func == "betaln":
        value = special.betaln(args[0], args[1])
    elif func == "betainc":
        value = special.betainc(args[0], args[1], args[2])
    elif func == "gammainc":
        value = special.gammainc(args[0], args[1])
    elif func == "gammaincc":
        value = special.gammaincc(args[0], args[1])
    else:
        raise ValueError(func)
    results.append({"case_id": case["case_id"], "value": float(value)})
print(json.dumps(results))
"#;
    let oracle_results: Vec<ScalarOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 50_000_000,
        abs_floor: 1.0e-10,
    };

    for case in &cases {
        let actual = match case.func.as_str() {
            "gamma" => real_scalar(
                gamma(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "gammaln" => real_scalar(
                gammaln(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "rgamma" => real_scalar(
                rgamma(&scalar(case.args[0]), RuntimeMode::Strict),
                &case.case_id,
            )?,
            "beta" => real_scalar(
                beta(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "betaln" => real_scalar(
                betaln(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "betainc" => real_scalar(
                betainc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    &scalar(case.args[2]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "gammainc" => real_scalar(
                gammainc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            "gammaincc" => real_scalar(
                gammaincc(
                    &scalar(case.args[0]),
                    &scalar(case.args[1]),
                    RuntimeMode::Strict,
                ),
                &case.case_id,
            )?,
            other => return Err(format!("unsupported special case func {other}")),
        };
        assert_ulp_close(&case.case_id, actual, oracle[&case.case_id].value, policy)?;
    }
    Ok(())
}

#[test]
fn live_scipy_stats_norm_t_chi2_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_stats_norm_t_chi2_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = stats_cases();
    let script = r#"
import json
import sys
from scipy import stats

cases = json.load(sys.stdin)
results = []
for case in cases:
    dist = case["dist"]
    method = case["method"]
    value = float(case["value"])
    params = [float(v) for v in case["params"]]
    if dist == "norm":
        rv = stats.norm(loc=params[0], scale=params[1])
    elif dist == "t":
        rv = stats.t(df=params[0])
    elif dist == "chi2":
        rv = stats.chi2(df=params[0])
    else:
        raise ValueError(dist)
    result = getattr(rv, method)(value)
    results.append({"case_id": case["case_id"], "value": float(result)})
print(json.dumps(results))
"#;
    let oracle_results: Vec<ScalarOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 5_000_000_000,
        abs_floor: 1.0e-7,
    };

    for case in &cases {
        let actual = match case.dist.as_str() {
            "norm" => {
                let dist = Normal::new(case.params[0], case.params[1]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "sf" => dist.sf(case.value),
                    "ppf" => dist.ppf(case.value),
                    other => return Err(format!("unsupported norm method {other}")),
                }
            }
            "t" => {
                let dist = StudentT::new(case.params[0]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "sf" => dist.sf(case.value),
                    "ppf" => dist.ppf(case.value),
                    other => return Err(format!("unsupported t method {other}")),
                }
            }
            "chi2" => {
                let dist = ChiSquared::new(case.params[0]);
                match case.method.as_str() {
                    "pdf" => dist.pdf(case.value),
                    "cdf" => dist.cdf(case.value),
                    "sf" => dist.sf(case.value),
                    "ppf" => dist.ppf(case.value),
                    other => return Err(format!("unsupported chi2 method {other}")),
                }
            }
            other => return Err(format!("unsupported stats distribution {other}")),
        };
        assert_ulp_close(&case.case_id, actual, oracle[&case.case_id].value, policy)?;
    }
    Ok(())
}

#[test]
fn live_scipy_linalg_solve_ulp_conformance() -> Result<(), String> {
    let test_id = "live_scipy_linalg_solve_ulp_conformance";
    if !scipy_available_or_skip(test_id)? {
        return Ok(());
    }

    let cases = solve_cases();
    let script = r#"
import json
import sys
import numpy as np
from scipy import linalg

cases = json.load(sys.stdin)
results = []
for case in cases:
    a = np.array(case["a"], dtype=np.float64)
    b = np.array(case["b"], dtype=np.float64)
    x = linalg.solve(a, b)
    results.append({"case_id": case["case_id"], "values": [float(v) for v in x]})
print(json.dumps(results))
"#;
    let oracle_results: Vec<VectorOracle> = run_python_oracle(script, &cases)?;
    let case_ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
    let oracle = oracle_map(test_id, &case_ids, oracle_results, |result| &result.case_id)?;
    let policy = ULPPolicy {
        max_ulps: 1_000_000,
        abs_floor: 1.0e-11,
    };

    for case in &cases {
        let rust = solve(
            &case.a,
            &case.b,
            SolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: true,
                ..SolveOptions::default()
            },
        )
        .map_err(|err| format!("{}: Rust solve failed: {err}", case.case_id))?;
        let scipy = &oracle[&case.case_id].values;
        if rust.x.len() != scipy.len() {
            return Err(format!("{}: solution length diverged", case.case_id));
        }
        for (idx, (&actual, &expected)) in rust.x.iter().zip(scipy).enumerate() {
            assert_ulp_close(
                &format!("{}[{idx}]", case.case_id),
                actual,
                expected,
                policy,
            )?;
        }
    }
    Ok(())
}
