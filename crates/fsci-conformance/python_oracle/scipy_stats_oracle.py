#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy stats packet fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [_as_float(value) for value in values]


def _scalar_case(case_id: str, value: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": "scalar",
        "result": {"value": _as_float(value)},
        "error": None,
    }


def _run_case(case: dict[str, Any], stats: Any) -> dict[str, Any]:
    case_id = case["case_id"]
    function_name = case["function"]
    args = case.get("args", [])

    try:
        if function_name == "describe":
            result = stats.describe(*args)
            payload = {
                "nobs": int(result.nobs),
                "minmax": [_as_float(result.minmax[0]), _as_float(result.minmax[1])],
                "mean": _as_float(result.mean),
                "variance": _as_float(result.variance),
                "skewness": _as_float(result.skewness),
                "kurtosis": _as_float(result.kurtosis),
            }
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "describe_result",
                "result": payload,
                "error": None,
            }
        if function_name == "skew":
            return _scalar_case(case_id, stats.skew(*args))
        if function_name == "kurtosis":
            return _scalar_case(case_id, stats.kurtosis(*args))
        if function_name == "pearsonr":
            result = stats.pearsonr(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "correlation_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }
        if function_name == "spearmanr":
            result = stats.spearmanr(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "correlation_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }
        if function_name == "linregress":
            result = stats.linregress(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "linregress_result",
                "result": {
                    "slope": _as_float(result.slope),
                    "intercept": _as_float(result.intercept),
                    "rvalue": _as_float(result.rvalue),
                    "pvalue": _as_float(result.pvalue),
                    "stderr": _as_float(result.stderr),
                },
                "error": None,
            }
        if function_name == "ttest_1samp":
            result = stats.ttest_1samp(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "ttest_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }
        if function_name in ("ttest_ind", "mannwhitneyu", "wilcoxon"):
            # br-7k5n: 2-sample location tests share the (statistic,
            # pvalue) shape with ttest_1samp.
            fn = getattr(stats, function_name)
            result = fn(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "ttest_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }
        if function_name == "ks_2samp":
            result = stats.ks_2samp(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "goodness_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }
        if function_name == "zscore":
            result = stats.zscore(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "array",
                "result": {"values": _as_float_list(result)},
                "error": None,
            }
        if function_name == "sem":
            return _scalar_case(case_id, stats.sem(*args))
        if function_name == "iqr":
            return _scalar_case(case_id, stats.iqr(*args))
        if function_name == "moment":
            return _scalar_case(case_id, stats.moment(*args))
        if function_name == "variation":
            return _scalar_case(case_id, stats.variation(*args))
        if function_name == "entropy":
            return _scalar_case(case_id, stats.entropy(*args))
        if function_name == "shapiro":
            result = stats.shapiro(*args)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "goodness_result",
                "result": {
                    "statistic": _as_float(result.statistic),
                    "pvalue": _as_float(result.pvalue),
                },
                "error": None,
            }

        # --- Distribution method dispatcher (per frankenscipy-ygq3) ---
        # Routes every distribution pdf/cdf/ppf/sf/isf/logpdf/logcdf/mean/
        # var/entropy/fit/rvs query through scipy.stats.<dist>(...)(*args).
        # Fixture schema:
        #     function: "distribution_method"
        #     distribution: "norm" | "weibull_min" | "gamma" | ...
        #     method: "pdf" | "cdf" | "ppf" | "sf" | "mean" | "var" | "fit"
        #     params: {loc: 0.0, scale: 1.0}   # frozen distribution kwargs
        #     args: [...]                       # positional args to method
        if function_name == "distribution_method":
            dist_name = case["distribution"]
            method_name = case["method"]
            params = case.get("params", {}) or {}
            dist_cls = getattr(stats, dist_name, None)
            if dist_cls is None:
                return {
                    "case_id": case_id,
                    "status": "error",
                    "result_kind": "unsupported_distribution",
                    "result": {},
                    "error": f"scipy.stats has no distribution `{dist_name}`",
                }
            # fit is a class method on the unfrozen distribution; other
            # methods are resolved on the frozen instance.
            if method_name == "fit":
                result = dist_cls.fit(*args, **params)
            else:
                frozen = dist_cls(**params)
                method = getattr(frozen, method_name, None)
                if method is None:
                    return {
                        "case_id": case_id,
                        "status": "error",
                        "result_kind": "unsupported_method",
                        "result": {},
                        "error": f"distribution `{dist_name}` has no method `{method_name}`",
                    }
                result = method(*args)
            # Pack result based on method:
            # scalar methods -> scalar; array methods -> array; fit -> tuple of floats
            if method_name == "fit":
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "fit_params",
                    "result": {"params": [_as_float(v) for v in result]},
                    "error": None,
                }
            if method_name == "rvs":
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "array",
                    "result": {"values": _as_float_list(result)},
                    "error": None,
                }
            # pdf/cdf/ppf/sf/isf/logpdf/logcdf can return scalar OR array
            # depending on whether args[0] is a scalar or array
            if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "array",
                    "result": {"values": _as_float_list(result)},
                    "error": None,
                }
            return _scalar_case(case_id, result)

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function_name}",
        }
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy stats oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=False, default="",
                        help="(unused) legacy oracle root path — kept for CLI backwards compat")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np  # noqa: F401
        import scipy
        from scipy import stats
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": sys.modules["numpy"].__version__,
            "scipy_version": scipy.__version__,
        },
        "case_outputs": [_run_case(case, stats) for case in fixture.get("cases", [])],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
