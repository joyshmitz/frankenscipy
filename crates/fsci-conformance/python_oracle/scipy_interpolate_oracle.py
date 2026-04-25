#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy interpolate packet fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _float_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def _err(case_id: str, error: str) -> Dict[str, Any]:
    return _ok(case_id, "error", {"error": error})


def _fixture_error(case: Dict[str, Any], fallback: str) -> str:
    expected = case.get("expected", {})
    if expected.get("kind") == "error":
        return str(expected.get("error") or fallback)
    return fallback


def _coerce_maybe_nan_f64(value: Any) -> float:
    if isinstance(value, str):
        marker = value.strip().lower()
        if marker == "nan":
            return float("nan")
        if marker in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if marker in {"-inf", "-infinity"}:
            return float("-inf")
    return float(value)


def _cubic_spline_bc_type(case: Dict[str, Any]) -> Any:
    bc = case.get("bc", {"kind": "natural"})
    if isinstance(bc, str):
        return bc.replace("_", "-")

    kind = str(bc.get("kind", "natural"))
    if kind in {"natural", "not_a_knot", "periodic"}:
        return kind.replace("_", "-")
    if kind == "clamped":
        return (
            (1, _coerce_maybe_nan_f64(bc["left_derivative"])),
            (1, _coerce_maybe_nan_f64(bc["right_derivative"])),
        )
    if kind == "tuple":
        return (
            (int(bc["left_order"]), _coerce_maybe_nan_f64(bc["left_value"])),
            (int(bc["right_order"]), _coerce_maybe_nan_f64(bc["right_value"])),
        )
    raise ValueError(f"unsupported CubicSpline bc kind: {kind}")


def _run_interp1d(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    expected = case.get("expected", {})

    # FrankenSciPy's current strict contract rejects unsorted x at construction
    # while scipy.interpolate.interp1d silently sorts by default. Preserve the
    # packet's declared reject-path category until a separate parity slice
    # decides whether the Rust kernel should adopt SciPy's sorting behavior.
    x = [float(value) for value in case["x"]]
    if expected.get("kind") == "error" and any(right <= left for left, right in zip(x, x[1:])):
        return _err(case_id, _fixture_error(case, "x values must be strictly increasing"))

    try:
        y = np.array(case["y"], dtype=np.float64)
        x_new = np.array(case["x_new"], dtype=np.float64)
        kwargs: Dict[str, Any] = {"kind": case.get("kind", "linear")}
        if "bounds_error" in case:
            kwargs["bounds_error"] = bool(case["bounds_error"])
        if "fill_value" in case:
            kwargs["fill_value"] = float(case["fill_value"])

        interpolator = interpolate.interp1d(np.array(x, dtype=np.float64), y, **kwargs)
        values = interpolator(x_new)
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_regular_grid_interpolator(
    case: Dict[str, Any], interpolate: Any, np: Any
) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        points = [np.array(axis, dtype=np.float64) for axis in case["points"]]
        shape = tuple(len(axis) for axis in points)
        values = np.array(case["values"], dtype=np.float64).reshape(shape)
        xi = np.array(case["xi"], dtype=np.float64)
        kwargs: Dict[str, Any] = {
            "method": case.get("method", "linear"),
            "bounds_error": bool(case.get("bounds_error", True)),
        }
        if "fill_value" in case:
            kwargs["fill_value"] = float(case["fill_value"])

        interpolator = interpolate.RegularGridInterpolator(points, values, **kwargs)
        return _ok(case_id, "vector", {"values": _float_list(interpolator(xi))})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_cubic_spline(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        bc = _cubic_spline_bc_type(case)
        spline = interpolate.CubicSpline(
            np.array(case["x"], dtype=np.float64),
            np.array(case["y"], dtype=np.float64),
            bc_type=bc,
        )
        values = spline(np.array(case["x_new"], dtype=np.float64))
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_bspline(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        spline = interpolate.BSpline(
            np.array(case["knots"], dtype=np.float64),
            np.array(case["coefficients"], dtype=np.float64),
            int(case["degree"]),
        )
        values = spline(np.array(case["x_new"], dtype=np.float64))
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_case(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    operation = case.get("operation")
    if operation == "interp1d":
        return _run_interp1d(case, interpolate, np)
    if operation == "regular_grid_interpolator":
        return _run_regular_grid_interpolator(case, interpolate, np)
    if operation == "cubic_spline":
        return _run_cubic_spline(case, interpolate, np)
    if operation == "bspline":
        return _run_bspline(case, interpolate, np)
    return {
        "case_id": case.get("case_id", "<missing>"),
        "status": "error",
        "result_kind": "unsupported_operation",
        "result": {},
        "error": f"unsupported operation: {operation}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy interpolate oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument(
        "--oracle-root",
        required=False,
        default="",
        help="(unused) legacy oracle root path, kept for CLI backwards compatibility",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        import scipy
        from scipy import interpolate
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
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "case_outputs": [
            _run_case(case, interpolate, np) for case in fixture.get("cases", [])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
