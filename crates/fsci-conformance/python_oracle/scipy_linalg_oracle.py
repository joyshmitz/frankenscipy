#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy linalg packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _as_float(value: Any) -> float:
    return float(value)


def _coerce_maybe_nan_f64(value: Any) -> float:
    """Accept either a JSON number or a NaN/Inf string sentinel.

    Mirrors the Rust-side maybe_nan_f64 deserializer so fixtures can encode
    non-finite linalg inputs as "NaN" / "Infinity" / "-Infinity".
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "nan":
            return float("nan")
        if key in ("infinity", "inf", "+infinity", "+inf"):
            return float("inf")
        if key in ("-infinity", "-inf"):
            return float("-inf")
        return float(value)
    return float(value)


def _coerce_float_tree(value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_float_tree(v) for v in value]
    return _coerce_maybe_nan_f64(value)


def _float_array(value: Any, np: Any) -> Any:
    return np.asarray(_coerce_float_tree(value), dtype=np.float64)


def _optional_float(value: Any) -> Any:
    if value is None:
        return None
    return _coerce_maybe_nan_f64(value)


def _to_float_list(value: Any) -> List[float]:
    converted = _to_list(value)
    if converted is None:
        return []
    if isinstance(converted, (float, int)):
        return [float(converted)]
    return [float(v) for v in converted]


def _run_case(case: Dict[str, Any], linalg: Any, np: Any) -> Dict[str, Any]:
    operation = case["operation"]
    case_id = case["case_id"]

    try:
        if operation == "solve":
            a = _float_array(case["a"], np)
            b = _float_array(case["b"], np)
            assume_a = case.get("assume_a")
            transposed = bool(case.get("transposed", False))
            check_finite = bool(case.get("check_finite", True))
            if transposed:
                a = a.T

            if assume_a == "diagonal":
                diag = np.diag(a)
                if np.any(diag == 0.0):
                    raise linalg.LinAlgError("singular matrix")
                x = b / diag
            elif assume_a == "upper_triangular":
                x = linalg.solve_triangular(
                    a,
                    b,
                    lower=False,
                    trans=0,
                    unit_diagonal=False,
                    check_finite=check_finite,
                )
            elif assume_a == "lower_triangular":
                x = linalg.solve_triangular(
                    a,
                    b,
                    lower=True,
                    trans=0,
                    unit_diagonal=False,
                    check_finite=check_finite,
                )
            else:
                scipy_assume = {
                    None: "gen",
                    "general": "gen",
                    "symmetric": "sym",
                    "hermitian": "her",
                    "positive_definite": "pos",
                }.get(assume_a, "gen")
                x = linalg.solve(a, b, assume_a=scipy_assume, check_finite=check_finite)

            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "solve_triangular":
            a = _float_array(case["a"], np)
            b = _float_array(case["b"], np)
            trans_name = case.get("trans", "no_transpose")
            trans = {
                "no_transpose": 0,
                "transpose": 1,
                "conjugate_transpose": 2,
            }.get(trans_name, 0)
            x = linalg.solve_triangular(
                a,
                b,
                trans=trans,
                lower=bool(case.get("lower", False)),
                unit_diagonal=bool(case.get("unit_diagonal", False)),
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "solve_banded":
            l_and_u = case["l_and_u"]
            ab = _float_array(case["ab"], np)
            b = _float_array(case["b"], np)
            x = linalg.solve_banded(
                (int(l_and_u[0]), int(l_and_u[1])),
                ab,
                b,
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "inv":
            a = _float_array(case["a"], np)
            x = linalg.inv(a, check_finite=bool(case.get("check_finite", True)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix",
                "result": {"values": [[float(v) for v in row] for row in _to_list(x)]},
                "error": None,
            }

        if operation == "det":
            a = _float_array(case["a"], np)
            value = linalg.det(a, check_finite=bool(case.get("check_finite", True)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "scalar",
                "result": {"value": _as_float(value)},
                "error": None,
            }

        if operation == "lstsq":
            a = _float_array(case["a"], np)
            b = _float_array(case["b"], np)
            x, residuals, rank, singular_values = linalg.lstsq(
                a,
                b,
                cond=_optional_float(case.get("cond", None)),
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "lstsq",
                "result": {
                    "x": _to_float_list(x),
                    "residuals": _to_float_list(residuals),
                    "rank": int(rank),
                    "singular_values": _to_float_list(singular_values),
                },
                "error": None,
            }

        if operation == "pinv":
            a = _float_array(case["a"], np)
            x = linalg.pinv(
                a,
                atol=_optional_float(case.get("atol", 0.0)),
                rtol=_optional_float(case.get("rtol", None)),
                check_finite=bool(case.get("check_finite", True)),
            )
            singular_values = np.linalg.svd(a, compute_uv=False)
            max_s = float(np.max(singular_values)) if singular_values.size > 0 else 0.0
            atol = _coerce_maybe_nan_f64(case.get("atol", 0.0))
            if case.get("rtol", None) is None:
                rtol = float(max(a.shape) * np.finfo(np.float64).eps)
            else:
                rtol = _coerce_maybe_nan_f64(case["rtol"])
            threshold = atol + rtol * max_s
            rank = int(np.sum(singular_values > threshold))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "pinv",
                "result": {
                    "values": [[float(v) for v in row] for row in _to_list(x)],
                    "rank": rank,
                },
                "error": None,
            }

        # --- Decompositions (added per frankenscipy-5dr4) -----------------
        # The following dispatchers expand oracle coverage beyond the
        # solve/inv/det/lstsq/pinv set. Each uses np.asarray WITHOUT a
        # forced float64 dtype so complex fixture inputs survive.
        if operation == "lu":
            a = np.asarray(case["a"])
            permute_l = bool(case.get("permute_l", False))
            if permute_l:
                pl, u = linalg.lu(a, permute_l=True)
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "lu_permute_l",
                    "result": {
                        "pl": _to_list(pl),
                        "u": _to_list(u),
                    },
                    "error": None,
                }
            p, lower_factor, u = linalg.lu(a)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "lu",
                "result": {
                    "p": _to_list(p),
                    "l": _to_list(lower_factor),
                    "u": _to_list(u),
                },
                "error": None,
            }

        if operation == "qr":
            a = np.asarray(case["a"])
            runtime_or_qr_mode = case.get("mode", "full")
            mode = case.get("qr_mode")
            if mode is None:
                if str(runtime_or_qr_mode).lower() in {"strict", "hardened"}:
                    mode = "full"
                else:
                    mode = runtime_or_qr_mode
            if mode == "r":
                r, = linalg.qr(a, mode="r")
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "matrix",
                    "result": {"values": _to_list(r)},
                    "error": None,
                }
            q, r = linalg.qr(a, mode="full" if mode != "economic" else "economic")
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "qr",
                "result": {
                    "q": _to_list(q),
                    "r": _to_list(r),
                },
                "error": None,
            }

        if operation == "svd":
            a = np.asarray(case["a"])
            full_matrices = bool(case.get("full_matrices", True))
            compute_uv = bool(case.get("compute_uv", True))
            if not compute_uv:
                s = linalg.svd(a, compute_uv=False)
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "vector",
                    "result": {"values": [float(v) for v in _to_list(s)]},
                    "error": None,
                }
            u, s, vh = linalg.svd(a, full_matrices=full_matrices)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "svd",
                "result": {
                    "u": _to_list(u),
                    "s": [float(v) for v in _to_list(s)],
                    "vh": _to_list(vh),
                },
                "error": None,
            }

        if operation == "cholesky":
            a = np.asarray(case["a"])
            lower = bool(case.get("lower", False))
            result = linalg.cholesky(a, lower=lower)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix",
                "result": {"values": _to_list(result)},
                "error": None,
            }

        if operation == "eig":
            a = np.asarray(case["a"])
            w, v = linalg.eig(a)
            if bool(case.get("values_only", False)):
                real_values = [float(x.real) for x in _to_list(w) if abs(float(x.imag)) <= 1e-12]
                if len(real_values) != len(w):
                    raise ValueError("values_only eig fixture produced complex eigenvalues")
                real_values.sort()
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "vector",
                    "result": {"values": real_values},
                    "error": None,
                }
            # eigenvalues may be complex; encode re+im pairs
            w_list = [[float(x.real), float(x.imag)] for x in _to_list(w)]
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "eig",
                "result": {
                    "eigenvalues_reim": w_list,
                    "eigenvectors": _to_list(v),
                },
                "error": None,
            }

        if operation == "eigh":
            a = np.asarray(case["a"])
            lower = bool(case.get("lower", True))
            w, v = linalg.eigh(a, lower=lower)
            if bool(case.get("values_only", False)):
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "vector",
                    "result": {"values": [float(x) for x in _to_list(w)]},
                    "error": None,
                }
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "eigh",
                "result": {
                    "eigenvalues": [float(x) for x in _to_list(w)],
                    "eigenvectors": _to_list(v),
                },
                "error": None,
            }

        if operation == "expm":
            a = np.asarray(case["a"])
            result = linalg.expm(a)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "expm",
                "result": {"values": _to_list(result)},
                "error": None,
            }

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_operation",
            "result": {},
            "error": f"unsupported operation: {operation}",
        }

    # br-p3be: narrow catch. numpy.linalg.LinAlgError ⊆ ValueError, so
    # singular-matrix / non-convergence is covered. MemoryError / OSError
    # propagate.
    except (
        ArithmeticError,
        OverflowError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
    ) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy linalg oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=False, default="",
                        help="(unused) legacy oracle root path — kept for CLI backwards compat")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import linalg
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1
    case_outputs: List[Dict[str, Any]] = []

    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(case, linalg=linalg, np=np))

    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": getattr(np, "__version__", "unknown"),
            "scipy_version": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
        },
        "case_outputs": case_outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
