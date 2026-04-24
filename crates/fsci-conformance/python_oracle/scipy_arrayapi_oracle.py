#!/usr/bin/env python3
"""NumPy-backed oracle capture for FrankenSciPy Array API fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def _err(case_id: str, error: str, result_kind: str = "exception") -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "error",
        "result_kind": result_kind,
        "result": {},
        "error": error,
    }


def _expected_error(case: Dict[str, Any]) -> Dict[str, Any] | None:
    expected = case.get("expected", {})
    if expected.get("kind") != "error_kind":
        return None
    return _ok(case["case_id"], "error_kind", {"error": expected.get("error", "Error")})


def _dtype(name: str, np: Any) -> Any:
    return {
        "bool": np.bool_,
        "int64": np.int64,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }[name]


def _dtype_name(dtype: Any, np: Any) -> str:
    normalized = np.dtype(dtype)
    if normalized == np.dtype("bool"):
        return "bool"
    if normalized == np.dtype("int64"):
        return "int64"
    if normalized == np.dtype("uint64"):
        return "uint64"
    if normalized == np.dtype("float32"):
        return "float32"
    if normalized == np.dtype("float64"):
        return "float64"
    if normalized == np.dtype("complex64"):
        return "complex64"
    if normalized == np.dtype("complex128"):
        return "complex128"
    if normalized.kind in {"i", "u"}:
        return "int64"
    if normalized.kind == "f":
        return "float64"
    if normalized.kind == "c":
        return "complex128"
    return str(normalized)


def _scalar_value(value: Dict[str, Any]) -> Any:
    kind = value["kind"]
    if kind == "bool":
        return bool(value["value"])
    if kind == "i64":
        return int(value["value"])
    if kind == "u64":
        return int(value["value"])
    if kind == "f64":
        return float(value["value"])
    if kind == "complex_f64":
        return complex(float(value["re"]), float(value["im"]))
    raise ValueError(f"unsupported scalar kind: {kind}")


def _scalar_payload(value: Any, dtype_name: str) -> Dict[str, Any]:
    if dtype_name == "bool":
        return {"kind": "bool", "value": bool(value)}
    if dtype_name == "int64":
        return {"kind": "i64", "value": int(value)}
    if dtype_name == "uint64":
        return {"kind": "u64", "value": int(value)}
    if dtype_name in {"complex64", "complex128"}:
        c = complex(value)
        return {"kind": "complex_f64", "re": float(c.real), "im": float(c.imag)}
    return {"kind": "f64", "value": float(value)}


def _array_payload(array: Any, np: Any, dtype_hint: str | None = None) -> Dict[str, Any]:
    dtype_name = dtype_hint or _dtype_name(array.dtype, np)
    flat = array.reshape(-1, order="C") if array.shape else [array.item()]
    return {
        "shape": list(array.shape),
        "dtype": dtype_name,
        "values": [_scalar_payload(value, dtype_name) for value in flat],
    }


def _array_from_fixture(
    values: List[Dict[str, Any]],
    shape: List[int],
    dtype_name: str,
    np: Any,
) -> Any:
    array = np.array([_scalar_value(value) for value in values], dtype=_dtype(dtype_name, np))
    return array.reshape(tuple(shape))


def _slice(spec: Dict[str, Any]) -> slice:
    return slice(spec.get("start"), spec.get("stop"), spec.get("step", 1))


def _getitem_index(index: Dict[str, Any], np: Any) -> Any:
    kind = index["kind"]
    if kind == "basic":
        return tuple(_slice(spec) for spec in index["slices"])
    if kind == "advanced":
        indices = index["indices"]
        if len(indices) == 1:
            return np.array(indices[0], dtype=np.int64)
        return tuple(np.array(axis, dtype=np.int64) for axis in indices)
    raise ValueError(f"unsupported index kind: {kind}")


def _run_case(case: Dict[str, Any], np: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    operation = case.get("operation", "<missing>")

    expected_error = _expected_error(case)
    if expected_error is not None:
        return expected_error

    try:
        if operation == "zeros":
            dtype_name = case["dtype"]
            array = np.zeros(tuple(case["shape"]), dtype=_dtype(dtype_name, np))
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "ones":
            dtype_name = case["dtype"]
            array = np.ones(tuple(case["shape"]), dtype=_dtype(dtype_name, np))
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "full":
            dtype_name = case["dtype"]
            array = np.full(
                tuple(case["shape"]),
                _scalar_value(case["fill_value"]),
                dtype=_dtype(dtype_name, np),
            )
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "arange":
            dtype_name = case.get("dtype")
            array = np.arange(
                _scalar_value(case["start"]),
                _scalar_value(case["stop"]),
                _scalar_value(case["step"]),
                dtype=_dtype(dtype_name, np) if dtype_name else None,
            )
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "linspace":
            dtype_name = case.get("dtype")
            array = np.linspace(
                _scalar_value(case["start"]),
                _scalar_value(case["stop"]),
                int(case["num"]),
                endpoint=bool(case["endpoint"]),
                dtype=_dtype(dtype_name, np) if dtype_name else None,
            )
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "broadcast_shapes":
            dims = np.broadcast_shapes(*(tuple(shape) for shape in case["shapes"]))
            return _ok(case_id, "shape", {"dims": list(dims)})

        if operation == "result_type":
            dtype = np.result_type(*(_dtype(name, np) for name in case["dtypes"]))
            return _ok(case_id, "dtype", {"dtype": _dtype_name(dtype, np)})

        if operation == "from_slice":
            dtype_name = case["dtype"]
            array = _array_from_fixture(case["values"], case["shape"], dtype_name, np)
            return _ok(case_id, "array", _array_payload(array, np, dtype_name))

        if operation == "getitem":
            dtype_name = case["source_dtype"]
            array = _array_from_fixture(
                case["source_values"],
                case["source_shape"],
                dtype_name,
                np,
            )
            selected = array[_getitem_index(case["index"], np)]
            return _ok(case_id, "array", _array_payload(np.asarray(selected), np, dtype_name))

        if operation == "reshape":
            dtype_name = case["source_dtype"]
            array = _array_from_fixture(
                case["source_values"],
                case["source_shape"],
                dtype_name,
                np,
            )
            reshaped = array.reshape(tuple(case["new_shape"]))
            return _ok(case_id, "array", _array_payload(reshaped, np, dtype_name))

        if operation == "transpose":
            dtype_name = case["source_dtype"]
            array = _array_from_fixture(
                case["source_values"],
                case["source_shape"],
                dtype_name,
                np,
            )
            return _ok(case_id, "array", _array_payload(np.transpose(array), np, dtype_name))

        if operation == "relation_broadcast_commutative":
            left = tuple(case["left_shape"])
            right = tuple(case["right_shape"])
            value = np.broadcast_shapes(left, right) == np.broadcast_shapes(right, left)
            return _ok(case_id, "bool", {"value": bool(value)})

        if operation == "relation_result_type_symmetry":
            left = _dtype(case["left_dtype"], np)
            right = _dtype(case["right_dtype"], np)
            value = np.result_type(left, right) == np.result_type(right, left)
            return _ok(case_id, "bool", {"value": bool(value)})

        if operation == "relation_index_roundtrip":
            dtype_name = case["dtype"]
            values = [_scalar_value(value) for value in case["values"]]
            array = np.array(values, dtype=_dtype(dtype_name, np))
            index = int(case["index"])
            selected = array[index]
            normalized = index if index >= 0 else len(values) + index
            expected = array[normalized]
            return _ok(case_id, "bool", {"value": bool(selected == expected)})

        return _err(case_id, f"unsupported operation: {operation}", "unsupported_function")

    except (
        ArithmeticError,
        OverflowError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
    ) as exc:
        return _err(case_id, f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture NumPy Array API oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        import scipy
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    case_outputs = [_run_case(case, np=np) for case in fixture.get("cases", [])]
    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": getattr(np, "__version__", "unknown"),
            "scipy_version": getattr(scipy, "__version__", "unknown"),
        },
        "case_outputs": case_outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
