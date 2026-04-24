#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy ndimage packet fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _float_list(values: Any) -> List[float]:
    if hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    elif hasattr(values, "tolist"):
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


def _err(case_id: str, error: str, result_kind: str = "exception") -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "error",
        "result_kind": result_kind,
        "result": {},
        "error": error,
    }


def _ok_error(case_id: str, error: str) -> Dict[str, Any]:
    return _ok(case_id, "error", {"error": error})


def _expected_error(case: Dict[str, Any]) -> Dict[str, Any] | None:
    expected = case.get("expected", {})
    if expected.get("kind") == "error":
        return _ok_error(case["case_id"], str(expected.get("error", "error")))
    return None


def _array(case: Dict[str, Any], np: Any) -> Any:
    return np.asarray(case["input"], dtype=np.float64).reshape(tuple(case["shape"]))


def _array_payload(array: Any) -> Dict[str, Any]:
    return {"values": _float_list(array), "shape": [int(v) for v in array.shape]}


def _run_case(case: Dict[str, Any], np: Any, ndimage: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    operation = case.get("operation", "<missing>")

    expected_error = _expected_error(case)
    if expected_error is not None:
        return expected_error

    try:
        if operation == "gaussian_filter":
            result = ndimage.gaussian_filter(
                _array(case, np),
                sigma=float(case["sigma"]),
                mode=str(case.get("boundary", "reflect")),
                cval=float(case.get("cval", 0.0)),
            )
            return _ok(case_id, "array", _array_payload(result))

        if operation == "label":
            labels, num_features = ndimage.label(_array(case, np) != 0.0)
            return _ok(
                case_id,
                "label",
                {
                    "labels": _float_list(labels.astype(np.float64)),
                    "shape": [int(v) for v in labels.shape],
                    "num_features": int(num_features),
                },
            )

        if operation == "binary_erosion":
            structure_size = int(case["structure_size"])
            structure = np.ones((structure_size,) * len(case["shape"]), dtype=bool)
            result = ndimage.binary_erosion(
                _array(case, np) != 0.0,
                structure=structure,
                iterations=int(case["iterations"]),
                border_value=0,
            )
            return _ok(case_id, "array", _array_payload(result.astype(np.float64)))

        if operation == "binary_dilation":
            structure_size = int(case["structure_size"])
            structure = np.ones((structure_size,) * len(case["shape"]), dtype=bool)
            result = ndimage.binary_dilation(
                _array(case, np) != 0.0,
                structure=structure,
                iterations=int(case["iterations"]),
                border_value=0,
            )
            return _ok(case_id, "array", _array_payload(result.astype(np.float64)))

        if operation == "distance_transform_edt":
            kwargs: Dict[str, Any] = {}
            if case.get("sampling") is not None:
                kwargs["sampling"] = [float(value) for value in case["sampling"]]
            result = ndimage.distance_transform_edt(_array(case, np), **kwargs)
            return _ok(case_id, "array", _array_payload(result))

        return _err(
            case_id,
            f"unsupported operation: {operation}",
            result_kind="unsupported_operation",
        )

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
    parser = argparse.ArgumentParser(description="Capture SciPy ndimage oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        import scipy
        from scipy import ndimage
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
            _run_case(case, np=np, ndimage=ndimage) for case in fixture.get("cases", [])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
