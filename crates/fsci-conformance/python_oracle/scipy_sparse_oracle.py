#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy sparse helper fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _matrix_from_spec(spec: Dict[str, Any], sparse: Any, np: Any) -> Any:
    shape = tuple(spec["shape"])
    row = np.asarray(spec["row"], dtype=np.int64)
    col = np.asarray(spec["col"], dtype=np.int64)
    data = np.asarray(spec["data"], dtype=np.float64)
    coo = sparse.coo_array((data, (row, col)), shape=shape)
    fmt = spec.get("format", "coo")
    if fmt == "coo":
        return coo
    if fmt == "csr":
        return coo.tocsr()
    if fmt == "csc":
        return coo.tocsc()
    raise ValueError(f"unsupported sparse fixture format: {fmt}")


def _matrix_result(matrix: Any) -> Dict[str, Any]:
    coo = matrix.tocoo()
    return {
        "shape": [int(coo.shape[0]), int(coo.shape[1])],
        "row": [int(v) for v in coo.row.tolist()],
        "col": [int(v) for v in coo.col.tolist()],
        "data": [float(v) for v in coo.data.tolist()],
    }


def _find_result(matrix: Any, sparse: Any) -> Dict[str, Any]:
    row, col, data = sparse.find(matrix)
    return {
        "row": [int(v) for v in row.tolist()],
        "col": [int(v) for v in col.tolist()],
        "data": [float(v) for v in data.tolist()],
    }


def _run_case(case: Dict[str, Any], sparse: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    operation = case["operation"]

    try:
        if operation == "find":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "find_triplets",
                "result": _find_result(matrix, sparse),
                "error": None,
            }

        if operation == "tril":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            result = sparse.tril(matrix, k=int(case.get("k", 0)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "triu":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            result = sparse.triu(matrix, k=int(case.get("k", 0)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "vstack":
            blocks = [_matrix_from_spec(block, sparse, np) for block in case["blocks"]]
            result = sparse.vstack(blocks)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "hstack":
            blocks = [_matrix_from_spec(block, sparse, np) for block in case["blocks"]]
            result = sparse.hstack(blocks)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_operation",
            "result": {},
            "error": f"unsupported operation: {operation}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy sparse helper oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import sparse
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
        case_outputs.append(_run_case(case, sparse=sparse, np=np))

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
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
