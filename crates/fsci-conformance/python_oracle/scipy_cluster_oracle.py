#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy cluster fixture.

Closes the cluster slice of frankenscipy-di9p. Covers the 12 functions
used by FSCI-P2C-009: linkage / fcluster / vq / whiten / cophenet /
inconsistent / kmeans / silhouette_score / adjusted_rand_score /
is_valid_linkage / is_monotonic / leaves_list / num_obs_linkage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _arr(v: Any) -> Any:
    return v


def _run_case(case: Dict[str, Any], np: Any, hierarchy: Any, vq_mod: Any, metrics: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", [])

    try:
        if function == "linkage":
            data = np.asarray(args[0], dtype=float)
            method = args[1] if len(args) > 1 else "single"
            metric = args[2] if len(args) > 2 else "euclidean"
            z = hierarchy.linkage(data, method=method, metric=metric)
            return _ok(case_id, "linkage_matrix", {
                "rows": [[float(v) for v in row] for row in z.tolist()],
            })

        if function == "fcluster":
            z = np.asarray(args[0], dtype=float)
            t = args[1]
            criterion = args[2] if len(args) > 2 else "maxclust"
            labels = hierarchy.fcluster(z, t=t, criterion=criterion)
            return _ok(case_id, "array", {
                "values": [int(v) for v in labels.tolist()],
            })

        if function == "kmeans":
            data = np.asarray(args[0], dtype=float)
            k = int(args[1])
            max_iter = int(args[2]) if len(args) > 2 else 100
            centroids, labels = vq_mod.kmeans2(
                data,
                k,
                iter=max_iter,
                minit="++",
                seed=case.get("seed"),
            )
            return _ok(case_id, "kmeans_result", {
                "centroids": [[float(v) for v in row] for row in centroids.tolist()],
                "labels": [int(v) for v in labels.tolist()],
            })

        if function == "vq":
            obs = np.asarray(args[0], dtype=float)
            code_book = np.asarray(args[1], dtype=float)
            codes, dists = vq_mod.vq(obs, code_book)
            return _ok(case_id, "vq_result", {
                "codes": [int(v) for v in codes.tolist()],
                "distances": [float(v) for v in dists.tolist()],
            })

        if function == "whiten":
            obs = np.asarray(args[0], dtype=float)
            result = vq_mod.whiten(obs)
            return _ok(case_id, "matrix", {
                "values": [[float(v) for v in row] for row in result.tolist()],
            })

        if function == "cophenet":
            z = np.asarray(args[0], dtype=float)
            condensed = hierarchy.cophenet(z)
            return _ok(case_id, "array", {
                "values": [float(v) for v in condensed.tolist()],
            })

        if function == "inconsistent":
            z = np.asarray(args[0], dtype=float)
            depth = int(args[1]) if len(args) > 1 else 2
            result = hierarchy.inconsistent(z, d=depth)
            return _ok(case_id, "matrix", {
                "values": [[float(v) for v in row] for row in result.tolist()],
            })

        if function == "silhouette_score":
            from sklearn.metrics import silhouette_score as sk_silhouette

            data = np.asarray(args[0], dtype=float)
            labels = np.asarray(args[1], dtype=int)
            score = float(sk_silhouette(data, labels))
            return _ok(case_id, "scalar", {"value": score})

        if function == "adjusted_rand_score":
            from sklearn.metrics import adjusted_rand_score as sk_ari

            a = np.asarray(args[0], dtype=int)
            b = np.asarray(args[1], dtype=int)
            score = float(sk_ari(a, b))
            return _ok(case_id, "scalar", {"value": score})

        if function == "is_valid_linkage":
            z = np.asarray(args[0], dtype=float)
            return _ok(case_id, "bool", {"value": bool(hierarchy.is_valid_linkage(z))})

        if function == "is_monotonic":
            z = np.asarray(args[0], dtype=float)
            return _ok(case_id, "bool", {"value": bool(hierarchy.is_monotonic(z))})

        if function == "leaves_list":
            z = np.asarray(args[0], dtype=float)
            result = hierarchy.leaves_list(z)
            return _ok(case_id, "array", {
                "values": [int(v) for v in result.tolist()],
            })

        if function == "num_obs_linkage":
            z = np.asarray(args[0], dtype=float)
            return _ok(case_id, "scalar_int", {
                "value": int(hierarchy.num_obs_linkage(z)),
            })

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function}",
        }

    except (ArithmeticError, OverflowError, TypeError, ValueError, KeyError, ImportError) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy cluster oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy.cluster import hierarchy, vq as vq_mod
        # sklearn metrics optional; silhouette + ARI cases will error out
        # if unavailable.
        metrics = None
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

    case_outputs: List[Dict[str, Any]] = []
    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(
            case, np=np, hierarchy=hierarchy, vq_mod=vq_mod, metrics=metrics,
        ))

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
