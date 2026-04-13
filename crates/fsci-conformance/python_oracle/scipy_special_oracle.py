#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy special function packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_float(value: Any) -> float:
    return float(value)


def _run_case(case: Dict[str, Any], special: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    function_name = case.get("function")
    args = case.get("args", [])

    try:
        # Map function name to scipy.special function
        func_map = {
            "gamma": special.gamma,
            "gammaln": special.gammaln,
            "gammainc": special.gammainc,
            "gammaincc": special.gammaincc,
            "digamma": special.digamma,
            "polygamma": special.polygamma,
            "rgamma": special.rgamma,
            "loggamma": special.loggamma,
            "beta": special.beta,
            "betaln": special.betaln,
            "betainc": special.betainc,
            "betaincinv": special.betaincinv,
            "erf": special.erf,
            "erfc": special.erfc,
            "erfinv": special.erfinv,
            "erfcinv": special.erfcinv,
            "i0": special.i0,
            "i1": special.i1,
            "i0e": special.i0e,
            "i1e": special.i1e,
            "iv": special.iv,
            "ive": special.ive,
            "j0": special.j0,
            "j1": special.j1,
            "jv": special.jv,
            "jve": special.jve,
            "y0": special.y0,
            "y1": special.y1,
            "yv": special.yv,
            "yve": special.yve,
            "k0": special.k0,
            "k1": special.k1,
            "k0e": special.k0e,
            "k1e": special.k1e,
            "kv": special.kv,
            "kve": special.kve,
            "zeta": special.zeta,
            "zetac": special.zetac,
            "riemann_zeta": lambda x: special.zeta(x, 1),  # Riemann zeta
            "factorial": special.factorial,
            "factorial2": special.factorial2,
            "comb": special.comb,
            "perm": special.perm,
            "stirling2": lambda n, k: special.comb(n, k, exact=True) * special.factorial(k, exact=True) if k <= n else 0,
            "expit": special.expit,
            "logit": special.logit,
            "log_expit": special.log_expit,
            "log_ndtr": special.log_ndtr,
            "ndtr": special.ndtr,
            "ndtri": special.ndtri,
            "log1p": np.log1p,
            "expm1": np.expm1,
            "sinc": np.sinc,
            "xlogy": special.xlogy,
            "xlog1py": special.xlog1py,
            "entr": special.entr,
            "rel_entr": special.rel_entr,
            "kl_div": special.kl_div,
            "huber": special.huber,
            "pseudo_huber": special.pseudo_huber,
            "softmax": special.softmax,
            "log_softmax": special.log_softmax,
            "logsumexp": special.logsumexp,
            "exprel": special.exprel,
            "airy": special.airy,
            "airye": special.airye,
            "ai_zeros": special.ai_zeros,
            "bi_zeros": special.bi_zeros,
            "ellipk": special.ellipk,
            "ellipkm1": special.ellipkm1,
            "ellipe": special.ellipe,
            "ellipj": special.ellipj,
            "ellipkinc": special.ellipkinc,
            "ellipeinc": special.ellipeinc,
            "elliprc": special.elliprc,
            "elliprd": special.elliprd,
            "elliprf": special.elliprf,
            "elliprg": special.elliprg,
            "elliprj": special.elliprj,
            "lambertw": special.lambertw,
            "wrightomega": special.wrightomega,
        }

        if function_name not in func_map:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "unsupported_function",
                "result": {},
                "error": f"unsupported function: {function_name}",
            }

        func = func_map[function_name]
        result = func(*args)

        # Handle different return types
        if isinstance(result, tuple):
            # Multi-output functions like airy
            values = [_as_float(v) if np.isscalar(v) else _to_list(v) for v in result]
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "tuple",
                "result": {"values": values},
                "error": None,
            }
        elif np.isscalar(result) or (hasattr(result, 'ndim') and result.ndim == 0):
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "scalar",
                "result": {"value": _as_float(result)},
                "error": None,
            }
        else:
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(result)]},
                "error": None,
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
    parser = argparse.ArgumentParser(description="Capture SciPy special oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=True, help="Legacy oracle root path")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import special
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
        case_outputs.append(_run_case(case, special=special, np=np))

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
