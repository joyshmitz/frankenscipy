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


def _decode_arg(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"re", "im"}:
            return complex(float(value["re"]), float(value["im"]))
        return {key: _decode_arg(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_decode_arg(item) for item in value]
    return value


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _complex_payload(value: Any) -> Dict[str, float]:
    if hasattr(value, "item"):
        value = value.item()
    return {"re": float(value.real), "im": float(value.imag)}


def _normalize_value(value: Any, np: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_normalize_value(item, np) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item, np) for item in value]
    if np.iscomplexobj(value):
        return _complex_payload(value)
    return _as_float(value)


def _result_payload(case_id: str, result: Any, np: Any) -> Dict[str, Any]:
    if isinstance(result, tuple):
        values = [_normalize_value(value, np) for value in result]
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "tuple",
            "result": {"values": values},
            "error": None,
        }

    if np.isscalar(result) or (hasattr(result, "ndim") and result.ndim == 0):
        if np.iscomplexobj(result):
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "complex_scalar",
                "result": {"value": _complex_payload(result)},
                "error": None,
            }
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "scalar",
            "result": {"value": _as_float(result)},
            "error": None,
        }

    values = _normalize_value(result, np)
    if np.iscomplexobj(result):
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "complex_vector",
            "result": {"values": values},
            "error": None,
        }
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": "vector",
        "result": {"values": values},
        "error": None,
    }


def _run_case(case: Dict[str, Any], special: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    function_name = case.get("function")
    args = [_decode_arg(arg) for arg in case.get("args", [])]

    try:
        def rel_erf_erfc_identity(x: Any) -> Any:
            x_arr = np.asarray(x)
            return special.erf(x_arr) + special.erfc(x_arr)

        def rel_gamma_recurrence(x: Any) -> Any:
            x_arr = np.asarray(x)
            return special.gamma(x_arr + 1.0) - x_arr * special.gamma(x_arr)

        def rel_beta_symmetry(a: Any, b: Any) -> Any:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            return special.beta(a_arr, b_arr) - special.beta(b_arr, a_arr)

        def rel_gammainc_complement(a: Any, x: Any) -> Any:
            a_arr = np.asarray(a)
            x_arr = np.asarray(x)
            return special.gammainc(a_arr, x_arr) + special.gammaincc(a_arr, x_arr)

        def rel_jn_recurrence(n: Any, x: Any) -> Any:
            n_val = np.asarray(n, dtype=float)
            x_arr = np.asarray(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(x_arr == 0, np.nan, x_arr)
                return (
                    special.jv(n_val - 1.0, x_arr)
                    - (2.0 * n_val / denom) * special.jv(n_val, x_arr)
                    + special.jv(n_val + 1.0, x_arr)
                )

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
            "jvp": special.jvp,
            "jve": special.jve,
            "y0": special.y0,
            "y1": special.y1,
            "yv": special.yv,
            "yvp": special.yvp,
            "yve": special.yve,
            "k0": special.k0,
            "k1": special.k1,
            "k0e": special.k0e,
            "k1e": special.k1e,
            "kv": special.kv,
            "kvp": special.kvp,
            "kve": special.kve,
            "ivp": special.ivp,
            "zeta": special.zeta,
            "zetac": special.zetac,
            "riemann_zeta": lambda x: special.zeta(x, 1),  # Riemann zeta
            "factorial": special.factorial,
            "factorial2": special.factorial2,
            "comb": special.comb,
            "perm": special.perm,
            "multigammaln": special.multigammaln,
            "chdtr": special.chdtr,
            "chdtrc": special.chdtrc,
            "chdtri": special.chdtri,
            "chdtriv": special.chdtriv,
            "gdtr": special.gdtr,
            "gdtrc": special.gdtrc,
            "gdtrix": special.gdtrix,
            "gdtria": special.gdtria,
            "gdtrib": special.gdtrib,
            "pdtr": special.pdtr,
            "pdtrc": special.pdtrc,
            "pdtri": special.pdtri,
            "pdtrik": special.pdtrik,
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
            "hyp0f1": special.hyp0f1,
            "hyp1f1": special.hyp1f1,
            "hyp2f1": special.hyp2f1,
            "hankel1": special.hankel1,
            "hankel2": special.hankel2,
            "wright_bessel": special.wright_bessel,
            "spherical_jn": special.spherical_jn,
            "spherical_yn": special.spherical_yn,
            "spherical_in": special.spherical_in,
            "spherical_kn": special.spherical_kn,
            "eval_legendre": special.eval_legendre,
            "eval_chebyt": special.eval_chebyt,
            "eval_chebyu": special.eval_chebyu,
            "eval_hermite": special.eval_hermite,
            "eval_hermitenorm": special.eval_hermitenorm,
            "eval_laguerre": special.eval_laguerre,
            "eval_genlaguerre": special.eval_genlaguerre,
            "eval_jacobi": special.eval_jacobi,
            "eval_gegenbauer": special.eval_gegenbauer,
            "eval_sh_legendre": special.eval_sh_legendre,
            "eval_sh_chebyt": special.eval_sh_chebyt,
            "eval_sh_chebyu": special.eval_sh_chebyu,
            "roots_legendre": special.roots_legendre,
            "roots_chebyt": special.roots_chebyt,
            "roots_chebyu": special.roots_chebyu,
            "roots_hermite": special.roots_hermite,
            "roots_hermitenorm": special.roots_hermitenorm,
            "roots_laguerre": special.roots_laguerre,
            "roots_genlaguerre": special.roots_genlaguerre,
            "roots_jacobi": special.roots_jacobi,
            "roots_gegenbauer": special.roots_gegenbauer,
            "lpmv": special.lpmv,
            "sph_harm_y": special.sph_harm_y,
            "rel_erf_erfc_identity": rel_erf_erfc_identity,
            "rel_gamma_recurrence": rel_gamma_recurrence,
            "rel_beta_symmetry": rel_beta_symmetry,
            "rel_gammainc_complement": rel_gammainc_complement,
            "rel_jn_recurrence": rel_jn_recurrence,
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
        return _result_payload(case_id, result, np)

    except (ArithmeticError, FloatingPointError, OverflowError, TypeError, ValueError) as exc:
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
