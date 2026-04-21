#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy optimize packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


def _rosenbrock(x: Any) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def _rosenbrock_grad(x: Any) -> Any:
    """Gradient of Rosenbrock function."""
    import numpy as np
    n = len(x)
    g = np.zeros(n)
    g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    for i in range(1, n - 1):
        g[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    g[n-1] = 200 * (x[n-1] - x[n-2]**2)
    return g


def _sphere(x: Any) -> float:
    """Sphere function: f(x) = sum(x^2)"""
    return sum(xi**2 for xi in x)


def _beale(x: Any) -> float:
    """Beale function."""
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


def _booth(x: Any) -> float:
    """Booth function."""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def _himmelblau(x: Any) -> float:
    """Himmelblau function."""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def _rastrigin(x: Any) -> float:
    """Rastrigin function."""
    import numpy as np
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)


def _ackley(x: Any) -> float:
    """Ackley function."""
    import numpy as np
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / n)) - np.exp(sum(np.cos(2 * np.pi * xi) for xi in x) / n) + 20 + np.e


def _scaled_quadratic(x: Any) -> float:
    """Scaled quadratic with minimum at (1, -2)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return (x[0] - 1.0) ** 2 + 4.0 * (x[1] + 2.0) ** 2


def _translated_quadratic(x: Any) -> float:
    """Translated quadratic with minimum at (4, -3)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return (x[0] - 4.0) ** 2 + (x[1] + 3.0) ** 2


def _shifted_quadratic_grad(x: Any) -> Any:
    """Gradient of shifted_quadratic."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)], dtype=float)


def _shifted_quadratic_hess(_: Any) -> Any:
    """Hessian of shifted_quadratic."""
    import numpy as np
    return np.array([[2.0, 0.0], [0.0, 8.0]], dtype=float)


def _translated_quadratic_grad(x: Any) -> Any:
    """Gradient of translated_quadratic."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return np.array([2.0 * (x[0] - 4.0), 4.0 * (x[1] + 3.0)], dtype=float)


def _translated_quadratic_hess(_: Any) -> Any:
    """Hessian of translated_quadratic."""
    import numpy as np
    return np.array([[2.0, 0.0], [0.0, 4.0]], dtype=float)


def _scaled_quadratic_grad(x: Any) -> Any:
    """Gradient of scaled_quadratic."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return np.array([20.0 * (x[0] - 1.0), 80.0 * (x[1] + 2.0)], dtype=float)


def _scaled_quadratic_hess(_: Any) -> Any:
    """Hessian of scaled_quadratic."""
    import numpy as np
    return np.array([[20.0, 0.0], [0.0, 80.0]], dtype=float)


def _rotated_quadratic(x: Any) -> float:
    """Rotated quadratic with minimum at (1, -2)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    dx0 = x[0] - 1.0
    dx1 = x[1] + 2.0
    # 45-degree rotation
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    u = (dx0 + dx1) * inv_sqrt2
    v = (-dx0 + dx1) * inv_sqrt2
    return u ** 2 + 2.0 * v ** 2


def _nan_branch(x: Any) -> float:
    """Return NaN for near-origin inputs to trigger adversarial handling."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.all(np.abs(x) <= 0.5):
        return float('nan')
    return float(np.sum(x ** 2))


def _flat_quartic(x: Any) -> float:
    """Flat quartic landscape to stress maxfev handling."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 4))


def _get_objective(name: str) -> Callable:
    """Get objective function by name."""
    objectives = {
        "rosenbrock2": _rosenbrock,
        "rosenbrock": _rosenbrock,
        "rosenbrock4": _rosenbrock,
        "rosenbrock_4d": _rosenbrock,
        "sphere": _sphere,
        "sphere2": _sphere,
        "sphere_2d": _sphere,
        "beale": _beale,
        "booth": _booth,
        "himmelblau": _himmelblau,
        "rastrigin": _rastrigin,
        "rastrigin2": _rastrigin,
        "ackley": _ackley,
        "ackley2": _ackley,
        "scaled_quadratic": _scaled_quadratic,
        "translated_quadratic": _translated_quadratic,
        "rotated_quadratic": _rotated_quadratic,
        "nan_branch": _nan_branch,
        "flat_quartic": _flat_quartic,
    }
    return objectives.get(name, lambda x: float('nan'))


def _get_jacobian_and_hessian(name: str) -> tuple[Callable | None, Callable | None]:
    """Get exact derivatives for trust-region methods that require them."""
    derivatives = {
        "shifted_quadratic": (_shifted_quadratic_grad, _shifted_quadratic_hess),
        "translated_quadratic": (_translated_quadratic_grad, _translated_quadratic_hess),
        "scaled_quadratic": (_scaled_quadratic_grad, _scaled_quadratic_hess),
    }
    return derivatives.get(name, (None, None))


def _get_root_function(name: str, np: Any) -> Callable:
    """Get root-finding function by name."""
    funcs = {
        "linear_shift_03": lambda x: x - 0.3,
        "linear_shift": lambda x: x - 0.3,
        "linear_shift03": lambda x: x - 0.3,
        "cubic_root": lambda x: x**3 - 1,
        "cubic_minus_two": lambda x: x**3 - 2,
        "sin_root": lambda x: np.sin(x),
        "sin_minus_half": lambda x: np.sin(x) - 0.5,
        "cos_root": lambda x: np.cos(x),
        "cos_minus_x": lambda x: np.cos(x) - x,
        "poly_root": lambda x: x**2 - 2,
        "exp_root": lambda x: np.exp(x) - 2,
        "nan_branch": lambda x: np.nan,
    }
    return funcs.get(name, lambda x: float('nan'))


def _run_case(case: Dict[str, Any], optimize: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    operation = case.get("operation")

    try:
        if operation == "minimize":
            objective_name = case.get("objective")
            method_map = {
                "Bfgs": "BFGS",
                "Lbfgsb": "L-BFGS-B",
                "ConjugateGradient": "CG",
                "Powell": "Powell",
                "NelderMead": "Nelder-Mead",
                "Tnc": "TNC",
                "Cobyla": "COBYLA",
                "Slsqp": "SLSQP",
                "TrustNcg": "trust-ncg",
                "TrustKrylov": "trust-krylov",
                "TrustConstr": "trust-constr",
                "TrustExact": "trust-exact",
                "DogLeg": "dogleg",
            }
            method = method_map.get(case.get("method"), case.get("method"))

            x0 = np.array(case.get("x0", [0.0]))
            tol = case.get("tol")
            maxiter = case.get("maxiter")

            options = {}
            if maxiter is not None:
                options["maxiter"] = maxiter
            if case.get("maxfev"):
                options["maxfev"] = case["maxfev"]

            obj_func = _get_objective(objective_name)
            minimize_kwargs: Dict[str, Any] = {
                "method": method,
                "tol": tol,
                "options": options if options else None,
            }
            if method == "trust-exact":
                jac, hess = _get_jacobian_and_hessian(objective_name)
                if jac is None or hess is None:
                    raise ValueError(
                        f"objective {objective_name!r} lacks jac/hess support for trust-exact"
                    )
                minimize_kwargs["jac"] = jac
                minimize_kwargs["hess"] = hess

            result = optimize.minimize(
                obj_func,
                x0,
                **minimize_kwargs,
            )

            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "minimize",
                "result": {
                    "x": [float(v) for v in result.x],
                    "fun": float(result.fun),
                    "success": bool(result.success),
                    "nit": int(result.nit) if hasattr(result, 'nit') else 0,
                    "nfev": int(result.nfev) if hasattr(result, 'nfev') else 0,
                },
                "error": None,
            }

        elif operation == "root":
            func_name = case.get("objective")
            method_map = {
                "Brentq": "brentq",
                "Bisect": "bisect",
                "Newton": "newton",
                "Secant": "secant",
                "Ridder": "ridder",
                "Toms748": "toms748",
            }
            method = method_map.get(case.get("method"), case.get("method", "brentq")).lower()

            bracket = case.get("bracket", [0.0, 1.0])
            xtol = case.get("xtol", 1e-12)
            rtol = case.get("rtol", 1e-12)
            maxiter = case.get("maxiter", 100)

            root_func = _get_root_function(func_name, np)

            if method in ("brentq", "bisect", "ridder", "toms748"):
                result = optimize.root_scalar(
                    root_func,
                    bracket=bracket,
                    method=method,
                    xtol=xtol,
                    rtol=rtol,
                    maxiter=maxiter,
                )
            else:
                x0 = bracket[0] if bracket else 0.5
                result = optimize.root_scalar(
                    root_func,
                    x0=x0,
                    method=method,
                    xtol=xtol,
                    maxiter=maxiter,
                )

            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "root",
                "result": {
                    "root": float(result.root),
                    "converged": bool(result.converged),
                    "iterations": int(result.iterations) if hasattr(result, 'iterations') else 0,
                    "function_calls": int(result.function_calls) if hasattr(result, 'function_calls') else 0,
                },
                "error": None,
            }

        else:
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
    parser = argparse.ArgumentParser(description="Capture SciPy optimize oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=True, help="Legacy oracle root path")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import optimize
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
        case_outputs.append(_run_case(case, optimize=optimize, np=np))

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
