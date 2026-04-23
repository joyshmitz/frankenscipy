#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy integrate fixture.

Per frankenscipy-di9p: fsci-integrate previously had NO scipy oracle
script. This closes that gap for the FSCI-P2C-013 quadrature surface
(trapezoid / simpson / cumulative_* / romb / newton_cotes / fixed_quad /
gauss_legendre / cubature). IVP/solve_ivp/bvp are the subject of
frankenscipy-9cla (fixture extension) and remain un-oracled here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


def _translate(expr: str) -> str:
    """Translate fixture expression syntax to valid Python.

    `^` is the XOR operator in Python; fixtures use it for powers.
    """
    return expr.replace("^", "**")


def _build_callable(expr_or_name: str) -> Callable[..., float]:
    """Resolve a fixture `func` string to a callable.

    Named identifiers map to specific closures. Generic math expressions
    are compiled with a restricted `math` namespace.
    """
    named_1d = {
        "constant_1": (1, lambda x: 1.0),
        "powers_1d": (1, lambda x: x ** 3 + 2.0 * x ** 2 + 3.0 * x + 4.0),
    }
    named_nd = {
        # 2D pair-input: used by cubature zero-dim test cases.
        "zero_dim_pair": (2, lambda x, y: 0.0),
    }
    if expr_or_name in named_1d:
        arity, fn = named_1d[expr_or_name]
        return lambda x: fn(x)
    if expr_or_name in named_nd:
        arity, fn = named_nd[expr_or_name]
        return lambda *xs: fn(*xs[:arity])

    py = _translate(expr_or_name)
    # Discover variable names present in the expression.
    candidates = sorted(
        {tok for tok in ("x", "x0", "x1", "x2", "x3", "y") if tok in py}
    )
    if not candidates:
        # Pure constant expression (e.g. "1.0").
        def const_fn(*_args: float) -> float:
            return float(eval(py, {"__builtins__": {}}, vars(math)))  # noqa: S307
        return const_fn
    # Use a fresh namespace with math functions available.
    namespace = {"__builtins__": {}}
    for name in ("sin", "cos", "tan", "exp", "log", "log10", "sqrt", "pi", "e"):
        namespace[name] = getattr(math, name)

    def dispatched(*args: float) -> float:
        local = dict(namespace)
        for i, tok in enumerate(candidates):
            if i < len(args):
                local[tok] = args[i]
            else:
                local[tok] = 0.0
        return float(eval(py, {"__builtins__": {}}, local))  # noqa: S307

    return dispatched


def _build_bvp_problem(name: str):
    """Named BVP (f, bc) pair dispatcher for solve_bvp fixtures (br-9cla-4).

    Returns (f, bc) where:
      f(x, y)  — vectorized RHS: x is shape (n,), y is shape (k, n),
                 returns dy/dx shape (k, n).
      bc(ya, yb) — boundary-condition residuals: ya, yb are length-k,
                   returns length-k residual vector.
    """
    import numpy as np

    if name == "linear_y_double_prime_zero":
        # y'' = 0, y(0) = 0, y(1) = 1. Exact: y(x) = x.
        # System: y0' = y1, y1' = 0.
        def f(x, y):
            return np.vstack([y[1], np.zeros_like(x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0] - 1.0])

        return f, bc
    if name == "poisson_constant_source":
        # y'' = -1, y(0) = 0, y(1) = 0. Exact: y(x) = x(1-x)/2.
        def f(x, y):
            return np.vstack([y[1], -np.ones_like(x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        return f, bc
    raise ValueError(f"unknown bvp problem: {name!r}")


def _build_ivp_rhs(name: str) -> Callable[..., list]:
    """Named RHS dispatcher for solve_ivp fixtures (br-9cla-1).

    The Rust runner maintains an identically-keyed lookup table so both
    sides of the differential comparison integrate the *same* ODE. Do
    not add entries here without mirroring the Rust side in slice 9cla-2.
    """
    if name == "exponential_decay":
        # dy/dt = -y; y(0) = y0 → y(t) = y0 * exp(-t).
        return lambda t, y: [-y[0]]
    if name == "linear_growth":
        # dy/dt = 1; y(0) = y0 → y(t) = y0 + t.
        return lambda t, y: [1.0]
    if name == "harmonic_oscillator":
        # d²x/dt² + x = 0 → [dy0/dt, dy1/dt] = [y1, -y0]. Energy
        # E = 0.5*(y0² + y1²) is conserved.
        return lambda t, y: [y[1], -y[0]]
    raise ValueError(f"unknown ivp rhs: {name!r}")


def _run_case(case: Dict[str, Any], integrate: Any, np: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", {}) if isinstance(case.get("args"), dict) else {}

    try:
        if function == "trapezoid":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = float(integrate.trapezoid(y, x=x))
            return _ok(case_id, "scalar", {"value": result})

        if function == "simpson":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = float(integrate.simpson(y, x=x))
            return _ok(case_id, "scalar", {"value": result})

        if function == "cumulative_trapezoid":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = integrate.cumulative_trapezoid(y, x=x, initial=0.0)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "cumulative_simpson":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = integrate.cumulative_simpson(y, x=x, initial=0.0)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "romb":
            y = np.asarray(args["y"], dtype=float)
            dx = float(args.get("dx", 1.0))
            result = float(integrate.romb(y, dx=dx))
            return _ok(case_id, "scalar", {"value": result})

        if function == "newton_cotes":
            n = int(args["n"])
            weights, error = integrate.newton_cotes(n)
            return _ok(
                case_id,
                "newton_cotes",
                {
                    "weights": [float(w) for w in weights.tolist()],
                    "error": float(error),
                },
            )

        if function == "fixed_quad":
            fn = _build_callable(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            n = int(args.get("n", 5))
            # scipy.integrate.fixed_quad passes arrays; adapt.
            def vec_fn(xs: Any) -> Any:
                return np.asarray([fn(float(x)) for x in np.atleast_1d(xs)], dtype=float)
            value, _none = integrate.fixed_quad(vec_fn, a, b, n=n)
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "gauss_legendre":
            # scipy exposes gauss_legendre weights+nodes via roots_legendre.
            from scipy.special import roots_legendre

            n = int(args.get("n", 5))
            a = float(args["a"])
            b = float(args["b"])
            fn = _build_callable(args["func"])
            nodes, weights = roots_legendre(n)
            half = (b - a) / 2.0
            mid = (a + b) / 2.0
            total = 0.0
            for xi, wi in zip(nodes, weights):
                total += wi * fn(mid + half * xi)
            total *= half
            return _ok(case_id, "scalar", {"value": float(total)})

        if function == "solve_bvp":
            # br-9cla-4: collocation-based BVP. Named (f, bc) pair
            # plus a mesh `x` and an initial guess `y_init` of shape
            # (k, len(x)). Returns solution sampled at `x_eval` points
            # via scipy's interpolating `.sol(x_eval)` callable.
            problem = args["problem"]
            f, bc = _build_bvp_problem(problem)
            x = np.asarray(args["x"], dtype=float)
            y_init = np.asarray(args["y_init"], dtype=float)
            tol = float(args.get("tol", 1e-8))
            x_eval = np.asarray(
                args.get("x_eval", args["x"]), dtype=float
            )
            res = integrate.solve_bvp(f, bc, x, y_init, tol=tol)
            y_sampled = res.sol(x_eval)
            return _ok(
                case_id,
                "bvp_result",
                {
                    "x": [float(v) for v in x_eval.tolist()],
                    "y": [[float(v) for v in row] for row in y_sampled.tolist()],
                    "success": bool(res.success),
                    "niter": int(res.niter),
                    "rms_residuals": [
                        float(v) for v in np.atleast_1d(res.rms_residuals).tolist()
                    ],
                },
            )

        if function == "solve_ivp":
            # br-9cla-1: named-RHS dispatch. The Rust runner (9cla-2)
            # mirrors this name -> closure table so both sides agree on
            # the ODE being integrated.
            rhs_name = args["rhs"]
            t_span = tuple(float(v) for v in args["t_span"])
            y0 = [float(v) for v in args["y0"]]
            method = args.get("method", "RK45")
            rtol = float(args.get("rtol", 1e-3))
            atol = float(args.get("atol", 1e-6))
            t_eval_raw = args.get("t_eval")
            t_eval = None
            if t_eval_raw is not None:
                t_eval = [float(v) for v in t_eval_raw]
            rhs = _build_ivp_rhs(rhs_name)
            res = integrate.solve_ivp(
                rhs,
                t_span,
                y0,
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=t_eval,
            )
            return _ok(
                case_id,
                "ivp_result",
                {
                    "t": [float(v) for v in res.t.tolist()],
                    "y": [[float(v) for v in row] for row in res.y.tolist()],
                    "status": int(res.status),
                    "success": bool(res.success),
                    "nfev": int(res.nfev),
                },
            )

        if function == "quad":
            # br-9cla-5: scipy.integrate.quad adaptive scalar quadrature.
            # Named scalar integrand lookup; identical registry keyed
            # by `func` lives in the Rust runner's make_integrate_func.
            fn = _build_callable(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("atol", 1.49e-8))
            epsrel = float(args.get("rtol", 1.49e-8))
            limit = int(args.get("max_subdivisions", 50))
            value, _err = integrate.quad(
                fn, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "quad_vec":
            # br-9cla-5: scipy.integrate.quad_vec. Named vector integrand.
            # Only a handful of named vector integrands are supported;
            # each must match make_integrate_quad_vec_func on the Rust
            # side.
            name = args["func"]
            if name == "linear_square":
                vec_fn = lambda x: np.array([x, x * x], dtype=float)  # noqa: E731
            else:
                raise ValueError(f"quad_vec: unknown func: {name}")
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("atol", 1.49e-8))
            epsrel = float(args.get("rtol", 1.49e-8))
            limit = int(args.get("max_subdivisions", 50))
            result = integrate.quad_vec(
                vec_fn, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            integral = result[0]
            return _ok(
                case_id,
                "array",
                {"value": [float(v) for v in np.atleast_1d(integral).tolist()]},
            )

        if function == "dblquad":
            # br-9cla-5: scipy.integrate.dblquad. Named 2D integrand with
            # constant-in-x inner bounds baked into the registry so the
            # fixture schema stays flat.
            name = args["func"]
            if name == "xy_prod_unit_y":
                # scipy takes f(y, x); we keep the same convention.
                inner_fn = lambda y, x: x * y  # noqa: E731
                y_lo, y_hi = 0.0, 1.0
            else:
                raise ValueError(f"dblquad: unknown func: {name}")
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("atol", 1.49e-8))
            epsrel = float(args.get("rtol", 1.49e-8))
            value, _err = integrate.dblquad(
                inner_fn, a, b, y_lo, y_hi, epsabs=epsabs, epsrel=epsrel
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "tplquad":
            # br-9cla-5: scipy.integrate.tplquad. Named 3D integrand
            # with constant inner+middle bounds baked into the registry.
            name = args["func"]
            if name == "xyz_prod_unit_yz":
                # scipy takes f(z, y, x).
                inner_fn = lambda z, y, x: x * y * z  # noqa: E731
                y_lo, y_hi = 0.0, 1.0
                z_lo, z_hi = 0.0, 1.0
            else:
                raise ValueError(f"tplquad: unknown func: {name}")
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("atol", 1.49e-8))
            epsrel = float(args.get("rtol", 1.49e-8))
            value, _err = integrate.tplquad(
                inner_fn, a, b, y_lo, y_hi, z_lo, z_hi, epsabs=epsabs, epsrel=epsrel
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "cubature":
            lower = [float(v) for v in args["lower"]]
            upper = [float(v) for v in args["upper"]]
            atol = float(args.get("atol", 1e-10))
            rtol = float(args.get("rtol", 1e-10))
            fn = _build_callable(args["func"])
            # scipy.integrate.cubature was added in scipy 1.15; fall back
            # to nquad for older installs.
            if hasattr(integrate, "cubature"):
                def cube_fn(xs: Any) -> Any:
                    # scipy.integrate.cubature expects f(x) where x is
                    # (..., ndim). Flatten leading axes and apply.
                    x = np.atleast_2d(xs)
                    return np.asarray([fn(*row) for row in x], dtype=float)
                res = integrate.cubature(cube_fn, lower, upper, atol=atol, rtol=rtol)
                value = float(res.estimate) if hasattr(res, "estimate") else float(res[0])
                return _ok(case_id, "cubature_scalar", {
                    "value": value,
                    "status": "converged",
                })
            # Fallback via nquad for older scipy.
            def nquad_fn(*xs: float) -> float:
                return fn(*xs)
            ranges = list(zip(lower, upper))
            value, _err = integrate.nquad(nquad_fn, ranges, opts={"epsabs": atol, "epsrel": rtol})
            return _ok(case_id, "cubature_scalar", {
                "value": float(value),
                "status": "converged_via_nquad",
            })

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function}",
        }

    except (ArithmeticError, OverflowError, TypeError, ValueError, KeyError) as exc:
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
    parser = argparse.ArgumentParser(description="Capture SciPy integrate oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import integrate
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
        case_outputs.append(_run_case(case, integrate=integrate, np=np))

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
