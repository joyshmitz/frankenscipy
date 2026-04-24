#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy constants fixture.

Closes the constants slice of frankenscipy-di9p / frankenscipy-utus.
Compares fsci-constants named constants + helper conversions against
`scipy.constants` (CODATA 2018 + SI-exact inputs).

The Rust side dispatches `family = constants_core` to
`run_differential_constants`; this oracle mirrors every function in
that dispatch so parity drift between CODATA revisions (or between
fsci-constants refactors and scipy upstream) is caught by the fixture
instead of by hand.
"""

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


def _ok_error(case_id: str, error: str) -> Dict[str, Any]:
    return _ok(case_id, "error", {"error": error})


# Mapping from Rust identifier (case-insensitive) → scipy.constants attribute.
# Keep this tight: the fixture names double as the contract for which
# constants parity is verified. scipy.constants exposes some of these as
# SI-exact (codata-independent) while others (hbar, R) are derived.
_CONSTANT_MAP = {
    "PI": ("pi", None),                       # math.pi
    "TAU": ("tau", None),                     # 2*pi
    "E": ("e", None),                         # elementary charge (scipy.constants.e)
    "GOLDEN_RATIO": (None, 1.618_033_988_749_895),  # not in scipy.constants
    "SPEED_OF_LIGHT": ("c", None),
    "C": ("c", None),
    "PLANCK": ("h", None),
    "H": ("h", None),
    "HBAR": ("hbar", None),
    "GRAVITATIONAL_CONSTANT": ("G", None),
    "G": ("G", None),
    "G_N": ("g", None),
    "ELEMENTARY_CHARGE": ("e", None),
    "E_CHARGE": ("e", None),
    "GAS_CONSTANT": ("R", None),
    "R": ("R", None),
    "AVOGADRO": ("N_A", None),
    "N_A": ("N_A", None),
    "BOLTZMANN": ("k", None),
    "K_B": ("k", None),
    "STEFAN_BOLTZMANN": ("sigma", None),
    "SIGMA": ("sigma", None),
    "WIEN": ("Wien", None),
    "RYDBERG": ("Rydberg", None),
    "ELECTRON_MASS": ("m_e", None),
    "M_E": ("m_e", None),
    "PROTON_MASS": ("m_p", None),
    "M_P": ("m_p", None),
    "NEUTRON_MASS": ("m_n", None),
    "M_N": ("m_n", None),
    "ATOMIC_MASS": ("m_u", None),
    "U": ("m_u", None),
    "FINE_STRUCTURE": ("alpha", None),
    "ALPHA": ("alpha", None),
    "BOHR_RADIUS": (None, 5.291_772_109_03e-11),  # expose via .value("Bohr radius")
    "ELECTRON_VOLT": ("electron_volt", None),
    "EV": ("electron_volt", None),
    "CALORIE": ("calorie", None),
    "ATMOSPHERE": ("atm", None),
    "ATM": ("atm", None),
    "BAR": ("bar", None),
    "POUND": ("pound", None),
    "INCH": ("inch", None),
    "FOOT": ("foot", None),
    "DEGREE": ("degree", None),
}


def _run_case(case: Dict[str, Any], np: Any, sc: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", [])

    try:
        if function == "constant_value":
            if not args or not isinstance(args[0], str):
                return _err(case_id, "constant_value requires string arg")
            name = args[0].upper()
            if name not in _CONSTANT_MAP:
                return _ok_error(case_id, "unknown constant")
            attr, literal = _CONSTANT_MAP[name]
            if attr is not None:
                value = float(getattr(sc, attr))
            else:
                value = float(literal)
            return _ok(case_id, "scalar", {"value": value})

        if function == "convert_temperature":
            if len(args) < 3:
                return _err(case_id, "convert_temperature requires (val, from, to)")
            val = float(args[0])
            frm = str(args[1])
            to = str(args[2])
            # scipy accepts {'Celsius', 'Kelvin', 'Fahrenheit', 'Rankine'}
            # with arbitrary case; single-letter codes must be expanded.
            _LONG = {
                "C": "Celsius",
                "K": "Kelvin",
                "F": "Fahrenheit",
                "R": "Rankine",
            }
            frm_l = _LONG.get(frm.upper(), frm)
            to_l = _LONG.get(to.upper(), to)
            try:
                result = float(sc.convert_temperature(val, frm_l, to_l))
            except NotImplementedError:
                return _ok_error(case_id, "unsupported temperature scale")
            return _ok(case_id, "scalar", {"value": result})

        if function == "ev_to_joules":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": val * float(sc.electron_volt)})

        if function == "joules_to_ev":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": val / float(sc.electron_volt)})

        if function == "wavelength_to_freq":
            val = float(args[0])
            if val == 0.0:
                return _err(case_id, "division by zero")
            return _ok(case_id, "scalar", {"value": float(sc.c) / val})

        if function == "freq_to_wavelength":
            val = float(args[0])
            if val == 0.0:
                return _err(case_id, "division by zero")
            return _ok(case_id, "scalar", {"value": float(sc.c) / val})

        if function == "deg2rad":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": float(np.radians(val))})

        if function == "rad2deg":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": float(np.degrees(val))})

        if function == "lb_to_kg":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": val * float(sc.pound)})

        if function == "kg_to_lb":
            val = float(args[0])
            return _ok(case_id, "scalar", {"value": val / float(sc.pound)})

        return _err(
            case_id,
            f"unsupported function: {function}",
            result_kind="unsupported_function",
        )

    # Narrow catch per frankenscipy-p3be: MemoryError, RecursionError,
    # OSError, ImportError must propagate so CI fails visibly rather than
    # being silently reclassified as a scipy-side domain error.
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
    parser = argparse.ArgumentParser(description="Capture SciPy constants oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import constants as sc
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
        case_outputs.append(_run_case(case, np=np, sc=sc))

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
