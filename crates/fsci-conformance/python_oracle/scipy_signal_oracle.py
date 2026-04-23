#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy signal fixture.

Closes the signal slice of frankenscipy-di9p. Covers savgol_filter,
the hann/hamming/blackman/kaiser window functions, convolve, correlate,
find_peaks, butter, freqz, hilbert, and detrend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _run_case(case: Dict[str, Any], np: Any, signal: Any, windows: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", [])

    try:
        if function == "savgol_filter":
            x = np.asarray(args[0], dtype=float)
            window_length = int(args[1])
            polyorder = int(args[2])
            result = signal.savgol_filter(x, window_length, polyorder)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function in {"hann", "hamming", "blackman"}:
            n = int(args[0])
            fn = getattr(windows, function)
            result = fn(n)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "kaiser":
            n = int(args[0])
            beta = float(args[1])
            result = windows.kaiser(n, beta)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "convolve":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            mode = args[2] if len(args) > 2 else "full"
            result = signal.convolve(a, b, mode=mode)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "correlate":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            mode = args[2] if len(args) > 2 else "full"
            result = signal.correlate(a, b, mode=mode)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "find_peaks":
            x = np.asarray(args[0], dtype=float)
            peaks, _props = signal.find_peaks(x)
            return _ok(case_id, "array", {
                "values": [int(v) for v in peaks.tolist()],
            })

        if function == "butter":
            n = int(args[0])
            wn = args[1]
            btype = args[2] if len(args) > 2 else "low"
            b, a = signal.butter(n, wn, btype=btype)
            return _ok(case_id, "ba_filter", {
                "b": [float(v) for v in np.asarray(b).tolist()],
                "a": [float(v) for v in np.asarray(a).tolist()],
            })

        if function == "freqz":
            b = np.asarray(args[0], dtype=float)
            a = np.asarray(args[1], dtype=float) if len(args) > 1 else np.array([1.0])
            worN = int(args[2]) if len(args) > 2 else 512
            w, h = signal.freqz(b, a, worN=worN)
            return _ok(case_id, "freqz_result", {
                "frequencies": [float(v) for v in w.tolist()],
                "response_real": [float(v.real) for v in h.tolist()],
                "response_imag": [float(v.imag) for v in h.tolist()],
            })

        if function == "hilbert":
            x = np.asarray(args[0], dtype=float)
            result = signal.hilbert(x)
            return _ok(case_id, "complex_array", {
                "real": [float(v.real) for v in result.tolist()],
                "imag": [float(v.imag) for v in result.tolist()],
            })

        if function == "detrend":
            x = np.asarray(args[0], dtype=float)
            type_ = args[1] if len(args) > 1 else "linear"
            result = signal.detrend(x, type=type_)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function}",
        }

    # See br-p3be: narrow catch. RuntimeError covers scipy.signal-raised
    # filter/window failures without swallowing MemoryError or OSError.
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


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy signal oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import signal
        from scipy.signal import windows
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
        case_outputs.append(_run_case(case, np=np, signal=signal, windows=windows))

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
