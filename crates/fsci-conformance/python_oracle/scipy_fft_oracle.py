#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy FFT packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _to_complex_list(arr: Any) -> List[List[float]]:
    """Convert numpy array to [[real, imag], ...] format."""
    result = []
    for val in arr.flat:
        result.append([float(val.real), float(val.imag)])
    return result


def _to_real_list(arr: Any) -> List[float]:
    """Convert numpy array to list of floats."""
    return [float(v) for v in arr.flat]


def _complex_input_to_numpy(data: List[List[float]], np: Any) -> Any:
    """Convert [[real, imag], ...] to numpy complex array."""
    return np.array([complex(r, i) for r, i in data], dtype=np.complex128)


def _run_case(case: Dict[str, Any], fft: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    transform = case.get("transform", "fft")
    normalization = case.get("normalization", "backward")

    try:
        # Build input array
        if case.get("complex_input") is not None:
            x = _complex_input_to_numpy(case["complex_input"], np)
        elif case.get("real_input") is not None:
            x = np.array(case["real_input"], dtype=np.float64)
        else:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "missing_input",
                "result": {},
                "error": "no input provided",
            }

        n = case.get("output_len")

        # Execute transform
        if transform == "fft":
            result = fft.fft(x, n=n, norm=normalization)
        elif transform == "ifft":
            result = fft.ifft(x, n=n, norm=normalization)
        elif transform == "rfft":
            result = fft.rfft(x, n=n, norm=normalization)
        elif transform == "irfft":
            result = fft.irfft(x, n=n, norm=normalization)
        elif transform == "dct":
            dct_type = case.get("dct_type", 2)
            result = fft.dct(x.real if np.iscomplexobj(x) else x, type=dct_type, norm=normalization)
        elif transform == "idct":
            dct_type = case.get("dct_type", 2)
            result = fft.idct(x.real if np.iscomplexobj(x) else x, type=dct_type, norm=normalization)
        elif transform == "dst":
            dst_type = case.get("dst_type", 2)
            result = fft.dst(x.real if np.iscomplexobj(x) else x, type=dst_type, norm=normalization)
        elif transform == "idst":
            dst_type = case.get("dst_type", 2)
            result = fft.idst(x.real if np.iscomplexobj(x) else x, type=dst_type, norm=normalization)
        elif transform == "fft2":
            shape = tuple(case["shape"]) if case.get("shape") else None
            result = fft.fft2(x.reshape(case.get("input_shape", x.shape)), s=shape, norm=normalization)
        elif transform == "ifft2":
            shape = tuple(case["shape"]) if case.get("shape") else None
            result = fft.ifft2(x.reshape(case.get("input_shape", x.shape)), s=shape, norm=normalization)
        elif transform == "fftn":
            shape = tuple(case["shape"]) if case.get("shape") else None
            result = fft.fftn(x.reshape(case.get("input_shape", x.shape)), s=shape, norm=normalization)
        elif transform == "ifftn":
            shape = tuple(case["shape"]) if case.get("shape") else None
            result = fft.ifftn(x.reshape(case.get("input_shape", x.shape)), s=shape, norm=normalization)
        elif transform == "fftfreq":
            n_pts = int(case.get("n_points", len(x)))
            d = case.get("sample_spacing", 1.0)
            result = fft.fftfreq(n_pts, d=d)
        elif transform == "rfftfreq":
            n_pts = int(case.get("n_points", len(x)))
            d = case.get("sample_spacing", 1.0)
            result = fft.rfftfreq(n_pts, d=d)
        elif transform == "fftshift":
            result = fft.fftshift(x)
        elif transform == "ifftshift":
            result = fft.ifftshift(x)
        elif transform == "hilbert":
            from scipy import signal
            result = signal.hilbert(x.real if np.iscomplexobj(x) else x)
        else:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "unsupported_transform",
                "result": {},
                "error": f"unsupported transform: {transform}",
            }

        # Format output
        if np.iscomplexobj(result):
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "complex_vector",
                "result": {"values": _to_complex_list(result)},
                "error": None,
            }
        else:
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "real_vector",
                "result": {"values": _to_real_list(result)},
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
    parser = argparse.ArgumentParser(description="Capture SciPy FFT oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=True, help="Legacy oracle root path")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import fft
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
        case_outputs.append(_run_case(case, fft=fft, np=np))

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
