# frankenscipy-psn7x.4 native eigh stage profile

Agent: `RubyWaterfall`
Date: 2026-06-15
Crate: `fsci-linalg`
Worker: RCH selected `vmi1152480`

## Target

After the kept `psn7x.1` rank-2 update and the upstream `i3gnj` mirror-store
keep, the current native symmetric `eigh` wall needed a fresh stage split before
another source optimization.

## Change

Added ignored profiling test `symmetric_eigh_native_stage_breakdown_probe`.
It copies the production native `eigh` stages and prints:

- Householder reduction time
- tridiagonal eigensolver time
- eigenvector backtransform time
- eigenpair sort/materialization time
- worker count, max eigenvalue drift, and deterministic values digest

This is profiling instrumentation only; no production path changes.

## Profile

Artifact: `stage_profile_native_eigh_ovh_a_rch.txt`.

| n | reduction | tridiagonal eigensolver | backtransform | sort |
| ---: | ---: | ---: | ---: | ---: |
| 400 | `62.780826 ms` | `22.838759 ms` | `26.678778 ms` | `0.621096 ms` |
| 800 | `172.320292 ms` | `46.908303 ms` | `60.043983 ms` | `2.439279 ms` |
| 1200 | `543.771109 ms` | `96.610022 ms` | `241.994503 ms` | `6.661290 ms` |

The profile passed with max eigenvalue drift below the native tolerance:

- n=400: `5.24025267623073887e-13`
- n=800: `1.71418435002124170e-12`
- n=1200: `1.79412040779425297e-12`

## Route

Reduction remains the dominant measured wall at n=1200, with backtransform now
the second wall. Continue `frankenscipy-psn7x.4` by attacking a fundamentally
different reduction primitive: two-stage dense-to-band then band-to-tridiagonal
with scalar replay proof at panel boundaries.

## Gates

- `cargo fmt -p fsci-linalg -- --check`: passed.
- `check_fsci_linalg_lib_rch.txt`: `cargo check -j 1 -p fsci-linalg --lib --locked` passed on RCH.
- `clippy_fsci_linalg_lib_rch.txt`: `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings` passed on RCH.
- Known unrelated dependency warning: `fsci-fft/src/helpers.rs:58` unused `total` appears while compiling dependencies.
