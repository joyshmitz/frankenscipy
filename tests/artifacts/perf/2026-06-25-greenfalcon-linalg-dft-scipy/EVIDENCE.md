# fsci-linalg dft SciPy gauntlet

- Date: 2026-06-25
- Agent: GreenFalcon
- Crate: `fsci-linalg`
- Surface: `fsci_linalg::dft(n, None)` vs `scipy.linalg.dft(n)`.
- Decision: KEEP the existing root-precompute DFT lever and land the reusable SciPy comparator rows.

## Context

`main` already contains the source lever from commit `092a9e98`: `dft` precomputes the `n`
distinct scaled roots of unity for the reduced index `(j * k) % n` instead of calling
`cos`/`sin` in every matrix cell. This evidence pass adds direct BOLD-VERIFY rows against SciPy.

The separate `dft_matrix` helper is intentionally not changed because it uses the unreduced
`j * k` angle and would not be bit-identical under the same table lookup.

## Environment

```text
AGENT_NAME=GreenFalcon
RUSTUP_TOOLCHAIN=nightly-2026-06-10
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
local Python: python3
local SciPy comparator: scipy.linalg.dft
```

## Commands

Reusable Criterion gauntlet:

```text
cargo bench -p fsci-linalg --bench linalg_bench -- dft_gauntlet_scipy --noplot
```

Direct A/B and large spot check:

```text
cargo run --release -p fsci-linalg --bin perf_dft_ab
python3 - <<'PY'
import time
import numpy as np
import scipy.linalg as la
for n in [256, 512, 1024, 2048]:
    la.dft(n)
    best = 1e99
    checksum = 0.0
    for _ in range(5):
        t = time.perf_counter()
        m = la.dft(n)
        best = min(best, time.perf_counter() - t)
        checksum += float(np.real(m[1, 1])) + float(np.imag(m[1, 1]))
    print(n, best * 1e6, checksum)
PY
```

## Measurements

Criterion comparator rows:

| n | Rust median | SciPy median | Rust vs SciPy |
| ---: | ---: | ---: | ---: |
| 256 | 536.50 us | 8.2442 ms | 15.37x faster |
| 512 | 2.4323 ms | 40.323 ms | 16.58x faster |
| 1024 | 10.831 ms | 175.06 ms | 16.16x faster |

Raw Criterion output:

```text
dft_gauntlet_scipy/256_rust_dft
time:   [524.52 us 536.50 us 552.55 us]

dft_gauntlet_scipy/256_scipy_dft
time:   [7.7472 ms 8.2442 ms 8.7294 ms]

dft_gauntlet_scipy/512_rust_dft
time:   [2.3819 ms 2.4323 ms 2.4876 ms]

dft_gauntlet_scipy/512_scipy_dft
time:   [38.600 ms 40.323 ms 41.997 ms]

dft_gauntlet_scipy/1024_rust_dft
time:   [10.648 ms 10.831 ms 11.096 ms]

dft_gauntlet_scipy/1024_scipy_dft
time:   [169.75 ms 175.06 ms 180.16 ms]
```

Direct best-of-5 spot check:

| n | Rust current | SciPy best | Rust vs SciPy |
| ---: | ---: | ---: | ---: |
| 256 | 173.68 us | 6.85487 ms | 39.47x faster |
| 512 | 1.11034 ms | 34.17845 ms | 30.78x faster |
| 1024 | 5.69224 ms | 157.29818 ms | 27.63x faster |
| 2048 | 20.23661 ms | 695.60084 ms | 34.37x faster |

The direct Rust rows come from the existing same-process A/B helper and use tuple complex
values, so the reusable Criterion public-API rows above are the primary BOLD-VERIFY result.

## Behavior Proof

- Ordering preserved: yes. Matrix cells are written in the same row-major loop order.
- Tie-breaking/RNG: N/A.
- Floating-point: byte-identical to the previous `dft` path because both compute
  `theta = base * ((j * k) % n)` and use the same `cos`/`sin` result for each reduced index.
- Fallback/rollback: revert the benchmark/evidence commit if the comparator harness itself
  causes maintenance trouble; the source lever is already on `main` and separately proved exact.

## Validation

```text
rustfmt --check crates/fsci-linalg/benches/linalg_bench.rs
PASS

git diff --check
PASS

ubs crates/fsci-linalg/benches/linalg_bench.rs docs/NEGATIVE_EVIDENCE.md tests/artifacts/perf/2026-06-25-greenfalcon-linalg-dft-scipy/EVIDENCE.md
PASS: exit 0, critical issues 0

rch exec -- cargo test -p fsci-linalg dft --lib -- --nocapture
worker: vmi1149989
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 526 filtered out

rch exec -- cargo check -p fsci-linalg --all-targets
worker: hz2
PASS: exit 0
warning: pre-existing `unused_mut` in crates/fsci-linalg/src/bin/perf_control.rs

cargo clippy -p fsci-linalg --benches -- -D warnings
BLOCKED: pre-existing clippy debt outside the touched benchmark rows in crates/fsci-linalg/src/lib.rs
and crates/fsci-linalg/src/cossin.rs.
```

## Follow-up

Do not apply this exact reduced-root table to `dft_matrix` without changing its contract or
accepting non-bit-identical output: `dft_matrix` currently uses the unreduced `j * k` angle.
