# frankenscipy-zpunl Radau Diagonal Stage Solve Evidence

Date: 2026-06-20
Agent: cod-a / MistyBirch
Bead: `frankenscipy-zpunl`

## Decision

KEEP. Exactly diagonal Radau Jacobians now avoid dense `3n x 3n` stage-system
assembly/LU and dense real-shift error solves. The collocation equations split
into `n` independent `3 x 3` solves plus scalar real-shift solves. The dense
fallback remains unchanged for non-diagonal Jacobians.

This targets the measured remaining loss from the previous stiff-suite
gauntlet: Radau64 stiff diagonal decay was about 2.0x slower than SciPy after
the rejected streamed-Radau-norm candidate was reverted.

## Commands

Same-worker baseline:

```text
cd /data/projects/.scratch/frankenscipy-cod-a-zpunl-baseline-20260620
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-integrate --bin perf_integrate -- radau-stiff64 20
```

Final-source focused Rust:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-integrate --bin perf_integrate -- radau-stiff64 20
```

Final-source stiff suite:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-integrate --bin perf_integrate -- stiff-suite 10
```

SciPy oracle:

```text
python3 docs/perf_oracle_integrate_stiff.py bdf-stiff64,bdf-stiff128,radau-stiff32,radau-stiff64 10
python3 docs/perf_oracle_integrate_stiff.py radau-stiff64 20
```

## Same-Worker A/B

The baseline worktree was checked out at `8e8a5b453109915dbe948f3a44a8d19d1fbe709e`.
Both rows below ran on rch worker `ovh-a` with repeats=20.

| Route | Worker | per-call | nfev | njev | nlu | checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline dense Radau stage LU | `ovh-a` | 70047.428100 us | 20900 | 20 | 5600 | 3.234169482298e4 |
| diagonal stage/scalar solve candidate | `ovh-a` | 1124.530700 us | 20900 | 20 | 5600 | 3.234169482298e4 |

Internal result: `62.29x` faster, with unchanged counters and checksum.

## Final-Source SciPy Head-To-Head

Focused Radau64 final-source row after the clippy helper cleanup:

| Workload | Rust final source | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `radau-stiff64`, repeats=20 | 1687.783900 us | 36545.873049 us | 21.65x faster | SciPy win |

Full stiff-suite final-source row from rch worker `hz2`:

| Workload | Rust final source | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `bdf-stiff64` | 2306.348100 us | 25448.461401 us | 11.03x faster | SciPy win |
| `bdf-stiff128` | 12041.547000 us | 29874.872195 us | 2.48x faster | SciPy win |
| `radau-stiff32` | 591.275600 us | 33708.498604 us | 57.01x faster | SciPy win |
| `radau-stiff64` | 1306.515700 us | 36488.462600 us | 27.93x faster | SciPy win |

Final-source stiff-suite score versus SciPy: `4/0/0`.

## Correctness And Gates

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-integrate radau --lib -- --nocapture`:
  3 passed / 0 failed.
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-integrate --all-targets`:
  PASS.
- `rustfmt --edition 2024 --check crates/fsci-integrate/src/radau.rs`:
  PASS.
- `git diff --check`:
  PASS.
- `cargo clippy -p fsci-integrate --all-targets -- -D warnings` via rch:
  BLOCKED by pre-existing non-Radau lint debt only after the touched helper fix:
  `api.rs` too-many-arguments, `rk.rs` too-many-arguments/type-complexity, and
  `quad.rs` excessive-precision/type-complexity.
- `cargo test -p fsci-conformance --test e2e_ivp -- --nocapture` via rch:
  BLOCKED before IVP tests by unrelated `fsci-cluster` compile errors for
  missing `fsci_linalg::randomized_svd`, missing `fsci_linalg::randomized_eigh`,
  and one ambiguous float in `fsci-cluster/src/lib.rs`.

## Negative Evidence

- Do not retry the rejected Radau streamed scaled-RMS norm candidate. It already
  regressed same-worker Radau32/Radau64 and was reverted.
- The profitable lever was not norm materialization. The measured gap was dense
  stage-system construction/LU and real-shift solves for diagonal Jacobians.
- Further Radau work should target non-diagonal structured Jacobians, banded
  Jacobians, or LU reuse across accepted steps. It must preserve the same
  `nfev/njev/nlu` accounting and tolerance behavior.
