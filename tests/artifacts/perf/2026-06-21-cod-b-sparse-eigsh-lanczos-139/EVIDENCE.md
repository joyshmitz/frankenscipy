# frankenscipy-8l8r1.140 - sparse eigsh three-term Lanczos gauntlet

Date: 2026-06-21
Agent: cod-b / BlackThrush
Decision: REJECT AND RESTORE SOURCE
Note: renumbered from local `.139` during rebase because upstream already used
`.139` for an interpolate task; the artifact path retains the original capture
suffix.

## Target

Remaining `fsci-sparse::eigsh` loss after `frankenscipy-8l8r1.131`:
`eigsh n=8000 k=6` on the deterministic symmetric banded matrix.

Prior negative evidence ruled out row-major Arnoldi basis arenas, mutable
operator scratch, and unconditional residual removal. This pass tested a true
symmetric three-term Lanczos recurrence, then a stabilized variant that
reorthogonalized only numerically stray older directions.

## Commands

Parent baseline, same worker:

```bash
env AGENT_NAME=BlackThrush RCH_WORKER=hz1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-sparse --bin perf_eigsh
```

Candidate runs:

```bash
env AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-sparse --bin perf_eigsh

env AGENT_NAME=BlackThrush RCH_WORKER=hz1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo run --release -p fsci-sparse --bin perf_eigsh
```

SciPy oracle:

```bash
python3 - <<'PY'
# builds the same LCG symmetric banded matrix and runs scipy.sparse.linalg.eigsh
PY
```

## Results

### Parent baseline

`origin/main` / `17e29927`, same-worker baseline on `hz1`:

| Workload | Rust parent | Converged | Max residual |
| --- | ---: | --- | ---: |
| `eigsh n=2000 k=6` | 1.537 ms | true | 1.96e-11 |
| `eigsh n=8000 k=6` | 5.520 ms | true | 3.45e-11 |
| `eigsh n=20000 k=8` | 15.043 ms | true | 2.57e-11 |

### Pure three-term Lanczos

Same worker `vmi1153651`, parent then candidate:

| Workload | Parent | Pure Lanczos | Verdict |
| --- | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 2.475 ms | 1.213 ms | reject: `conv=false`, max residual 4.98e-1 |
| `eigsh n=8000 k=6` | 11.228 ms | 4.548 ms | reject: `conv=false`, max residual 7.41e-2 |
| `eigsh n=20000 k=8` | 25.224 ms | 15.016 ms | reject: `conv=false`, max residual 2.93e0 |

The raw recurrence is fast but produces ghost/duplicate Ritz values and fails
the residual contract. It cannot ship.

### Stabilized three-term Lanczos

The stabilized variant converged, but the target row was not a reliable win.

| Workload | Parent `hz1` | Candidate `hz1` samples | Candidate median | Internal verdict |
| --- | ---: | ---: | ---: | --- |
| `eigsh n=2000 k=6` | 1.537 ms | 1.182 / 1.200 / 1.181 ms | 1.182 ms | 1.30x faster |
| `eigsh n=8000 k=6` | 5.520 ms | 4.368 / 5.556 / 7.241 ms | 5.556 ms | reject: 1.01x slower on target median |
| `eigsh n=20000 k=8` | 15.043 ms | 12.317 / 16.617 / 12.507 ms | 12.507 ms | 1.20x faster, but already not the target loss |

Local SciPy 1.17.1 / NumPy 2.4.3 oracle:

| Workload | SciPy median | Restored final Rust / SciPy | Stabilized candidate median / SciPy |
| --- | ---: | ---: | ---: |
| `eigsh n=2000 k=6` | 1.267 ms | 1.21x slower | 1.07x faster |
| `eigsh n=8000 k=6` | 2.909 ms | 1.90x slower | 1.91x slower |
| `eigsh n=20000 k=8` | 6.316 ms | 2.38x slower | 1.98x slower |

The candidate does not close the target `n=8000, k=6` SciPy loss and adds
timing instability. Source was restored to the parent Arnoldi route.

## Negative evidence

Do not retry plain three-term Lanczos for `eigsh`; without ghost control it is
fast but violates the eigenpair residual contract. Do not ship the lightly
stabilized recurrence either: the target row is median-neutral/slower and still
about 1.9x slower than the local SciPy oracle. The next credible route is a real
implicitly restarted or thick-restarted symmetric Lanczos primitive with a
measured restart policy, not a no-restart recurrence rewrite.

## Verification notes

- Source diff after restoration: no `crates/fsci-sparse/src/linalg.rs` changes.
- A shell-loop attempt through `rch exec -- bash -lc ...` was rejected as a
  non-compilation command and fell back into a local target-dir rustc mismatch
  (`E0514`). No cleanup command was run.
