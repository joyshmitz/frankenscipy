# frankenscipy-x9ckc - jnjnp_zeros Root-Cost Lever

Date: 2026-06-19
Agent: cod-b / MistyBirch
Worktree: `/data/projects/.scratch/frankenscipy-cod-b-bold-clean-20260619T104351Z`
Base: `835db7f3`

## Lever

`jnjnp_zeros` was still dominated by per-root scalar Bessel work after the
output-sensitive frontier landed. This pass keeps the existing bracket safety
contract but reduces the constant factor inside the zero primitive:

- direct integer-order `J_n` evaluation for `jn_zeros` instead of generic
  strict-mode scalar dispatch;
- direct integer-order `J_n'` identity for `jnp_zeros`;
- guarded secant refinement for generic bracketed roots;
- guarded Newton refinement for `J_n` and `J_n'` roots using Bessel derivative
  identities, falling back to bracketed refinement if the correction is not
  finite or leaves the bracket.

No unsafe code was introduced.

## Rust A/B

Command:

```bash
RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot
```

Worker: `hz1`

| Workload | Baseline current mean | Candidate current mean | Candidate/baseline | Verdict |
| --- | ---: | ---: | ---: | --- |
| `jnjnp_zeros(nt=64)` | 5.4622 ms | 4.6666 ms | 0.854x time, 1.17x faster | keep |
| `jnjnp_zeros(nt=128)` | 9.9958 ms | 8.3620 ms | 0.837x time, 1.20x faster | keep |

Legacy duplicate comparison rows from the same benchmark stayed ahead of the
original baseline as well: `nt=64` was 126.59 ms vs original 132.38 ms, and
`nt=128` was 611.57 ms vs original 694.51 ms.

## SciPy Oracle

The rch worker could not import `scipy.special`, so the SciPy oracle was timed
locally with SciPy 1.17.1:

```bash
python3 - <<'PY'
import time
import scipy.special as s
for nt, iters in [(64, 400), (128, 300)]:
    s.jnjnp_zeros(nt)
    start = time.perf_counter()
    for _ in range(iters):
        s.jnjnp_zeros(nt)
    elapsed = (time.perf_counter() - start) / iters
    print(f"scipy_jnjnp_zeros_nt{nt}: {elapsed*1e6:.2f} us ({iters} iters)")
PY
```

| Workload | Candidate Rust mean | SciPy mean | Candidate/SciPy | Verdict |
| --- | ---: | ---: | ---: | --- |
| `jnjnp_zeros(nt=64)` | 4.6666 ms | 439.49 us | 10.62x slower | residual loss |
| `jnjnp_zeros(nt=128)` | 8.3620 ms | 787.18 us | 10.62x slower | residual loss |

SciPy win/loss/neutral: `0/2/0`.
Same-worker internal keep/loss/neutral: `2/0/0`.

## Correctness

```bash
RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-special jnjnp -- --nocapture
```

Result: PASS. The two focused guards passed:

- `bessel::tests::jnyn_and_jnjnp_zeros_match_scipy`
- `bessel::tests::jnjnp_adaptive_envelope_matches_oversized_reference`

## Decision

KEEP. The lever is a credible same-worker root-cost reduction and preserves the
existing zero-order/tolerance tests, but it does not close the SciPy gap. The
next retry should not retune the same guarded Newton/secant primitive unless a
profile shows it still dominates. Route deeper to a SciPy-style global zero
enumerator, a batched recurrence/evaluation cache across neighboring orders, or
a generated lower-constant integer-order Bessel kernel.
