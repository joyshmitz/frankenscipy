# frankenscipy-wh8ac - jnjnp_zeros Cephes j1 seed evaluator

Date: 2026-06-20
Agent: cod-b / MistyBirch
Decision: KEEP

## Lever

Replace the local `j1_core` small-series plus generic asymptotic split with the
Cephes fixed rational kernels used by SciPy's `j1` ufunc:

- `[0, 5]`: rational approximation around the first two J1 zeros.
- `(5, inf)`: 5/x asymptotic rational correction.

This is the same constant-kernel direction that paid for `j0`: remove the
variable convergence/large-order seed noise in a hot `jnjnp_zeros` evaluator
without relaxing the SciPy-observable tolerance contract.

Source anchors:

- SciPy `scipy.special.j1` documentation states the wrapper uses the Cephes
  routine and the same two-domain split.
- Netlib Cephes `bessel.tgz`, `j1.c`, supplied the coefficients used here.

## Head-to-head evidence

Baseline and candidate Criterion rows were run with:

```bash
RCH_REQUIRE_REMOTE=1 \
RCH_WORKER=vmi1149989 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
rch exec -- cargo bench -p fsci-special --bench special_bench -- acoco_gauntlet_jnjnp_zeros --noplot
```

The initial baseline request used the same target dir and was scheduled on
`vmi1149989`; the candidate rerun was pinned to the same worker.

| Workload | Baseline Rust | Candidate Rust | Internal speedup | SciPy oracle | Candidate vs SciPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jnjnp_zeros(nt=64)` | 608.21 us | 381.89 us | 1.59x faster | 463.913 us | 1.22x faster |
| `jnjnp_zeros(nt=128)` | 1.1970 ms | 742.06 us | 1.61x faster | 832.786 us | 1.12x faster |

Criterion candidate change estimates:

- `nt=64`: `[−41.822% −39.188% −36.402%]`, p=0, improved.
- `nt=128`: `[−41.699% −38.076% −34.453%]`, p=0, improved.

SciPy oracle command:

```bash
python3 - <<'PY'
import scipy, scipy.special as sp, time
for nt, iters in [(64, 2000), (128, 1200)]:
    sp.jnjnp_zeros(nt)
    t0 = time.perf_counter()
    for _ in range(iters):
        sp.jnjnp_zeros(nt)
    dt = (time.perf_counter() - t0) / iters * 1e6
    print(nt, dt)
PY
```

Local SciPy version: 1.17.1.

SciPy win/loss/neutral for this sub-cluster: `2/0/0`.

## Correctness and gates

| Gate | Result | Notes |
| --- | --- | --- |
| Focused `jnjnp` tests | PASS | `cargo test -p fsci-special jnjnp -- --nocapture` via rch `hz1`: 3 passed / 0 failed |
| Focused `j1` reference tests | PASS | `cargo test -p fsci-special j1_matches_scipy_reference_values -- --nocapture` via rch `ovh-a`: 1 passed / 0 failed |
| Focused `kve` test cleanup guard | PASS | `cargo test -p fsci-special complex_kve_matches_scipy -- --nocapture` via rch `hz1`: 1 passed / 0 failed |
| Live SciPy conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_special_bessel_zeros -- --nocapture`: 1 passed / 0 failed |
| Per-crate compile | PASS | `cargo check -p fsci-special --all-targets` via rch `hz1`; existing warnings only |
| Diff hygiene | PASS | `git diff --check` |
| Changed-file UBS | PASS | `ubs` on changed files: 0 critical issues after removing a pre-existing test-only `panic!` in touched `bessel.rs`; warnings remain inventory |
| rch SciPy rows | PARTIAL | rch workers for the Criterion bench could not import `scipy.special`; local SciPy oracle supplied the head-to-head |
| `cargo fmt -p fsci-special --check` | BLOCKED | pre-existing rustfmt drift in multiple `fsci-special` files outside this patch |
| `cargo clippy -p fsci-special --all-targets -- -D warnings` | BLOCKED | existing dependency lints in `fsci-integrate` and `fsci-linalg` stop before this patch |

## Negative evidence and retry rule

Do not retry local J1 power-series/asymptotic split tuning for this workload.
The fixed Cephes rational evaluator is both faster and closer to the incumbent
oracle surface. The next `jnjnp_zeros` pass, if needed, should profile below the
remaining root-generation/frontier logic rather than another Bessel J1
approximation micro-variant.
