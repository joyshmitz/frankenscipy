# frankenscipy-8l8r1.135 - filter1d contiguous Reflect direct queue

Date: 2026-06-20
Agent: cod-b / BlackThrush
Crate: `fsci-ndimage`
Decision: KEEP

## Lever

Specialize the public `maximum_filter1d` / `minimum_filter1d` queue route for
contiguous `Reflect`, `origin=0`, `size <= line_len` lines.

The previous queue materialized a full boundary-resolved `ext` line and kept a
full-length index queue. The kept path computes the reflected source index only
when an element enters/leaves the window and uses a circular monotonic deque
sized to the window (`size + 1`). All other cases keep the generic
boundary-resolved queue.

Alien/perf source: this is the narrow-interface version of the fixed-ring-buffer
idea from the alien graveyard's ring-queue section, constrained by the
"constants kill you" warning: do not change complexity or public API; remove the
extra linear buffer and pointer traffic only for the measured contiguous case.

## Isomorphism proof

- Scope gate: `mode == Reflect`, `origin == 0`, contiguous axis (`inner == 1`),
  and `size <= line_len`. Every other input takes the prior generic path.
- Boundary mapping: `ext[t]` still represents coordinate `t - size/2`; for the
  gated case only one reflection is possible, so `reflect_origin0_ext_index`
  is algebraically identical to `boundary_index_1d`.
- Ordering/ties: queue eviction still uses the existing `<=` / `>=` predicate,
  preserving newest-tie signed-zero behavior.
- NaNs: NaN values are counted out-of-band exactly as before; windows with any
  NaN emit the canonical `f64::NAN`.
- Unsafe: none added.

## Head-to-head evidence

Fresh current baseline before source edit:

```bash
AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- minmax_filter1d --noplot
```

The worker hint was scheduled on `hz1`; results are recorded in
`baseline_minmax_filter1d.txt`.

| Workload | Current Criterion mean (`hz1`) |
| --- | ---: |
| `maximum_filter1d`, n=65536 size=31 | 760.87 us |
| `minimum_filter1d`, n=65536 size=31 | 1.0810 ms |
| `maximum_filter1d`, n=65536 size=101 | 966.29 us |
| `minimum_filter1d`, n=65536 size=101 | 1.0142 ms |

Same-process A/B after the patch, pinned to `hz1`, compares the prior generic
queue body to the direct queue and asserts bit identity before timing:

| Workload | Generic queue | Direct queue | Internal ratio |
| --- | ---: | ---: | ---: |
| max size=31 | 1116.8 us | 470.7 us | 2.37x faster |
| min size=31 | 1094.4 us | 465.8 us | 2.35x faster |
| max size=101 | 1089.1 us | 464.2 us | 2.35x faster |
| min size=101 | 1091.0 us | 466.8 us | 2.34x faster |

After Criterion rows:

```bash
AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- minmax_filter1d --noplot
```

The worker hint was scheduled on `vmi1149989`; treat this as absolute after
evidence, not the internal keep gate.

| Workload | After Criterion mean (`vmi1149989`) |
| --- | ---: |
| `maximum_filter1d`, n=65536 size=31 | 344.48 us |
| `minimum_filter1d`, n=65536 size=31 | 339.06 us |
| `maximum_filter1d`, n=65536 size=101 | 339.74 us |
| `minimum_filter1d`, n=65536 size=101 | 321.55 us |

Local SciPy 1.17.1 oracle:

| Workload | SciPy median | Conservative direct-vs-SciPy | Criterion after-vs-SciPy |
| --- | ---: | ---: | ---: |
| max size=31 | 524.98 us | Rust 1.12x faster | Rust 1.52x faster |
| min size=31 | 575.42 us | Rust 1.24x faster | Rust 1.70x faster |
| max size=101 | 529.05 us | Rust 1.14x faster | Rust 1.56x faster |
| min size=101 | 592.31 us | Rust 1.27x faster | Rust 1.84x faster |

Scores:

- Same-process internal: `4/0/0`.
- Conservative SciPy score using direct A/B times: `4/0/0`.
- Absolute Criterion/SciPy score: `4/0/0`.

## Gates

| Gate | Result | Artifact |
| --- | --- | --- |
| Fold/generic byte identity | PASS | `filter1d_byte_identity.txt` |
| Direct/generic same-process A/B + bit identity | PASS | `filter1d_direct_vs_generic_ab.txt` |
| rch Criterion baseline | PASS | `baseline_minmax_filter1d.txt` |
| rch Criterion after | PASS | `after_minmax_filter1d.txt` |
| Local SciPy oracle | PASS | `filter1d_local_scipy.txt` |
| Live SciPy conformance | PASS | `diff_ndimage_filter_1d.txt` |
| Per-crate check | PASS | `cargo_check_fsci_ndimage.txt` |
| Per-crate release build | PASS | `cargo_build_release_fsci_ndimage.txt` |
| Touched-file rustfmt | PASS | `rustfmt_check_touched.txt` |
| Diff hygiene | PASS | `git_diff_check.txt` |
| Changed-file UBS | PASS | `ubs_changed_files.txt`; exit 0, 0 critical issues, broad existing `fsci-ndimage` warning inventory remains |
| Strict clippy | BLOCKED | `cargo_clippy_fsci_ndimage.txt`; existing `fsci-linalg` dependency lints stop before this patch |

## Negative evidence / retry rule

This closes the tracked filter1d residual for the contiguous 1-D Reflect rows.
Do not retry full-line `ext` materialization or whole-line queue storage for this
workload. Future work should target non-contiguous axes, `size > line_len`, or
additional SciPy max/min filter1d conformance coverage rather than another
contiguous Reflect queue micro-variant.
