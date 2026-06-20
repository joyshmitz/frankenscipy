# ndimage gaussian_filter inner1 reflect reject

- Date: 2026-06-20
- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-6l77z`
- Decision: REJECT AND REVERT
- Source status: no production code from this attempted fast path is staged.

## Lever

Specialize `convolve1d_along_axis` for the row-contiguous
`inner == 1`, `BoundaryMode::Reflect`, odd-kernel, `origin == 0` path used by
the final axis of `gaussian_filter(..., sigma=2.0)` on 2-D images. The candidate
kept border samples on the existing boundary mapper and used direct in-bounds
indexing for the interior dot.

This tested a narrower version of the cache/branch-removal idea than the earlier
always-line-walk rejection: the outer-axis fallback was left intact.

## Commands

Baseline/current Rust:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot
```

Candidate Rust:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot
```

SciPy oracle:

```bash
AGENT_NAME=MistyBirch python3 docs/perf_oracle_ndimage.py
```

Focused gaussian guard:

```bash
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo test -p fsci-ndimage gaussian_filter1d_matches_scipy_axis1_reflect --lib -- --nocapture
```

Live SciPy ndimage conformance:

```bash
AGENT_NAME=MistyBirch FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-conformance --test diff_ndimage -- --nocapture
```

## Benchmark Results

| Workload / route | Mean | Interval | Verdict |
| --- | ---: | ---: | --- |
| Current Rust `gaussian_sigma2/256`, rch `hz2` | 3.4399 ms | [3.3426, 3.5375] ms | retained current; 3.03x slower than SciPy |
| Candidate inner1 reflect direct interior dot, rch `hz2` | 4.0213 ms | [3.8424, 4.1989] ms | reject; 1.17x slower than current, 3.54x slower than SciPy |
| SciPy `ndimage.gaussian_filter`, local SciPy 1.17.1 | 1.13557 ms | p50 | oracle |

Same-worker candidate/current score: `0/1/0`.
Final restored current/SciPy score: `0/1/0`.

## Correctness

- PASS: focused gaussian guard on rch `hz2`: `1 passed / 0 failed`.
- PASS: local live SciPy ndimage conformance: `5 passed / 0 failed`.
- PASS: candidate source was reverted before staging.

## Negative Evidence

Do not retry another scalar `inner == 1` reflect-only interior/border split for
this gaussian workload without a fresh profile proving the old closure/boundary
branch is still dominant. The next plausible route is a deeper separable-layout
primitive: transpose/cache-tile between axes so both passes are contiguous, or
introduce a shared vector-friendly dot kernel with a same-worker A/B against the
restored current route.
