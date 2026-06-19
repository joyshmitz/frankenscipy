# frankenscipy-wm14d residual order=1 zoom evidence

- Agent: cod-b / MistyBirch
- Date: 2026-06-19
- Crate: `fsci-ndimage`
- Lane: residual `scipy.ndimage.zoom` order=1 2D Reflect gap after the first
  `wm14d` cardinal fast path.

## Lever

Added a 2D Reflect/order=1 `zoom` fast path that precomputes separable row and
column linear supports once, then evaluates each output pixel with a fixed
four-load bilinear sum over the already-prefiltered padded coefficient image.

The fast path preserves the existing prefilter and coordinate-offset contract;
it only replaces the generic per-pixel recursive sampler for the narrow 2D
linear Reflect case.

## Same-worker internal A/B

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- zoom/2x_256/1 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

| Workload | Baseline current mean | Candidate mean | Candidate/baseline | Verdict |
| --- | ---: | ---: | ---: | --- |
| `zoom/2x_256/order=1` | 34.034 ms | 7.9684 ms | 0.234x time, 4.27x faster | keep |

Raw logs:

- `baseline_zoom_order1_ovh-b_rch.txt`
- `candidate_zoom_order1_ovh-b_rch.txt`

## Head-to-head vs SciPy

SciPy oracle command:

```bash
python3 docs/perf_oracle_zoom.py
```

| Workload | Candidate Rust mean | SciPy median | Candidate/SciPy | Verdict |
| --- | ---: | ---: | ---: | --- |
| `ndimage.zoom(256x256, 2x, order=1)` | 7.9684 ms | 3.88937 ms | 2.05x slower | residual loss |

SciPy win/loss/neutral: `0/1/0`.
Same-worker internal keep/loss/neutral: `1/0/0`.

## Rejected sub-variant

Tested a serial fill loop after the fixed four-load bilinear fast path. It lost
to the kept parallel fill:

| Variant | Mean | Versus kept path | Decision |
| --- | ---: | ---: | --- |
| kept parallel fill | 7.9684 ms | 1.00x | keep |
| serial fill probe | 9.6976 ms | 1.22x slower | reject and revert |

Raw log: `candidate_serial_zoom_order1_ovh-b_rch.txt`.

## Correctness gates

- PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage zoom_order_one_reflect_fast_path_matches_generic_sampler_bits -- --nocapture`
  - `1 passed; 0 failed; 238 filtered out`
- PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-ndimage zoom_ -- --nocapture`
  - `6 passed; 0 failed; 233 filtered out`; metamorphic `mr_zoom_by_one_is_identity` also passed.
- PASS: `FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-conformance --test diff_ndimage_zoom -- --nocapture`
  - `1 passed; 0 failed` against the live local SciPy oracle.
- PASS: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-ndimage --all-targets`.
- PASS: `git diff --check`.
- PASS: `ubs crates/fsci-ndimage/src/lib.rs docs/progress/perf-negative-results.md docs/GAUNTLET_RELEASE_SCORECARD.md docs/perf_ledger_cc.md tests/artifacts/perf/frankenscipy-wm14d-residual-20260619/EVIDENCE.md`
  - no critical findings; warnings are the existing broad ndimage inventory.
- BLOCKED: `cargo fmt -p fsci-ndimage --check` and
  `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs` still report
  pre-existing formatting drift outside this patch, including
  `ndimage_bench.rs`, `diff_fourier.rs`, and older `src/lib.rs` assertions and
  helper formatting.
- BLOCKED: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo clippy -p fsci-ndimage --all-targets -- -D warnings`
  stops on existing `fsci-linalg` dependency lints before checking this patch.
- BLOCKED: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-b CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo clippy -p fsci-ndimage --lib --no-deps -- -D warnings`
  still stops on existing `fsci-ndimage` library lint debt unrelated to this
  fast path (`type_complexity`, `needless_range_loop`, `too_many_arguments`,
  and `collapsible_if` in older helpers).

## Negative evidence

The direct separable-support fast path is a real 4.27x internal win, but Rust
still loses to SciPy by 2.05x on this focused 2D linear zoom workload. Do not
retry the serial-fill micro-lever without a fresh profile; it was measured and
reverted.

Next useful routes are deeper than loop-shape tweaks: remove the padded
prefilter cost for order=1 when equivalence can be proved, code-generate/tile
the geometric order=1 family, or SIMD the contiguous row interpolation while
preserving the current tolerance and boundary contracts.
