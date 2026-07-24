# Cholesky-wall CYCLES self-time gate — built + validated (frankenscipy-64wo0)

Date: 2026-07-23 · Agent: CopperFalcon (cc) · Host: rch worker AMD Ryzen Threadripper PRO 5975WX (32c/64t, 128 MiB L3) · Release + `chol-wall-bench`, rustflags `+avx2,+fma`

## Why (baseline + profile)

A fresh `perf record` of the blocked n=1000 Cholesky factor on the fleet showed the
kernels are BALANCED — `cholesky_syrk_flat_rows_mr4_nr8_fma` ~7% ≈
`cholesky_panel_trsm_blocked_fma_rows` ~6%, data movement ~7.7% — with no fat
single-kernel frame. Every remaining kernel-side lever (pack fusion of
`copy_l21_and_pack_transpose_into` into the SYRK first pass; scratch reuse) is only
a few % of the blocked path, i.e. BELOW the ±5% wall-clock A/B floor at n=1000
(subprocess launch + OS scheduling jitter on a shared 64-thread box). Those levers
are un-measurable — and thus un-landable — on wall-clock. `perf stat -e cycles`
counts RETIRED work across all threads, immune to scheduling, so its A/A null floor
is ~±2-4% — decidable for the sub-floor inventory. Prior art: d97283534.

## The gate

- `crates/fsci-linalg/src/bin/perf_chol_cycles_gate.rs` — runs ONE arm `reps` times
  over a cheap O(n²) SPD matrix (setup ≪ O(n³) factor ⇒ no ratio dilution); prints a
  checksum. `full` digest mode folds every element (execution proof); `light`
  (measured runs) folds a strided sample only (defeats DCE without adding common
  O(n²) work that would compress the ratio).
- `crates/fsci-linalg/benches/chol_cycles_gate.sh` — wraps each invocation in
  `perf stat -e cycles -x,`, interleaves base/cand/nullb, and reports median
  cycle-ratios + an A/A null. `rch exec -- bash crates/fsci-linalg/benches/chol_cycles_gate.sh`.

## Validation (against the known FMA-SYRK lever)

Arms isolate the trailing-SYRK kernel (both share TRSM_ROWS2 + `chol_nb_for(n)`):
base = `cholesky_wall_mr4_nr8_orig` (plain mul+add), cand =
`cholesky_wall_mr4_nr8_fma_candidate` (fused `mul_add`). N=1000, REPS=16, K=21:

```
EXEC_PROOF base digest=0xe1f923aa30e1bfa3   cand digest=0x16845c947fa4dde5   (differ ⇒ arm switch live)
base cycles median = 1.0992e9  cv 0.98%
cand cycles median = 1.0406e9  cv 1.45%
LEVER base/cand = 1.0563x   (FMA-SYRK = 5.63% fewer cycles)
NULL  base/nullb median = 1.0000   range [0.9634, 1.0237]
VERDICT: DECIDED (lever 1.0563 > null_hi 1.0237)
```

The gate reproduces the FMA-SYRK lever (its wall-clock verdict was ~1.143x on the
older thinkstation1 box; on cycles on this Threadripper it is ~1.05-1.06x of retired
work) and DECIDES it against a null centred exactly at 1.0000 with a ~±2-4% range —
substantially tighter than wall-clock's ±5% at n=1000, exactly the tightening the
prior art promised. A repeat run gave 1.0525x / null [0.9613, 1.0167] — consistent.

## Unlocks

The gate makes the sub-floor SYRK / data-movement inventory measurable and landable.
To gate a NEW lever, point the two arms in `perf_chol_cycles_gate.rs` at the new
baseline / candidate factor fns and re-run. Next target: pack fusion (fold
`copy_l21_and_pack_transpose_into` into the SYRK first pass — ~5% of the blocked
path, previously sub-floor).
