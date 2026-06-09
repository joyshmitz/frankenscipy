# frankenscipy-2a9bz keep: row-chunked right-reflector replay

## Target

- Bead: `frankenscipy-2a9bz`
- Profile source: `tests/artifacts/perf/2026-06-08-linalg-post-head-reprofile/public_routes_post_head_rch.txt`
- Public residual: 512x512 square `svd()` remained the largest public SVD-family route at `224.479806 ms` on `ovh-a` after the previous deterministic square route, with reconstruction error `9.88027437642813311e-11` and singular-value / `svdvals` max diff `4.07453626394271851e-10`.
- Stage profile: `tests/artifacts/perf/2026-06-09-linalg-square-svd-stage/baseline_stage_breakdown_rch.txt`

Baseline stage breakdown on `vmi1227854`:

```text
reduction_ms=72.404
bidiagonal_svd_ms=55.906
back_transform_u_ms=16.670
back_transform_v_ms=66.175
```

The profiled lever is the right-reflector back-transform, not the ordering/sign/rank policy.

## Lever

One safe-Rust lever: replay right Householder reflectors over independent row chunks in a row-major scratch buffer, then copy back into the existing column-major `DMatrix`.

This is a communication-avoiding/cache-layout lever from the dense linear algebra no-gaps route. It does not replace the SVD algorithm, singular-value ordering, tolerance gates, sign canonicalization, RNG policy, or public route guards.

## Isomorphism Proof

- Ordering preserved: every row applies the same right reflectors in the same reverse order as the sequential path.
- Tie-breaking preserved: singular values and their tie/order policy come from the unchanged bidiagonal SVD stage.
- Floating point preserved: within a row, each dot product and update walks `reflector.values` in the same ascending offset order as `apply_householder_right`; rows are independent, so chunk scheduling cannot change arithmetic for any row.
- Sign policy preserved: `canonicalize_svd_factor_signs` runs unchanged after reflector replay.
- RNG preserved: none used.
- Bit proof: `thin_bidiag_parallel_right_replay_matches_serial_bits` passed over `(256,160)`, `(384,192)`, and `(512,512)`, comparing the thin-SVD digest, every singular value, every `U` entry, and every `Vt` entry by `to_bits()`.
- Public golden proof: `public_svd_lstsq_pinv_golden_payload` stayed at SHA-256 `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`.

## Benchmarks

Stage after run on `vmi1227854`:

```text
worker_count=10
reduction_ms=75.967
bidiagonal_svd_ms=51.810
back_transform_u_ms=23.393
back_transform_v_ms=26.628
```

- Target stage: `back_transform_v_ms 66.175 -> 26.628`, `2.485166x`.
- Full profiled stage sum: `211.155 ms -> 177.798 ms`, `1.187612x`.
- Public square route on `ovh-a`: `224.479806 ms -> 169.059034 ms`, `1.327819x`.
- Current public proof output after the lever:

```text
shape=512x512
previous_route_ms=1032.528330
routed_ms=169.059034
speedup_vs_previous_route=6.107502
recon_err=9.88027437642813311e-11
singular_value_max_diff_vs_nalgebra=4.07453626394271851e-10
svdvals_max_diff_vs_svd=4.07453626394271851e-10
```

## Score

`Impact 3 * Confidence 5 / Effort 2 = 7.5`.

Keep. The win is on the profiler-selected largest public SVD-family residual, the isolated stage improves by `2.49x`, the public route improves by `1.33x` on the same worker used for the post-head public baseline, and the proof is bit-identical for the transformed factor plus public golden-stable.

## Validation

- `cargo fmt -p fsci-linalg --check`: clean.
- `ubs crates/fsci-linalg/src/lib.rs`: 0 critical findings; fmt/check/clippy/build sections clean.
- RCH `cargo test -p fsci-linalg --release --lib --locked thin_bidiag_parallel_right_replay_matches_serial_bits -- --nocapture --test-threads=1`: passed.
- RCH `cargo test -p fsci-linalg --release --lib --locked public_svd_lstsq_pinv_golden_payload -- --nocapture --test-threads=1`: passed.
- RCH `cargo check -p fsci-linalg --lib --locked`: clean.
- RCH `cargo clippy -p fsci-linalg --lib --locked --no-deps -- -D warnings`: clean.

## Next Profile Target

Re-profile after this keep. The remaining square SVD stage residual is now reduction and bidiagonal SVD:

```text
reduction_ms=75.967
bidiagonal_svd_ms=51.810
back_transform_v_ms=26.628
```

The post-keep broad `route_perf_probe` reprofile on `ovh-a` passed all 9 public route probes and still points at square `svd()` as the largest absolute SVD-family route (`PUBLIC_SQUARE_SVD_ROUTE_PERF routed_ms=257.567465` in the broad multi-probe run; a repeated isolated square probe on `vmi1227854` reported `routed_ms=231.031053`). The next no-gaps primitive should attack the Golub-Kahan reduction or bidiagonal SVD backend with a fundamentally different communication-avoiding/cache-blocked kernel, not another right-replay micro-lever.
