# Sparse SpMM Post-Rejection Trial

Bead: `frankenscipy-9msnm`
Crate: `fsci-sparse`
Target: `sparse_spmm/2000x2000_d1/2000`

## Profile Context

Fresh RCH sparse reprofile on 2026-06-07 ranked `sparse_spmm/2000x2000_d1/2000`
first at `16.357 ms` median `[15.227, 17.566]` on `ts1`, ahead of
`sparse_spilu/1024_bw32` at `4.1265 ms` and
`sparse_arithmetic/10000x10000_d0_add` at `3.3445 ms`.

Focused baseline for this bead:

- Worker: `vmi1227854`
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot`
- Time: `6.6583 ms` median, CI `[6.4867, 6.8058]`

## Golden Proof

Before strict payload:

- Source: `golden_before_rch_raw_2.txt`
- Strict payload: `golden_before_payload.strict.txt`
- SHA-256: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- `cmp` against `tests/artifacts/perf/2026-06-05-sparse-spmm-work-balanced-rows/golden_after_payload.strict.txt`: `0`

## Rejected Levers

### Serial Parallel-Gate Trial

Changed only `spmm_chunk_count` to keep the 800k-work benchmark case serial by
raising the parallel gate from `300_000` to `1_000_000`. This preserves row order,
A/B encounter order, floating-point accumulation order, reverse first-seen output
order, explicit zero elision, metadata semantics, and RNG absence because it uses
the existing serial Gustavson row kernel.

Result:

- Worker: `vmi1149989`
- Time: `18.355 ms` median, CI `[16.865, 20.082]`
- Verdict: rejected. This is far outside the current parallel profile envelope
  and fails Score >= 2.0.
- Source restored; `git diff -- crates/fsci-sparse/src/linalg.rs` was empty after restore.

### Lazy Sorted-Certificate Trial

Changed only the output metadata check so `spmm_row_chunk` stops comparing
subsequent output columns after `sorted_indices` has already become false. The
column/value output path and arithmetic path stay unchanged.

Result:

- Worker: `vmi1167313`
- Time: `28.300 ms` median, CI `[25.015, 32.364]`
- Verdict: rejected. Cross-worker evidence is not used as keep proof, and the
  candidate is not credible as a Score >= 2.0 win.
- Source restored; `git diff -- crates/fsci-sparse/src/linalg.rs` was empty after restore.

## Decision

No source change kept for `frankenscipy-9msnm`. The SpMM family already has
multiple rejected marker, epoch, count, capacity, replay, row-plan, panel,
sorted-certificate, and thread-gate attempts. Next optimization should route to
the next profile-backed primitive or a materially different SpGEMM worker model
rather than another local accumulator/bookkeeping tweak.
