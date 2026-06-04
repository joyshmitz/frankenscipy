# Stats ECDF sorted-query merge rejection

Bead: `frankenscipy-kdqor`

Target:
- `ordering_and_bins/ecdf/4096x512`
- Profile source: `tests/artifacts/perf/2026-06-04-stats-reprofile-after-qmc-rejections/reprofile_stats_rch.txt`
- Reprofile row: `[86.524 us 87.897 us 89.175 us]`

Proposed lever:
- Keep current data sort by `f64::total_cmp`.
- Sort finite query values with original indices.
- Scan the sorted data once using the same `v <= x` relation.
- Write counts back to original query order.
- Fall back to the existing `partition_point` implementation if data or query inputs contain NaN.

Behavior proof:
- Output order preserved by original query index writeback.
- Tie semantics preserved by the same `v <= x` count relation.
- No RNG, no ordering side effects, and no floating-point arithmetic changes other than the same `count as f64 / n` ratio.
- Golden before SHA256: `ab9f2defb6e38cac46966e7bbdacb1ed2f798f457a625cd0429ee910d42c9846`
- Golden after SHA256: `ab9f2defb6e38cac46966e7bbdacb1ed2f798f457a625cd0429ee910d42c9846`
- `cmp golden_before.txt golden_after.txt`: passed

Baseline:
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- ordering_and_bins/ecdf/4096x512 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`
- Worker: `ts2`
- Result: `[133.41 us 134.05 us 134.76 us]`

Same-process A/B:
- Command: `RCH_FORCE_REMOTE=1 rch exec -- cargo run -p fsci-stats --release --bin perf_stats --locked -- ecdf-ab 100`
- Worker: `vmi1156319`
- Result: `mode=ecdf-ab repeats=100 bit_identical=true old_us=134.382000 new_us=142.455000 speedup=0.943329`

Decision:
- Rejected. The lever was bit-identical but slower on the RCH same-process A/B, so it failed the Score >= 2.0 keep gate.
- Production code and helper-harness edits were reverted; no ECDF implementation change is kept.
- Next direction must pivot away from query-order micro-tuning toward a deeper stats primitive.
