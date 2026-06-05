# SpMM Numeric-Only Parallel Chunk Build Rejection

Bead: `frankenscipy-8l8r1.36`

## Profile Target

Fresh RCH sparse reprofile after the byte-state rejection ranked
`sparse_spmm/2000x2000_d1/2000` first:

- Worker: `ts2`
- Command: `rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot`
- Median: `11.701 ms`
- Interval: `[11.638 ms, 11.791 ms]`

## Lever Tested

Remove the parallel path's duplicate exact symbolic Gustavson row-count pass and
let the numeric row chunk produce counts, columns, and values in one pass, using
a conservative work-derived capacity hint.

## Isomorphism Proof

The candidate kept the same row ranges, row order, A traversal, B row encounter
order, reverse first-seen column emission, floating-point accumulation order,
explicit zero elision, sorted-index metadata, error behavior, and RNG absence.
The only intended change was the allocation capacity source and removal of the
pre-count replay in the parallel exact worker.

Strict golden payload stayed byte-identical:

- Before SHA256: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- After SHA256: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- Strict payload compare: `strict_cmp=0`

## Benchmark Result

Focused before-edit baseline:

- Worker: `ts1`
- Command: `rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 5 --sample-size 30 --noplot`
- Median: `9.5794 ms`
- Interval: `[9.4721 ms, 9.6694 ms]`

After-edit results landed on the same worker as the fresh reprofile target:

- Worker: `ts2`
- First after median: `11.759 ms`
- First after interval: `[11.622 ms, 11.899 ms]`
- Confirmation median: `12.047 ms`
- Confirmation interval: `[11.934 ms, 12.154 ms]`

Compared to the same-worker `ts2` profile target, the candidate moved
`11.701 ms` to `11.759 ms` and then `12.047 ms`. This is a regression, not a
campaign-quality win.

## Score And Verdict

Score: `0.0 = impact 0 * confidence 3 / effort 1`

Verdict: REJECTED / NO-SHIP. Source was restored. Do not continue row-count,
capacity-hint, marker, replay, or scheduling-only SpMM levers from here.

Next primitive: replace the row-local sparse accumulator/traversal model with a
fundamentally different safe-Rust SpGEMM primitive, such as a row-local
hash/sort accumulator or sorted contribution stream with explicit proof of the
same floating-point operation order and output ordering.
