# SpMM Byte-State Accumulator Rejection

- Bead: `frankenscipy-8l8r1.35`
- Target: `sparse_spmm/2000x2000_d1/2000`
- Candidate: replace the `Vec<bool>` dense seen markers in the symbolic-count and numeric Gustavson SpMM row kernels with byte-state markers.

## Baseline

- Command: `rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot`
- Worker: `ts1`
- Median: `9.8532 ms`
- Interval: `[9.7244 ms, 9.9780 ms]`
- Artifact: `baseline_rch.txt`

## Isomorphism Proof

- Row partitioning, row traversal, A traversal, B encounter order, reverse first-seen emission order, floating-point accumulation order, explicit zero elision, sorted metadata, and RNG absence were intended unchanged.
- Strict golden before SHA: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- Strict golden after SHA: `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- `cmp -s golden_before_payload.strict.txt golden_after_payload.strict.txt` returned `0`.

## After

- Command: `rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked -- sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 3 --sample-size 20 --noplot`
- Worker: `ts1`
- Median: `9.6922 ms`
- Interval: `[9.5527 ms, 9.8625 ms]`
- Artifact: `after_rch.txt`

## Verdict

- Delta: `1.02x` by median, with overlapping intervals.
- Score: `0.8 = impact 1 * confidence 0.8 / effort 1`.
- Result: rejected below the `Score >= 2.0` keep threshold.
- Source: restored to the pre-candidate `Vec<bool>` marker representation before closeout.
- Next primitive: avoid additional marker-family micro-levers; attack a larger sparse accumulator/traversal replacement such as a true CSC/column-panel or semiring-symbolic SpGEMM path.
