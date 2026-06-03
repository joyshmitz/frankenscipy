# 4x16 no-pack matmul tile conclusion

Bead: `frankenscipy-8l8r1.18`

## Target

Try one lever on the profile-backed dense GEMM hotspot: widen the existing
safe-Rust no-pack register micro-kernel from `MR x NR = 4 x 8` to `4 x 16`.

## Baseline

RCH Criterion baseline on `vmi1149989`, current 4x8 tile:

- `matmul/256x256`: `[3.8532 ms, 4.0600 ms, 4.2857 ms]`
- `matmul/512x512`: `[31.857 ms, 32.956 ms, 34.376 ms]`
- `matmul/768x768`: `[111.25 ms, 123.64 ms, 135.08 ms]`
- `matmul/1024x1024`: `[302.71 ms, 321.49 ms, 340.21 ms]`

## Candidate

RCH Criterion candidate on `vmi1153651`, 4x16 tile:

- `matmul/256x256`: `[13.337 ms, 14.043 ms, 14.671 ms]`
- `matmul/512x512`: `[171.15 ms, 193.51 ms, 218.34 ms]`
- `matmul/768x768`: `[983.07 ms, 1.0277 s, 1.0821 s]`
- `matmul/1024x1024`: `[2.9361 s, 3.0526 s, 3.1647 s]`

The candidate regressed every benchmark row, so the source was restored to the
committed 4x8 kernel.

## Isomorphism Proof

- Ordering preserved: yes. Each candidate output cell retained monotonic `k`
  accumulation and unchanged row/column traversal.
- Tie-breaking unchanged: N/A; matmul has no tie surface.
- Floating-point behavior: golden before/after normalized SHA256 stayed
  `02572d2d21db57707cada3378710a68ec5223e02131d20125b48d8d4e7a`.
- RNG seeds: N/A.
- Restored source check: `source_restored_matmul_hunk_check.txt` is empty,
  proving no remaining diff mentions `NR: usize`, `b8`, or `b15`.

## Validation

- Candidate golden RCH test: passed.
- Restored-source golden RCH test: passed.
- `cargo fmt -p fsci-linalg --check`: passed.

## Score

`0.0`: impact is negative. The lever fails the Score >= 2.0 keep gate.

Verdict: rejected and source restored.
