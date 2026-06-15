# psn7x row-major Givens replay rejection

Bead: `frankenscipy-psn7x`
Agent: `RubyWaterfall`
Base commit: `15b1859d`
Worker: `vmi1152480`
Date: 2026-06-15

## Profile-backed target

The prior stage profile put native symmetric-eigh cost at:

| stage | time |
| --- | ---: |
| Householder reduction | 903.217 ms |
| Tridiagonal QR/eigenvectors | 1089.798 ms |
| Back-transform | 296.175 ms |
| Sort/copy | 27.073 ms |

The selected one-lever probe targeted the tridiagonal QR/eigenvector stage.

## Candidate

Windowed/batched Givens replay was narrowed to one source lever:

- keep the scalar tridiagonal `diag`/`off` updates unchanged;
- collect each sweep's Givens rotations;
- replay those rotations against a row-major eigenvector buffer in the exact same per-row rotation order;
- convert back to `DMatrix` on return;
- leave public `eigh` routing unchanged.

Excluded families:

- no retry of `rotate_eigenvector_columns` slice/index spelling;
- no retry of per-step thread spawning for rank-2 updates.

## Proof

`proof_row_major_givens_bits_rch.txt`:

- RCH worker `vmi1152480`
- `cargo test -j 1 -p fsci-linalg --release --locked --lib tridiagonal_row_major_givens_replay_matches_immediate_bits -- --nocapture`
- passed
- focused proof: buffered row-major replay matched immediate column rotation by `f64::to_bits` on a deterministic 192x192 fixture.

`after_row_major_givens_native_timing_vmi1152480_rch.txt`:

- RCH worker `vmi1152480`
- `cargo test -j 1 -p fsci-linalg --release --locked --lib symmetric_eigh_native -- --include-ignored --nocapture`
- passed both native correctness and timing tests.

## Benchmark gate

Same-worker baseline (`baseline_native_vs_nalgebra_vmi1152480_rch.txt`):

| n | baseline native | baseline nalgebra | native/nalgebra |
| ---: | ---: | ---: | ---: |
| 400 | 87.2 ms | 65.9 ms | 0.76x |
| 800 | 576.1 ms | 467.3 ms | 0.81x |
| 1200 | 2114.8 ms | 2142.7 ms | 1.01x |

Same-worker candidate (`after_row_major_givens_native_timing_vmi1152480_rch.txt`):

| n | candidate native | candidate nalgebra | native/nalgebra |
| ---: | ---: | ---: | ---: |
| 400 | 1112.7 ms | 61.6 ms | 0.06x |
| 800 | 4914.8 ms | 476.0 ms | 0.10x |
| 1200 | 10493.2 ms | 1632.1 ms | 0.16x |

Same-worker native delta:

| n | speedup |
| ---: | ---: |
| 400 | 0.08x |
| 800 | 0.12x |
| 1200 | 0.20x |

Score: `Impact 0.0 * Confidence 4.0 / Effort 2.0 = 0.0`.

Verdict: REJECT. Source was restored to zero linalg diff. Keep the evidence only.

## Isomorphism proof summary

- Ordering preserved: yes for the proof candidate; eigenpair sorting still used the same `total_cmp` path.
- Tie-breaking unchanged: yes; no sorting/tie code changed.
- Floating-point: row-major replay matched immediate replay by bits in the focused helper test, but the full candidate still changed storage/replay locality and was rejected on performance.
- RNG: unchanged; deterministic test seeds were unchanged.
- Public `eigh`: unchanged; no public routing was touched.

## Next route

Do not repeat row-major buffered replay. The next `frankenscipy-psn7x` route must be a structurally different tridiagonal eigensolver primitive:

- divide-and-conquer tridiagonal eigensolver,
- MRRR-style representation work,
- or another algorithm that reduces/avoids the sequential eigenvector-update burden rather than replaying the same Givens rotations with a different layout.
