# frankenscipy-psn7x: tridiagonal inverse-iteration eigenvectors

Agent: RubyWaterfall
Date: 2026-06-15
Crate: `fsci-linalg`
Lever: replace the staged native symmetric-eigh tridiagonal eigenvector stage with deterministic inverse iteration over eigenvalues from `eigh_tridiagonal`, falling back to the existing full tridiagonal QR path if the residual gate fails.

## Baseline

Command transcript: `baseline_current_tip_native_timing_rch.txt`
Worker: `ovh-a`

| n | baseline native | baseline nalgebra |
|---|---:|---:|
| 400 | 47.2 ms | 43.7 ms |
| 800 | 335.0 ms | 316.3 ms |
| 1200 | 1095.3 ms | 1045.6 ms |

## Final Timing

Command transcript: `after_inverse_iteration_timing_final_rch.txt`
Worker: `ovh-a`

| n | final native | final nalgebra | native speedup vs baseline |
|---|---:|---:|---:|
| 400 | 31.9 ms | 41.6 ms | 1.48x |
| 800 | 196.7 ms | 318.0 ms | 1.70x |
| 1200 | 615.9 ms | 1027.2 ms | 1.78x |

Score: `Impact 3.5 * Confidence 4.0 / Effort 2.0 = 7.0` (KEEP).

## Behavior Proof

- Tridiagonal inverse-iteration residual/orthogonality proof: `proof_inverse_iteration_tridiagonal_final_rch.txt` passed.
- Native symmetric-eigh vs nalgebra proof: `proof_native_inverse_iteration_final_rch.txt` passed.
- Public `eigh` bitwise materialized-pair golden guard: `proof_public_golden_inverse_iteration_final_rch.txt` passed with `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`.
- Ordering and tie behavior: public `eigh` route remains on the same materialized-pair sort golden; staged native eigenvalues come from `eigh_tridiagonal` ascending output and are sorted by the existing native sorting pass.
- Floating-point scope: changed only the staged native tridiagonal eigenvector construction; residual gate falls back to the previous QR eigenvector path when the inverse-iteration vectors are not clean.
- RNG: no randomness; inverse-iteration right-hand sides are deterministic integer-pattern vectors and signs are canonicalized.

## Gates

- `check_fsci_linalg_lib_final_rch.txt`: `cargo check -j 1 -p fsci-linalg --lib --locked` passed on `ovh-a`; existing dependency warning in `fsci-fft/src/helpers.rs:58` remains.
- `clippy_fsci_linalg_lib_nodeps_after_workspace_rch.txt`: no new inverse-iteration lint remains; clippy is still blocked by pre-existing `needless_range_loop` findings at `crates/fsci-linalg/src/lib.rs:3709`, `3720`, and `4170`.
- `fmt_fsci_linalg_check_final.txt`: changed hunk is formatted; `cargo fmt -p fsci-linalg -- --check` still reports pre-existing formatting drift at `orthogonal_procrustes`, `matmul_toeplitz`, and `pinvh` tests.
- `ubs_fsci_linalg_lib_final.txt`: changed-file UBS scan completed with `Critical issues: 0`.

## Gap Guard Follow-Up

Follow-up commit: add `TRIDIAGONAL_INVERSE_MIN_GAP_REL` so inverse iteration is accepted only when adjacent tridiagonal eigenvalues are separated by more than `1e-6 * scale`. Clustered spectra now fail closed to the existing QR eigenvector path instead of accepting residual-clean but potentially non-orthogonal inverse-iteration vectors.

Additional artifacts:

- `proof_inverse_iteration_gap_guard_rch.txt`: RCH `cargo test -j 1 -p fsci-linalg --lib tridiagonal_inverse_iteration --locked -- --nocapture` passed both the residual/orthogonality test and the clustered-eigenvalue fallback test.
- `fmt_fsci_linalg_gap_guard.txt`: `cargo fmt -p fsci-linalg -- --check` passed after the guard.
- `after_inverse_iteration_gap_guard_timing_rch.txt`: RCH release smoke timing on `ovh-a` passed native correctness and showed native still faster than nalgebra at n=`400/800/1200` (`1.53x/1.74x/1.58x`). This run was noisier than the parent keep run, with nalgebra also materially slower, so it is retained as a guard smoke artifact rather than a new optimization score.
- `clippy_fsci_linalg_gap_guard_rch.txt`: RCH `cargo clippy -j 1 -p fsci-linalg --lib --no-deps --locked -- -D warnings` remains blocked by the same pre-existing `needless_range_loop` findings at `lib.rs:3709`, `3720`, and `4170`; the guard added no new clippy finding.

## Transcript SHA-256

- `baseline_current_tip_native_timing_rch.txt`: `8a59b97996202f4417679be41bf7f05563f3828b9cf4cdafa1c08c376cf161eb`
- `after_inverse_iteration_timing_final_rch.txt`: `d73d7653bcadf4b684e97324386124f82329694b64e107444c40fd759e6b954a`
- `proof_inverse_iteration_tridiagonal_final_rch.txt`: `b7ef8ffa24e8f8590dc6979b4606688a95622d2dfb7e27a94acda8ea83ec130b`
- `proof_native_inverse_iteration_final_rch.txt`: `3751d1d349e321157e68a61efd98a1631363a4d81dc69df184206ecf18ce0994`
- `proof_public_golden_inverse_iteration_final_rch.txt`: `b36fd4dab3fec59c265e94efbd34597506b973b75699015a7469437e870ad74d`
- `check_fsci_linalg_lib_final_rch.txt`: `6031e4b6487a6ed98ec8633493b3bc81715b5cfd1e081667e4d37bcd703adf82`
- `fmt_fsci_linalg_check_final.txt`: `86a572320a5c5838f3a27d204878ca4978645ac76b933f8a6af9b787de0ca15f`
- `ubs_fsci_linalg_lib_final.txt`: `5554bc1864f06222ac5185e43867b76abf9e6898dc245cf63f22ac88b9c77c9b`
