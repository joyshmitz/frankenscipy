# frankenscipy-kjzbw evidence

Agent: RubyWaterfall
Date: 2026-06-15
Target: `crates/fsci-linalg` public `eigh` after native symmetric-eigh inverse-iteration keep
Worker: RCH `ovh-a`

## Profile baseline

Temporary ignored stage probe for `symmetric_eigh_native` at n=1200:

- Householder reduction: 406.741 ms
- Tridiagonal eigen: 75.736 ms
- Eigenvector back-transform: 103.695 ms
- Sort/copy: 5.408 ms

Transcript: `baseline_stage_breakdown_rch.txt`

## Rejected probes

- Lower-triangle symmetric Householder matvec: rejected. Reduction stage regressed to 1015.218 ms. Transcript: `after_symmetric_matvec_stage_breakdown_rch.txt`.
- SIMD matvec spelling in trailing rank-2 update: rejected. Native timing moved from 31.9 / 196.7 / 615.9 ms to 36.5 / 223.2 / 671.9 ms at n=400/800/1200. Transcript: `after_simd_matvec_timing_rch.txt`.
- Back-transform worker cap 8 -> 16: rejected after confirmation. Best first run was small/noisy, confirm was 31.9 / 200.9 / 617.5 ms versus baseline 31.9 / 196.7 / 615.9 ms. Transcripts: `after_backtransform_16_workers_timing_rch.txt`, `after_backtransform_16_workers_timing_confirm_rch.txt`.

## Kept lever

Route public `eigh` through the safe-Rust native symmetric solver for matrices with `n >= 256`, with fallback to the old nalgebra path when the native candidate returns `None`.

Public route final exact-code probe on `ovh-a`:

| n | old nalgebra path | routed native path | speedup | max eigenvalue drift | values digest |
|---:|---:|---:|---:|---:|---:|
| 400 | 42.349386 ms | 36.491512 ms | 1.160527x | 5.24025267623073887e-13 | 0x0dbbde75b75c8612 |
| 800 | 317.278955 ms | 222.427040 ms | 1.426441x | 1.50635059981141239e-12 | 0xad8a7e5fa1980bfb |
| 1200 | 1036.363234 ms | 664.917996 ms | 1.558633x | 2.05346850634668954e-12 | 0x181b3486089d0e4a |

Transcript: `after_public_native_eigh_route_after_clippy_fix_rch.txt`

Score: Impact 4 x Confidence 3 / Effort 2 = 6.0. Keep threshold >= 2.0.

## Behavior and invariant proof

- Input validation, shape checks, finite checks, trace emission, and empty-matrix behavior stay before dispatch.
- `n < 256` remains on the old nalgebra public path; small golden digest is unchanged: `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`.
- Native solver proof `symmetric_eigh_native_matches_nalgebra_and_timing` passed. It checks ascending eigenvalues, residuals, orthonormal eigenvectors, and eigenvalue agreement with nalgebra for deterministic symmetric inputs.
- Public route proof checks old-path eigenvalues against routed values at n=400/800/1200 and captures deterministic value digests.
- No unsafe code added. No RNG added to production code. Test RNG is deterministic and isolated to ignored proof probes.

## Gates

- PASS: RCH `cargo test -j 1 -p fsci-linalg --lib public_eigh_native_route_perf_probe --release --locked -- --ignored --nocapture`.
- PASS: RCH `cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_matches_nalgebra_and_timing --release --locked -- --nocapture`.
- PASS: RCH `cargo test -j 1 -p fsci-linalg --lib eigh_index_sort_matches_materialized_pair_sort_bits --release --locked -- --nocapture`.
- PASS: RCH `cargo check -j 1 -p fsci-linalg --lib --locked`.
- PASS with zero critical findings: `ubs crates/fsci-linalg/src/lib.rs`.
- BLOCKED by pre-existing lint debt: `cargo clippy -j 1 -p fsci-linalg --lib --locked -- -D warnings` fails first in `fsci-fft/src/helpers.rs:58`.
- BLOCKED by pre-existing lint debt after `--no-deps`: `crates/fsci-linalg/src/lib.rs:3709`, `3720`, and `4170` `needless_range_loop`.
- BLOCKED by pre-existing formatting drift outside this lever: `cargo fmt --check -p fsci-linalg` reports older hunks around `orthogonal_procrustes`, `matmul_toeplitz`, `pinvh`, and Procrustes tests.

## Next route

After this public-route keep, reprofile public `eigh` and native symmetric-eigh separately. The next deeper primitive should target the Householder reduction stage or a fundamentally different tridiagonal/eigenvector primitive, not lower-triangle indexing spelling, SIMD spelling of the same rank-2 loop, worker-count retuning, rotate-slice rewrites, or per-step thread spawning.
