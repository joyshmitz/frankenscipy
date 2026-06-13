# frankenscipy-8l8r1.96 flat LU solve-only keep

Scope: replace the large strict/general `solve` route with the in-house flat contiguous LU factorization and register-blocked trailing update. The 256x256 inverse and determinant public dispatches were measured and rejected; final runtime dispatch keeps inverse gated at 1024 and leaves determinant on the existing nalgebra path.

## Benchmark evidence

Worker: `vmi1149989`, crate-scoped RCH commands.

| Surface | Baseline mean | Final / measured mean | Decision |
| --- | ---: | ---: | --- |
| `baseline_solve/1000x1000` | `112.47 ms` | `96.385 ms` | keep, `1.17x` faster |
| `inv/256x256` candidate route | `8.6720 ms` | `13.277 ms` | reject 256 flat-LU inverse dispatch |
| `inv/256x256` final gate 1024 | `8.6720 ms` | `10.037 ms` | no 256 flat-LU dispatch; retained as guard evidence |
| `det/256x256` candidate route | `1.0614 ms` | `1.2785 ms` | reject public flat-LU determinant dispatch |

Score: Impact `3` x Confidence `0.82` / Effort `1` = `2.46`; accepted because the kept public surface is the profiled solve route and rejected subroutes are not shipped.

## Behavior proof

- `proof_flat_lu_parity_rch.txt`: permutation parity proof passed.
- `proof_blocked_reference_rch.txt`: `lu_solve_blocked_matches_reference`, `cholesky_solve_blocked_matches_reference`, and `inv_blocked_matches_reference` passed.
- `proof_flat_lu_golden_digest_rch.txt`: ignored release golden passed with `flat_lu_golden_digest=0x2fc8ed294ef0427c`.

## Gates

- `rustfmt_linalg_lib_check.txt`: passed.
- `ubs_linalg_lib.txt`: exit 0; existing warning inventory only.
- `check_fsci_linalg_all_targets_rch.txt`: `cargo check -p fsci-linalg --all-targets` passed on `vmi1149989`.
- `clippy_fsci_linalg_all_targets_no_deps_rch.txt`: `cargo clippy -p fsci-linalg --all-targets --no-deps -- -D warnings` passed.
- `clippy_fsci_linalg_all_targets_rch.txt`: dependency-inclusive clippy failed in `fsci-fft` on pre-existing `manual_is_multiple_of`; not part of this linalg lever.
