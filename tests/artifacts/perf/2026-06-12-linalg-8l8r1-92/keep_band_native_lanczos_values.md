# Keep band-native Lanczos eig_banded values route

Bead: `frankenscipy-8l8r1.92`

## Verdict

Kept one values-only `eig_banded(..., lower=true, eigvals_only=true)` lever:
a deterministic full-reorthogonalized Lanczos tridiagonalization over the lower
symmetric band storage, followed by the existing tridiagonal eigenvalue solver.

This does not change the eigenvector-producing `eig_banded` path. It avoids
the rejected dense expansion, sparse maps, widened packed full-similarity replay,
external BLAS/LAPACK/MKL/XLA, and `unsafe`.

## Baseline

Baseline dense-expanded values path from `.90`:

- Artifact: `tests/artifacts/perf/2026-06-11-linalg-8l8r1-90/after_band_to_tridiagonal_perf_probe_rch.txt`
- Worker: `ovh-a`
- `256x256`, bandwidth 32: `66.890671 ms`
- `512x512`, bandwidth 32: `512.075240 ms`

## After

Primary `.92` release perf probe:

- Artifact: `after_eig_banded_lanczos_perf_probe_rch.txt`
- Worker: `ovh-a`
- `256x256`, bandwidth 32: `23.811788 ms` (`2.809985x`)
- `512x512`, bandwidth 32: `137.553186 ms` (`3.722014x`)
- Max abs eigenvalue drift at `512x512`: `1.30967237055301666e-10`

Post-ordering cleanup probe after switching final sort to `f64::total_cmp`:

- Artifact: `after_eig_banded_lanczos_total_cmp_perf_probe_rch.txt`
- Worker: `hz1`
- `256x256`, bandwidth 32: `30.984280 ms`
- `512x512`, bandwidth 32: `187.365615 ms`
- Max abs eigenvalue drift at `512x512`: `1.30967237055301666e-10`

The primary same-worker `ovh-a` timing is the keep comparison. The `hz1`
post-cleanup probe confirms the retained source still clears the target by a
wide margin, but is not used as the same-worker timing claim.

## Behavior Proof

- Ordering preserved: yes. The values route sorts final eigenvalues with
  `f64::total_cmp`, matching the project policy for ascending eigenvalues.
- Tie-breaking: values-only output has no eigenvector tie/sign policy; no public
  eigenvector route is changed.
- Floating point: new route computes the same mathematical eigenvalues through a
  deterministic Lanczos projection and existing tridiagonal solver. It is
  tolerance-equivalent, not bit-identical to the dense-expanded reference.
- RNG: unchanged / none. The start vector and restart basis are deterministic.
- Golden public `eigh`: unchanged digest
  `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`.

Proof artifacts:

- `proof_eig_banded_lanczos_rch.txt`: focused values proof passed.
- `proof_eig_banded_lanczos_total_cmp_rch.txt`: focused proof after
  `total_cmp` cleanup passed.
- `proof_public_eigh_golden_rch.txt` and
  `proof_public_eigh_golden_total_cmp_rch.txt`: public `eigh` golden digest
  unchanged.

## Score

Score using the same-worker `ovh-a` keep comparison:

```text
Impact 4.5 * Confidence 4.0 / Effort 2.5 = 7.2
```

This clears the required `Score >= 2.0` gate and the `>=1.25x` release target at
`512x512`.

## Validation

Passed:

- `RCH_REQUIRE_REMOTE=1 ... rch exec -- cargo test -j 1 -p fsci-linalg --lib --locked -- eig_banded_lanczos --nocapture`
- `RCH_REQUIRE_REMOTE=1 ... rch exec -- cargo check -j 1 -p fsci-linalg --all-targets --locked`
- `rustfmt --edition 2024 --check crates/fsci-linalg/src/lib.rs`
- `git diff --check -- crates/fsci-linalg/src/lib.rs ...`
- `ubs crates/fsci-linalg/src/lib.rs` exited 0 with no critical findings.

Clippy:

- Existing artifact `clippy_fsci_linalg_all_targets_no_deps_after_fix_rch.txt`
  records the crate-scoped RCH command passing.
- Two fresh retries on 2026-06-13 were blocked by RCH selecting `hz2`, whose
  nightly toolchain lacks `cargo-clippy`. UBS also ran clippy locally as part of
  the single-file scan and reported clean formatting/clippy for the touched file.

## Next Route

Re-profile after this commit. The remaining dense-linalg route is no longer this
general `eig_banded` values expansion path; continue with the next highest
profile-backed `[perf]` bead rather than repeating sparse maps, adjacent
full-similarity replay, or widened packed windows.
