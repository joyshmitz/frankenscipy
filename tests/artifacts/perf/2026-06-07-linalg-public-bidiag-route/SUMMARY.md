# Public Bidiagonal SVD Acceptance Path Rejection

Bead: `frankenscipy-8l8r1.43`

## Target

Fresh RCH profile on `ts1` after the GEMM tile rejection chain kept the SVD-backed
public route at the top of `fsci-linalg`:

- `pinv/512x256`: `[98.449 ms 99.197 ms 99.963 ms]`
- `lstsq/512x256`: `[83.221 ms 85.564 ms 87.914 ms]`
- `matmul/1024x1024`: `[80.493 ms 84.794 ms 89.304 ms]`

The current source already routes public full-rank tall `svd`, `svdvals`,
`lstsq`, and `pinv` through the guarded private bidiagonal SVD path when the
acceptance certificate passes.

## Baseline And Profile

RCH `ts1` public-route probe before the trial:

- `reference_lstsq_ms=101.543521`
- `routed_lstsq_ms=74.970714`
- `reference_pinv_ms=105.104419`
- `routed_pinv_ms=77.441517`
- `lstsq_speedup=1.354442`
- `pinv_speedup=1.357210`

Stage probes on `ts1` showed the public route is dominated by real SVD work, not
only by the certificate:

- `BIDIAG_LARGE_REDUCTION_PERF`: `204.909907 ms` for `1024x512`
- `THIN_BIDIAG_FACTOR_REPLAY`: `250.248466 ms` for `1024x512`
- `BIDIAG_SYMMETRIC_EIGEN_ROUTE`: `108.130714 ms` for `1024x512`
- `PUBLIC_BIDIAG_SVD_ROUTE`: `74.970714 ms` routed `lstsq`, `77.441517 ms`
  routed `pinv` for `512x256`

## Lever Tried

Replace the acceptance certificate reconstruction
`thin.u * thin.sigma_matrix() * thin.v_t` with a streaming diagonal-aware
reconstruction-error pass. This avoids materializing the diagonal matrix and the
dense multiply by a mostly-zero matrix during route acceptance.

This was a certification-only lever: singular values, `U`, `Vt`, rank threshold,
residual computation, certificates, RNG behavior, ordering, and public return
values were intended to remain unchanged.

## Proof

- RCH public golden payload passed.
- Public SVD/lstsq/pinv golden SHA-256 stayed:
  `1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`
- RCH route guards passed:
  - `public_bidiag_svd_route_matches_safe_svd_reference`
  - `public_bidiag_svd_route_rejects_clustered_spectrum`
  - temporary dense-vs-streaming acceptance guard
- `cargo fmt -p fsci-linalg --check` passed.
- `git diff --check` passed.

## Rebench

The after evidence did not clear the keep gate.

Cross-worker public-route probes were mixed and therefore non-decisive:

- `vmi1156319`: `routed_lstsq_ms=126.584997`, `routed_pinv_ms=142.255843`
  (regressed badly)
- `vmi1293453`: `routed_lstsq_ms=73.112053`, `routed_pinv_ms=75.203193`
  (positive but not comparable to the `ts1` baseline)
- `vmi1264463`: Criterion retry after source restoration measured
  `lstsq/512x256` `[160.53 ms 184.90 ms 217.92 ms]` and `pinv/512x256`
  `[136.52 ms 144.87 ms 156.91 ms]`, reinforcing the no-keep decision.

Attempts to force `ts1` with `RCH_WORKER=ts1` and `RCH_WORKERS=ts1` were ignored
by the shared daemon. An isolated `rchd` daemon with a one-worker `ts1` config
could not be started without replacing/interfering with the managed user service,
so it was not used.

A later `ts1` Criterion control capture after source restoration measured:

- `lstsq/512x256`: `[96.219 ms 98.053 ms 99.988 ms]`
- `pinv/512x256`: `[97.281 ms 98.101 ms 99.033 ms]`

That does not provide a same-worker Score >= 2.0 win against the fresh profile
target, and the mixed cross-worker route probes lower confidence.

## Decision

Rejected. Source was restored; `git diff -- crates/fsci-linalg/src/lib.rs` is
empty. No production code from the trial remains.

Score: `0.5 = impact 1 * confidence 1 / effort 2`.

Next primitive: stop acceptance-certificate micro-levers. Reprofile and attack
the deeper bidiagonal factor/replay cost with a structurally different primitive:
two-stage communication-avoiding bidiagonalization or a cache-oblivious
reflector replay tree with explicit reconstruction and public golden proof.
