# `frankenscipy-8l8r1.92` baseline contract

## Target

`eig_banded(..., lower=true, eigvals_only=true)` for general lower symmetric
band storage.

The active route must replace the dense-expanded scalar Householder reduction
with a true band-native values-only primitive. Previous attempts rejected two
families:

- sparse `BTreeMap` lower-band bulge chase: proof-clean but `0.327781x` at
  `512x512`,
- packed adjacent-Givens full-similarity replay: failed focused proof because
  fill escaped the retained band envelope.

## Baseline Evidence

The baseline remains the same target measured by the `.90` RCH release probe:

- Artifact:
  `tests/artifacts/perf/2026-06-11-linalg-8l8r1-90/after_band_to_tridiagonal_perf_probe_rch.txt`
- Worker: `ovh-a`
- `256x256`, bandwidth `32`: dense-expanded reference `66.890671 ms`
- `512x512`, bandwidth `32`: dense-expanded reference `512.075240 ms`
- Artifact sha256:
  `7477b565814151216220cd4dbd983c4a25454943b53c53d6133b5a43cb04ca3e`

The compact-WY full-to-band stage side baseline from `.91` remains:

- Artifact:
  `tests/artifacts/perf/2026-06-12-linalg-8l8r1-91/baseline_full_to_band_replay_rch.txt`
- Worker: `vmi1152480`
- `512x512`, bandwidth `32`: scalar `132.171825 ms`, compact-WY
  `82.742961 ms`, public `eigh` side-probe `127.528251 ms`
- Artifact sha256:
  `30b5218f361b81a238488af156fef70bf7d4bbcbb1de9b7e795bbfb1ea0d2497`

Public `eigh` ordering/golden baseline:

- Digest: `eigh_index_sort_public_golden_digest=0x287a5d3679a8bc6a`
- Artifact:
  `tests/artifacts/perf/2026-06-11-linalg-8l8r1-90/proof_public_eigh_golden_after_band_tridiagonal_rch.txt`
- Artifact sha256:
  `ad0454c2649903ed9bdeb26308dc529da5c64b4ac9a18cde169d2699c16f2aa1`

## Candidate Contract

One lever only: a band-native values-only eigenvalue route that stores only the
input lower band plus bounded vector/tridiagonal workspaces. It must not use a
sparse map, direct dense expansion, scalar full-similarity replay, widening
packed windows, external BLAS/LAPACK/MKL/XLA, or `unsafe`.

The route may leave the eigenvector-producing `eig_banded` path on the existing
dense transform because `.92` only claims the `eigvals_only=true` values path.

## Proof Requirements

- Preserve `eig_banded` shape, finite-check, lower-storage, diagonal, and
  tridiagonal special-case behavior.
- Eigenvalues agree with the existing dense symmetric reference within the
  current tolerance budget across deterministic banded fixtures.
- Public `eigh` golden digest remains `0x287a5d3679a8bc6a`.
- Ordering remains ascending under the existing comparator policy.
- No RNG, `unsafe`, or external-kernel path is introduced.

## Keep Gate

Target: at least `1.25x` over the dense-expanded reference at `512x512`,
Score `>= 2.0`, and crate-scoped RCH proof/check/clippy/fmt evidence.
