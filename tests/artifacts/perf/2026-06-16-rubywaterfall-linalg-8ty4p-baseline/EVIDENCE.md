# frankenscipy-8ty4p baseline and materialization rejection

Bead: `frankenscipy-8ty4p`
Agent: RubyWaterfall
Date: 2026-06-16

## Target

Current-head native symmetric `eigh` remains profile-backed by the public route and stage split. The measured 1200x1200 stage split on RCH worker `vmi1293453` is:

- reduction: `399.721635 ms`
- tridiagonal eigen: `87.435120 ms`
- backtransform: `158.591750 ms`
- sort: `4.704990 ms`

The reduction stage remains the dominant target for the next algorithmic pass.

## Baseline

RCH worker: `vmi1293453`

Criterion `eigh_dense` baseline:

- 256x256: `[12.558 ms, 12.755 ms, 12.906 ms]`
- 512x512: `[123.81 ms, 146.37 ms, 175.23 ms]`

Public route baseline:

- 400x400: routed `52.813164 ms`, nalgebra `44.817756 ms`, digest `0x4b8334c92ce624eb`
- 800x800: routed `216.875986 ms`, nalgebra `408.189286 ms`, digest `0xad8a7e5fa1980bfb`
- 1200x1200: routed `693.233721 ms`, nalgebra `1316.413195 ms`, digest `0x181b3486089d0e4a`

## Rejected Lever

Temporary source lever: changed only the public `eigh` native eigenvector materialization loop from column-major output writes to row-contiguous `Vec<Vec<f64>>` writes. The mapping was still exactly `eigenvectors[row][col] = native_vectors[(row, col)]`; no arithmetic, ordering, tie-breaking, RNG, fallback, or threshold behavior changed.

After RCH public-route benchmark on `vmi1293453`:

- 400x400: `52.813164 -> 58.950289 ms`, digest unchanged `0x4b8334c92ce624eb`
- 800x800: `216.875986 -> 301.105720 ms`, digest unchanged `0xad8a7e5fa1980bfb`
- 1200x1200: `693.233721 -> 969.322651 ms`, digest unchanged `0x181b3486089d0e4a`

Score: `Impact 0.0 * Confidence 4.0 / Effort 1.0 = 0.0`

Verdict: REJECT. Source restored; `git diff -- crates/fsci-linalg/src/lib.rs` is empty after restore.

## Isomorphism

- Ordering preserved: yes. The candidate copied the same sorted native-vector matrix coordinates into the same public row/column coordinates.
- Tie-breaking preserved: yes. The eigenvalue/eigenvector order was produced before materialization and was untouched.
- Floating-point preserved: yes. The candidate performed no arithmetic, only assignment; public eigenvalue digests remained unchanged.
- RNG preserved: yes. No RNG code or seeds changed.
- Golden outputs: public route digests remained unchanged for 400/800/1200.

## Next Route

Do not retry output materialization loop-order tweaks. The next pass must attack the measured reduction wall with an algorithmically different primitive: a production-faithful lower-storage stage profile, then a two-stage dense-to-band or band-to-tridiagonal/bulge-chasing slice with scalar replay proof.
