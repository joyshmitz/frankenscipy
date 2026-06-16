# frankenscipy-psn7x.10 Evidence - Rejected Banded Inverse Iteration

## Target

- Bead: `frankenscipy-psn7x.10`
- Target crate: `fsci-linalg`
- Candidate: band-native shifted inverse iteration for `eig_banded` eigenvectors.

## Candidate

The temporary lever reused the existing band-native Lanczos eigenvalue path,
then recovered each eigenvector by solving shifted symmetric band systems with
the existing packed band LU helper. The route guarded residuals and
orthogonality, and fell back to the current dense/native eigenvector path on
clustered spectra or failed solves.

No production source is retained.

## Proof

Focused vector proof passed before the local-only worker override:

| shape | bandwidth | residual | orthogonality |
| --- | ---: | ---: | ---: |
| 16x16 | 3 | 6.03961325396085158e-14 | 4.44089209850062616e-16 |
| 33x33 | 5 | 1.77635683940025046e-13 | 4.44089209850062616e-16 |
| 96x96 | 8 | 8.52651282912120223e-13 | 1.11022302462515654e-15 |

Ordering/tie behavior: eigenvalues remained sorted through the existing
ascending `total_cmp` route. RNG behavior: no public RNG was introduced;
deterministic right-hand-side seeds depended only on row and column indices.

## Local Hyperfine Rebench

Temporary override: `ts1`/RCH worker path offline, so rebench used local
`cargo` plus `hyperfine`.

Command:

```text
hyperfine --warmup 1 --runs 3 --show-output 'cargo test -j 1 -p fsci-linalg --lib eig_banded_eigenvectors_perf_probe --release --locked -- --ignored --nocapture'
```

Baseline worktree: `/data/projects/.scratch/frankenscipy-rubywaterfall-baseline-20260616-202203`
at `6864db4e`.

| shape | baseline ms | candidate ms | speedup |
| --- | ---: | ---: | ---: |
| 128x128 bw32 | 3.763551 / 3.780101 / 4.220096 | 1592.217136 / 129.008548 / 133.909723 | rejected |
| 256x256 bw32 | 14.686274 / 12.739115 / 13.781580 | 1474.263581 / 704.540817 / 661.067837 | rejected |

The candidate preserved the residual guard but was orders of magnitude slower
because it performed one shifted band solve per eigenvector.

## Decision

Score: `Impact 0.0 * Confidence 4.0 / Effort 2.0 = 0.0`.

Rejected/no-ship. Do not retry shifted inverse iteration over packed band
solves for public banded eigenvectors. Continue with an actual implicit
symmetric-band bulge chase with accumulated orthogonal transforms, or a
transformed live-basis compact-WY panel primitive.
