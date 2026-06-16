# frankenscipy-psn7x.9 Evidence - Band Lanczos Eigenvector Route

## Target

- Bead: `frankenscipy-psn7x.9`
- Target: `fsci-linalg` symmetric band-to-tridiagonal/eigenvector route
- Source candidate: full-reorthogonalized Lanczos over lower-band storage, with
  deterministic restarts and tridiagonal eigenvectors lifted back through the
  Lanczos basis.

## Baseline/Profile

The current `symmetric_eigh_native_stage_breakdown_probe` on RCH
`vmi1152480` still shows the dense native route dominated by reduction and
backtransform:

| shape | reduction | tridiagonal | backtransform | sort | values digest |
| --- | ---: | ---: | ---: | ---: | --- |
| 400x400 | 27.973202 ms | 72.611878 ms | 33.637319 ms | 0.596819 ms | `0x0dbbde75b75c8612` |
| 800x800 | 201.568494 ms | 89.439556 ms | 254.452438 ms | 2.718484 ms | `0x4461962827bdb038` |
| 1200x1200 | 650.642314 ms | 149.004755 ms | 619.193477 ms | 5.355747 ms | `0x2fc45e1f18ceb0ab` |

## Candidate Proof

`symmetric_lower_band_lanczos_eigen` passed the dense-reference proof on
`vmi1152480`:

- n=16/bw=3 residual `2.48689957516035065e-14`
- n=33/bw=5 residual `9.94759830064140260e-14`
- n=96/bw=8 residual `3.97903932025656104e-13`

Ordering/tie behavior: eigenvalues are sorted with `total_cmp`, matching the
existing native route. RNG behavior: no public RNG is used; deterministic
Lanczos seed/restart functions derive only from `n` and existing fixtures.
Floating-point behavior: the candidate intentionally changes the band
eigenvector route arithmetic, so acceptance requires residual/orthogonality and
same-worker timing evidence rather than bit identity.

## Same-Worker Timing

RCH worker for `eig_banded_eigenvectors_perf_probe`: `vmi1149989`.

| shape | baseline native route | Lanczos candidate | speedup |
| --- | ---: | ---: | ---: |
| 128x128 bw32 | 2.557600 ms | 4.370042 ms | 0.585259x |
| 256x256 bw32 | 12.474966 ms | 32.501430 ms | 0.383829x |

The candidate preserved residual and eigenvalue tolerance but regressed both
measured shapes, so no production source is kept.

## Decision

Score: `Impact 0.0 * Confidence 4.0 / Effort 2.0 = 0.0`.

Rejected/no-ship. Do not retry full-reorthogonalized Lanczos eigenvectors for
the public banded eigenvector route. The next primitive should be a true
implicit symmetric-band bulge chaser with accumulated orthogonal transforms, or
a transformed compact-WY panel generator that proves live-basis panel vectors
before timing.
