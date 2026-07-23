# Special-function array gap-hunt vs scipy (2026-07-23, thinkstation1, cc)

n=1e6 array. fsci default-threaded (par_continuous_map) vs scipy.special 1-thread cephes.

| fn | fsci | scipy | fsci vs scipy |
|---|---|---|---|
| gamma | 5.30 ms | 21.65 ms | 4.1x FASTER |
| erf | 6.47 ms | 13.48 ms | 2.1x FASTER |
| digamma | 4.47 ms | 16.46 ms | 3.7x FASTER |
| j0 | 6.65 ms | 23.70 ms | 3.6x FASTER |
| i0 | 22.29 ms | 40.22 ms | 1.8x FASTER |
| k0 | 11.78 ms (noisy) | 40.75 ms | 3.5x FASTER |
| expi | ~28 ms (noisy [11,71]) | 219.93 ms | ~8x FASTER |
| zeta | 11.01 ms | 151.30 ms | 13.7x FASTER |

## FINDING: fsci WINS every special-function array — NO vs-scipy gap.
fsci parallelizes the array map across cores + no Python dispatch; scipy's cephes is
single-threaded C. Even fsci's worst-variance case (expi 71ms) beats scipy (220ms). The
special-function surface is NOT a gap to close — fsci already dominates on throughput.
Minor: expi/k0 show data-dependent variance (some inputs hit a slower branch) — an internal
consistency question, NOT a vs-scipy gap (still beats scipy), low priority.

## COMPETITIVE MAP (this session's gap-hunts)
- Dense factorizations (cholesky/eigh/svd/qr/schur): ~2-2.7x SLOWER = the vs-LAPACK walls,
  blocked on per-step-spawn / pool substrate (vndri) or WY-blocked rewrite (2o0vp).
- Special functions (array): fsci FASTER (parallel + no-Python).
So fsci's vs-scipy gaps are CONCENTRATED in the dense LAPACK-backed factorizations; the
domain/scalar surfaces fsci is competitive-to-winning. Remaining dense wins need the heavy
structural work; domain gap-hunts keep confirming fsci-competitive.
