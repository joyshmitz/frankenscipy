# perf(fsci-interpolate): make_smoothing_spline sparse assembly + band-bounded solve

## Levers
1. SPARSE SMOOTHING ASSEMBLY: make_smoothing_spline_impl used O(n^2) eval_basis_all per
   sample. Apply the same lever as make_lsq_spline — bspline_find_interval (O(log n)) +
   windowed Cox-de Boor into a reused scratch + nonzero-window scatter -> O(n*k^2).
2. BAND-BOUNDED SOLVE: solve_banded bounds GEPP to [col, col+bw] rows and [col, col+2bw]
   cols. Bit-identical to the dense solve for a bandwidth-bw matrix (L keeps lower bw, U
   gains upper bw 2*bw with partial pivoting, so every skipped entry is exactly 0). Routes
   make_lsq_spline (bw=k) and make_smoothing (bw=k.max(2)); solve O(n^2-scan) -> O(n*bw^2).

## Byte-identity
perf_smoothing_spline eval-grid digests + perf_lsq_spline coeff digests OLD(committed)==NEW:
  smoothing n=800/1500/3000  90b183c8.. / f2ee4491.. / 47faf02b..
  lsq m1500/3000/5000        8117653a.. / d4069f02.. / 2fbddeee..
126 fsci-interpolate tests pass; clippy clean.

## Bench (release-perf, min of 3)
make_smoothing_spline (UnivariateSpline::new, s>0):
  n= 800:  4.29 -> 2.36 ms  1.82x
  n=1500: 16.32 -> 8.68 ms  1.88x
  n=3000: 79.67 -> 37.78 ms 2.11x   (grows ~n: the O(n^2) dense A^T A alloc becomes a
                                      smaller fraction; >2x by n>=3000)
make_lsq_spline (band solve adds to the prior O(m*k^2) build):
  n=200/400/600: 1.39x / 1.31x / 1.43x

## NEXT: the residual O(n^2) is the dense n*n A^T A allocation. Banded storage
(n*(2bw+1), LAPACK gbsv-style with the pivoting row-swap handled in band layout) removes
it for a fully O(n*bw^2) smoothing-spline build.
