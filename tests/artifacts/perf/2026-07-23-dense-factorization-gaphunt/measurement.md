# Dense factorization gap-hunt vs scipy (2026-07-23, thinkstation1, cc)

Same-box, scipy 1.17.1 1-thread (OMP/OPENBLAS=1) vs fsci criterion.
**WELL-CONDITIONED random full-rank matrix** (xorshift, matching scipy standard_normal).

| fn | fsci n=256 | scipy n=256 | ratio | fsci n=512 | scipy n=512 | ratio |
|---|---|---|---|---|---|---|
| svd | 25.87 ms | 11.58 ms | 2.23x | 170.27 ms | 76.71 ms | 2.22x |
| qr | 4.44 ms | 2.76 ms | 1.61x | 38.13 ms | 19.14 ms | 1.99x |
| schur | 60.33 ms | 27.79 ms | 2.17x | ~ | 230.78 ms | ~2x |

## PHANTOM CORRECTED
First pass used make_matmul_matrix (values = (i*31+j*17+seed)%97*0.01) = a STRUCTURED,
RANK-DEFICIENT matrix — pathological for SVD: it fails the fast-path acceptance gate
(clustered/tied singular values) and falls to the robust nalgebra/Jacobi fallback AFTER
already doing the bidiag reduction. That double-work gave a FALSE 10.7x svd gap and a
FALSE "schur 1.45x FASTER". With a well-conditioned random matrix the real gaps are all
~2x. LESSON: match the input DISTRIBUTION to the reference's (scipy used standard_normal);
a rank-deficient microbench input exercises a different (robust-fallback) code path.

## STRUCTURE
svd fast path (n=512, thin-svd stage probe): reduction 116ms / bidiag-svd 49ms /
back-transform U+V 30ms (ALREADY parallel). The Householder REDUCTION dominates (59%),
serial — SAME per-step-spawn wall as the eigh tridiagonalization (REJECTED d3c6f9679).
qr/schur share the Householder-reduction structure. ALL dense factorizations are uniformly
~2x behind LAPACK, gated by the SAME structural blocker: per-step-spawn / pool substrate
(vndri) OR blocked-algorithm rewrites (WY dsytrd, D&C). No dense-factorization outlier.
