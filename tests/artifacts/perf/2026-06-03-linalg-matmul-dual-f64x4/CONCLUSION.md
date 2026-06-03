# frankenscipy-lt8kr conclusion

Verdict: REJECTED / SOURCE RESTORED

One lever tried: split each full 4x8 flat-workspace GEMM row accumulator from
one `Simd<f64, 8>` into two `Simd<f64, 4>` lane groups for low/high columns.

Behavior proof passed before benchmarking:

- RCH release matmul tests passed after the edit.
- Stable before/after sorted test-line SHA-256:
  `b756bde7f00b52f08f37f77e67b4f03abcb06b2c551fe04315be57004b40551e`.
- Stable golden diff was empty.

Performance failed the keep gate on the same worker as the baseline:

- Worker: `vmi1149989`
- Baseline `matmul/1024x1024`: `188.06 ms`
- Trial `matmul/1024x1024`: `225.97 ms`
- Score: `0.0` because impact is negative.

Interpretation: splitting the `f64x8` vector into two `f64x4` groups reduces
wide-vector pressure but adds enough extra vector state and operations to lose
on the measured production gate. The next GEMM primitive should not retry SIMD
width splitting; it should attack a different data-movement or register-blocking
shape with a fresh profile-backed baseline.
