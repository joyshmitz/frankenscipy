# Conclusion

The `6 x 8` register-fit full-tile microkernel is behavior-preserving but rejected.

- Baseline `matmul/1024x1024`: `817.83 ms`
- After `matmul/1024x1024`: `882.22 ms`
- Stable golden SHA before/after: `b756bde7f00b52f08f37f77e67b4f03abcb06b2c551fe04315be57004b40551e`
- Stable golden diff: empty
- Source restore proof: `source_diff_after_reject_restore.txt` is empty
- Score: `0.0` because the decisive profile-backed target regressed

Next primitive target: stop row-count calibration and attack a fundamentally different GEMM primitive, preferably a bounded cache-oblivious or packed-panel design that preserves per-cell monotonic `k` order while improving memory locality.
