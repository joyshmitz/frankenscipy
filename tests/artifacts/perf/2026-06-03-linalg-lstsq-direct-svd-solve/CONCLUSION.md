# Conclusion: Kept

The profile-backed target was `baseline_lstsq/1000x500`, selected after the direct `pinv` sigma-scaling trial was rejected and restored.

The one lever changes rectangular `lstsq_with_casp` only: instead of materializing the full pseudoinverse and multiplying it by `b`, it computes the same SVD least-squares vector directly as `V * Sigma^-1 * U^T * b`. The `pinv` path and square fallback remain on the existing pseudoinverse helper.

The focused RCH Criterion gate improved from `832.97 ms` to `479.61 ms`, with a confirmation run at `470.34 ms`. The sorted stable RCH golden hash stayed `c2a369f0e0798d8f56cf5f323abdb22229710ab2e43cf7acd58afe2c0341f8f7`.

Decision: keep with Score `6.0`.
