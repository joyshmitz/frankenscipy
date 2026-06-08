# frankenscipy-8l8r1.61 proof

## Lever

One lever only: route strict-mode tall full-column-rank `pinv` through the safe-Rust
normal-equation Cholesky solve `pinv(A) = (A^T A)^-1 A^T`.

The route is fail-closed:

- shape gate: `rows >= 2 * cols`, `cols >= 128`
- default threshold gate: `atol == 0.0`, `rtol <= max(rows, cols) * eps`
- finite input and finite output checks
- Cholesky must accept `A^T A`
- right-inverse certificate: `max_abs(pinv(A) * A - I) <= 1e-8 * sqrt(cols)`
- otherwise falls back to the existing public SVD route

## Baseline

Saved pre-change RCH baseline:

- artifact: `baseline_public_route_rch.txt`
- worker: `vmi1227854`
- `routed_lstsq_ms=63.835741`
- `routed_pinv_ms=66.784922`
- ranks: `lstsq_rank=256`, `pinv_rank=256`
- max diffs: `lstsq_max_abs_diff=1.07647224467655178e-12`, `pinv_max_abs_diff=2.28428387316625958e-14`

## After

RCH after benchmark:

- artifact: `after_public_route_rch.txt`
- worker: `vmi1293453`
- `reference_lstsq_ms=146.213048`
- `routed_lstsq_ms=68.415404`
- `reference_pinv_ms=150.138787`
- `previous_route_pinv_ms=71.252799`
- `routed_pinv_ms=13.160669`
- `pinv_speedup_vs_previous_route=5.414071`
- ranks: `lstsq_rank=256`, `pinv_rank=256`
- `lstsq_max_abs_diff=1.07647224467655178e-12`
- `pinv_max_abs_diff=5.77285398303817310e-13`
- `previous_route_pinv_max_abs_diff=2.28428387316625958e-14`
- `routed_vs_previous_route_pinv_max_abs_diff=5.76128337745340779e-13`

The same-run previous-route timing is the keep gate because RCH did not expose
worker pinning and chose a different after worker than the saved baseline.

## Isomorphism

- Ordering/tie-breaking: unchanged. The new route does not sort or select singular values.
- Rank policy: strict full-rank only, `rank = cols`, guarded by Cholesky plus `pinv(A) * A`
  right-inverse verification. Rank-deficient matrices fall back or reject in the helper proof.
- Floating-point policy: output differs from safe SVD by `5.77285398303817310e-13`, below the
  existing public probe tolerance `1e-7`; output differs from the previous public SVD route by
  `5.76128337745340779e-13`.
- RNG: none introduced.
- Certificate action: kept as `SVDFallback`, matching the existing public `pinv` certificate surface.
- Public small golden route: unchanged.

Golden payload SHA-256:

`1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225`

## Score

- Impact: 5.414071
- Confidence: 0.95
- Effort: 1.5
- Score: `5.414071 * 0.95 / 1.5 = 3.429578`

Keep: score >= 2.0.

## Validation

- `rch exec -- cargo test -p fsci-linalg --release --lib --locked pinv_full_rank_tall_cholesky -- --nocapture`
- `rch exec -- cargo test -p fsci-linalg --release --lib public_svd_lstsq_pinv_golden_payload -- --nocapture`
- `rch exec -- cargo test -p fsci-linalg --release --lib --locked public_bidiag_svd_route_perf_probe -- --ignored --nocapture`
- `ubs crates/fsci-linalg/src/lib.rs` exit 0; no critical findings
- `cargo fmt -p fsci-linalg --check`
- `rch exec -- cargo check -p fsci-linalg --lib --locked`
- `rch exec -- cargo clippy -p fsci-linalg --lib --locked --no-deps -- -D warnings`

Full dependency clippy was attempted and is recorded separately; it was blocked
by an existing `fsci-fft` lint at `crates/fsci-fft/src/transforms.rs:2734`, outside this bead.
