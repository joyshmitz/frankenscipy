# sparse eye direct CSR keep

Target: `sparse_eye/eye/10000`, selected from the sparse reprofile where `eye`
was still constructing COO triplets and converting to CSR.

Lever: construct the canonical square identity CSR directly in `eye(size)`:
`data = vec![1.0; size]`, `indices = 0..size`, `indptr = 0..=size`.

Behavior proof:
- Golden payload generated before and after with a temporary `perf_sparse eye-golden` mode.
- Extracted payload SHA-256 before: `dbce92bdfd716949971b0450881e47b1c09ffa581eeae927dcfd1ce737cb0b50`.
- Extracted payload SHA-256 after: `dbce92bdfd716949971b0450881e47b1c09ffa581eeae927dcfd1ce737cb0b50`.
- `golden_eye_payload_from_log.diff` is empty.
- `cargo test -p fsci-sparse --lib eye -- --nocapture --test-threads=1`: 9 passed.

Benchmark evidence:
- Baseline, focused RCH Criterion on `ovh-a`: `sparse_eye/eye/10000`
  `[26.897 us 27.002 us 27.124 us]`.
- After, default RCH Criterion on `ovh-a`: `sparse_eye/eye/10000`
  `[16.618 us 16.711 us 16.816 us]`.
- Same-worker speedup by mean: `27.002 / 16.711 = 1.62x`.
- Cross-check: earlier full sparse reprofile baseline on `vmi1227854`
  `[27.251 us 28.024 us 28.780 us]`; after focused row on `vmi1227854`
  `[16.820 us 17.508 us 18.192 us]`, speedup by mean `1.60x`.

Score:
- Impact: 1.62
- Confidence: 0.95
- Effort: 0.5
- Impact x Confidence / Effort: 3.08

Validation:
- `cargo fmt --check --package fsci-sparse`: pass.
- `cargo check -p fsci-sparse --all-targets`: pass on RCH.
- `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings`: pass on RCH.
- `ubs crates/fsci-sparse/src/construct.rs`: exit 0.
- Full `cargo clippy -p fsci-sparse --all-targets -- -D warnings` was attempted and blocked by an unrelated existing `fsci-fft` lint: `manual_is_multiple_of` in `crates/fsci-fft/src/transforms.rs:2734`.
