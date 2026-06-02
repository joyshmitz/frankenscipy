# fsci-fft rfft unpack twiddle cache

Bead: `frankenscipy-7ztox`

Profile target:
- RCH P2C-005 profile before this lever ranked `rfft n=1024` at `108245 ns`, `4.97%` of total profile time.
- Post-lever RCH P2C-005 profile top three shifted to:
  - `polynomial_multiply_fft degree<1024 x degree<1024`: `1.106 ms`, `61.8%`
  - `fft2 32x32`: `0.271 ms`, `15.2%`
  - `irfft n=1024`: `0.123 ms`, `6.9%`

One production lever:
- `real_fft_specialized` reuses the existing `get_or_compute_twiddles(n, false)` table for unpack twiddles.
- Removed per-bin `cos`/`sin` calls from the `rfft` unpack loop.

RCH Criterion:
- Baseline: `cargo bench -p fsci-fft --bench fft_bench --locked fft_real/rfft/1024 -- --sample-size 50`
  - Worker: `vmi1156319`
  - Mean: `30.322 us`
  - Interval: `[29.539 us, 31.224 us]`
- After: same command
  - Worker: `vmi1227854`
  - Mean: `9.9084 us`
  - Interval: `[9.2992 us, 10.621 us]`

Behavior proof:
- Golden command before/after: `cargo run -q -p fsci-fft --profile release-perf --bin perf_fft --locked -- rfft-golden <path>`
- Before sha256: `d3e41795d153a1f884a968b318d23815a19ae7faeafbf01dc3d1f125af99c16a`
- After sha256: `d3e41795d153a1f884a968b318d23815a19ae7faeafbf01dc3d1f125af99c16a`
- `cmp` result: byte-identical
- Isomorphism: output ordering, half-spectrum length, normalization, conjugate symmetry, and deterministic input generation are unchanged. No RNG or tie-breaking path is involved. The twiddle values come from the same `-2*pi*k/n` formula already used by the transform twiddle cache.

Validation:
- RCH `cargo test -p fsci-fft --locked`: `144` unit tests, `54` metamorphic tests, and doc tests passed.
- RCH `cargo clippy -p fsci-fft --lib --bins --tests --locked -- -D warnings`: passed.
- RCH `cargo clippy -p fsci-fft --all-targets --locked -- -D warnings`: blocked before linting because Cargo wanted to update the lockfile under `--locked`.
- RCH `cargo clippy -p fsci-fft --all-targets --offline -- -D warnings`: blocked before linting because the remote offline index lacked `proptest` for another workspace crate.
- Local `cargo fmt -p fsci-fft --check`: passed.
- UBS on changed FFT files: exit `0`; no critical findings.

Score:
- Impact: `3`
- Confidence: `4`
- Effort: `1`
- Impact x Confidence / Effort: `12`
