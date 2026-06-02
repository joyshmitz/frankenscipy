# fsci-fft irfft reconstructed-spectrum clone pass

Bead: `frankenscipy-0j82y`

Status: rejected. The production lever was reverted because the remote
Criterion after-sample did not show a real win.

## Profile target

Post `frankenscipy-7ztox` P2C-005 re-profile shifted the top FFT targets to:

- `polynomial_multiply_fft`: 1.106 ms, 61.8%
- `fft2 32x32`: 0.271 ms, 15.2%
- `irfft n=1024`: 0.123 ms, 6.9%

Candidate lever: `real_ifft_unscaled` owns the reconstructed Hermitian spectrum,
so it could call `transform_1d_inplace` and avoid the clone inside
`transform_1d_unscaled`.

## Behavior proof

Golden generator: `perf_fft irfft-golden`.

- Before sha256: `5def432af7c8df2572704b874b4eb3fae42cd9df5e5dc1cf2e9a182d013009cc`
- After sha256: `5def432af7c8df2572704b874b4eb3fae42cd9df5e5dc1cf2e9a182d013009cc`
- Before size: 23287 bytes
- After size: 23287 bytes
- `cmp -s`: identical

Isomorphism: the candidate preserved `rebuild_hermitian` output ordering,
inverse flag, backend selection, normalization, and real-part extraction. It
changed only the allocation shape by transforming the owned reconstructed
spectrum in place rather than cloning it first.

## Benchmark

Baseline, via RCH Criterion:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-fft --bench fft_bench --locked fft_irfft/irfft/1024 -- --sample-size 50
worker: vmi1149989
fft_irfft/irfft/1024 time: [10.150 us 10.745 us 11.373 us]
```

After candidate, via RCH Criterion:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-fft --bench fft_bench --locked fft_irfft/irfft/1024 -- --sample-size 50
worker: vmi1156319
fft_irfft/irfft/1024 time: [21.974 us 22.636 us 23.339 us]
```

Result: rejected below the `Impact x Confidence / Effort >= 2.0` threshold.
The retained diff only extends the FFT perf harness with `irfft` and
`irfft-golden` modes for future profile-backed passes.
