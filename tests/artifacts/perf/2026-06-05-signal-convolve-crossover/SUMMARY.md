# fsci-signal convolve: FFT-dispatch threshold na·nb>1000 -> cost-model crossover

## Target (bead frankenscipy-z5p2b)

convolve (lib.rs) switched to FFT at `na·nb > 1000`. FFT's large constant (three
length-L transforms) means the direct loop is faster — and byte-identical — until
na·nb ≈ 2.1e5. So medium 1D convolutions (1000 < na·nb < ~2.1e5) took the SLOWER
FFT path: a latent perf regression AND a small parity drift (FFT rounding vs scipy
direct). correlate() inherits this via delegation.

## Lever (one)

Replace the constant threshold with `fft_conv_is_faster(na, nb)`:
FFT only when `na·nb > 20 · L·log2(L)`, L = next_pow2(na+nb-1). Constant 20 is the
measured break-even (direct ≈ 0.3 ns/op vs FFT ≈ 6 ns per L·log2 L unit), the same
calibration used by polymul and correlate2d. cwt's FFT path (which caches the
forward FFT of the shared `data` operand across scales, a different/lower crossover)
is intentionally left unchanged.

## Parity

Below the crossover convolve now runs the direct loop => BYTE-IDENTICAL to a
verbatim direct convolution (perf_convolve `exact_vs_direct=true`, max_abs=0 for
n=40..384), and strictly closer to scipy's exact direct result than the old
FFT-rounded output. Above the crossover it still uses FFT (n=512: max_abs 4e-13).
Conformance diff_signal_convolve + diff_signal_correlate pass; 15 convolve unit
tests pass; clippy + fmt clean.

## Same-process A/B sweep (perf_convolve bin, Full mode)

| n (= na = nb) | na·nb | convolve (now) | old forced-FFT | speedup | parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 40  | 1600   | 0.417us | 4.257us | 10.22x | byte-identical |
| 64  | 4096   | 0.757us | 4.056us | 5.36x  | byte-identical |
| 100 | 10000  | 1.686us | 10.320us | 6.12x | byte-identical |
| 160 | 25600  | 6.277us | 16.926us | 2.70x | byte-identical |
| 256 | 65536  | 18.11us | 16.83us | 0.93x | byte-identical (near crossover) |
| 384 | 147456 | 30.93us | 34.13us | 1.10x | byte-identical |
| 512 | 262144 | 33.66us | 27.88us | (both FFT) | FFT path retained |

The medium regime (na·nb 1.6e3..2.6e4) — exactly what the old gate mis-routed —
is now 2.7-10.2x faster and bit-exact. No meaningful regression (the lone 0.93x at
the crossover is byte-identical and matches scipy's direct choice).
