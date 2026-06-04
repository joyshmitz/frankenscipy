# fsci-signal CWT Same-slice extraction rejection

Bead: `frankenscipy-dtje6`

## Profile Target

`br ready --json` was empty after the special parity closeout, while active
perf beads were already owned on linalg and sparse. The fallback target is
profile-backed by the signal reprofile:

`tests/artifacts/perf/2026-06-02-signal-profile/reprofile_after_remez_broad_rch.txt`

After the remez optimization, `wavelets/cwt_ricker/2048x32` is the dominant
remaining `fsci-signal` benchmark in that artifact:

```text
time: [5.4500 ms 5.6145 ms 5.8008 ms]
```

Focused RCH baseline before this lever:

```text
RCH_FORCE_REMOTE=1 rch exec -- cargo bench -p fsci-signal --bench signal_bench --locked -- wavelets/cwt_ricker/2048x32 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
worker: vmi1149989
time: [2.5571 ms 2.7045 ms 2.8527 ms]
```

## Lever

In the CWT FFT convolution path, the old code materialized:

```text
conv_full -> full Vec<f64> -> Same slice Vec<f64>
```

The new code collects the returned `Same` slice directly from `conv_full`:

```text
conv_full[start..start + na].re -> Same slice Vec<f64>
```

No FFT inputs, twiddle/cache paths, multiply order, inverse FFT, output order,
tie-breaking, or RNG behavior changes.

## Isomorphism Proof

Golden output SHA-256 before:

```text
6d7e109d6aa72ed23d593e2a77f12b0b6f825c339b4ede0346a35450ed24d27b
```

Golden output SHA-256 after:

```text
6d7e109d6aa72ed23d593e2a77f12b0b6f825c339b4ede0346a35450ed24d27b
```

`cmp golden_before.txt golden_after.txt` exited `0`.

Same-process A/B harness output:

```text
mode=cwt-ab repeats=20 bit_identical=true old_ms=3.591057 new_ms=3.356606 speedup=1.069848
```

## Benchmark Result

The first after Criterion run landed on a different worker (`vmi1152480`) and is
not directly comparable to the baseline:

```text
time: [5.5840 ms 6.0087 ms 6.4307 ms]
```

An initial 20-repeat same-process A/B looked positive before the harness was
tightened:

```text
mode=cwt-ab repeats=20 bit_identical=true old_ms=3.591057 new_ms=3.356606 speedup=1.069848
```

After tightening the finalized harness to avoid new direct-indexing findings,
the confirmation runs did not show a stable win:

```text
mode=cwt-ab repeats=20 bit_identical=true old_ms=4.663348 new_ms=4.625346 speedup=1.008216
mode=cwt-ab repeats=100 bit_identical=true old_ms=4.522551 new_ms=5.331721 speedup=0.848235
```

Decision: rejected. The production and harness edits were not kept because the
finalized same-process confirmation did not clear the real-win threshold.
