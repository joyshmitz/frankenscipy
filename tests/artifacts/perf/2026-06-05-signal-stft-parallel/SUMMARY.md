# fsci-signal stft: single-threaded frame loop -> multithreaded frames

## Target (zero-threaded-crate vein; fsci-signal had ZERO threading)

stft (and spectrogram, which calls it) computed one window+rfft per frame in a
sequential while loop. Each frame is an independent, expensive O(nperseg log nperseg)
rfft over a disjoint segment -> embarrassingly parallel, byte-identical.

## Lever (one)

Compute each frame (window + rfft -> spectrum row + time) in parallel across
std::thread::scope workers, preserving frame order and returning the first FFT error
in order. Gated on FFT flops = frame_count * nperseg * log2(nperseg) >= 2^24, with
nthreads capped at frame_count/4 -- so cheap small-nperseg STFTs stay SEQUENTIAL (the
barycentric over-trigger lesson: at nperseg=512/1041 frames parallel was 0.95x).

## Isomorphism / proof (BYTE-IDENTICAL)

Each frame runs the same deterministic window+rfft, so every spectrum value is
identical regardless of thread; order preserved. Stash A/B digest over all complex
zxx values UNCHANGED (b5184... style; f583de40076920eb, 8ce374156acff23b). New test
stft_parallel_is_bit_identical (~4200 frames of nperseg=512 -> parallel path; f64::
to_bits equal vs a verbatim sequential reference using get_window). fsci-signal 502
passed / 0 failed; lib clippy-clean (the only -D-warnings failure is the active agent's
dirty fsci-linalg dependency, not this change); fmt clean.

## Same-process A/B (perf_stft bin, seq-vs-par in ONE process, worker-variance-immune)

| n | nperseg | frames | seq | par | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 800000 | 1024 | 1041 | 10.58 ms | 11.08 ms | 0.95x (gated sequential) |
| 1200000 | 2048 | 780 | 12.30 ms | 5.03 ms | 2.44x |
| 2500000 | 1024 | 3254 | 24.92 ms | 9.00 ms | 2.77x |

Clean >2x for large STFTs; small ones gated sequential (no regression). Byte-identical.
6th zero-threaded crate parallelized (spatial/ndimage/stats/interpolate/cluster/signal).
