# Pass 3 - Alien Primitive Selection: PSD Frequency-Major Traversal

Bead: `frankenscipy-kko0m`

## Measured Target

Focused RCH Criterion:

`rch exec -- cargo bench -p fsci-stats --bench stats_bench --locked -- time_series/psd_welch/4096_w128_o64 --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot`

- Baseline: `[717.61 us, 746.46 us, 799.07 us]`
- Worker: `vmi1153651`
- Golden before sha256: `85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`

## Selected Primitive

Name: frequency-major cached-segment traversal.

Graveyard lineage: polyhedral loop interchange plus cache-aware vectorized
execution. The dependence proof is simple because each output frequency is an
independent reduction over segments, and the segment order inside that
frequency can remain unchanged.

Implementation form:

1. Validate inputs and choose the existing `WelchPlan` exactly as before.
2. Pre-window all overlapping segments into one flat row-major `Vec<f64>`.
3. Traverse `k` first, then segment, then sample:
   - twiddle row for frequency `k` is loaded once and reused across all segments;
   - segment order for `psd[k] += power` remains ascending;
   - sample order inside each segment-frequency DFT remains ascending.
4. Average the PSD exactly as before.

Rejected variants for this pass:

- FFT/Goertzel replacement: likely faster, but changes floating-point operation
  order and fails the golden-sha contract.
- Segment-buffer reuse: already rejected in `frankenscipy-8l8r1.3`.
- Frequency special cases for DC/Nyquist: too small and risks signed-zero or
  trigonometric edge differences.

## Score

| Primitive | Impact | Confidence | Effort | Score |
| --- | ---: | ---: | ---: | ---: |
| Frequency-major cached-segment traversal | 2 | 3 | 2 | 3.0 |

Keep only if the RCH after-benchmark and golden proof keep the rescored value
at `>=2.0`.

## Isomorphism Obligations

- Output ordering: `psd[k]` remains frequency-index ascending.
- Segment ordering: for each `k`, segment powers are added to `psd[k]` in the
  same ascending segment order as the generic path.
- Sample ordering: each `(segment, k)` DFT loops sample index ascending.
- Floating point: each segment sample is still `data[start+n] * win[n]`; each
  DFT step still uses `re += s * cos` and `im -= s * sin` with the same cached
  twiddle values.
- RNG/global state: unchanged; none exists.
- Tie-breaking: unchanged; no comparisons or ordering decisions exist.
- Validation/error behavior: empty input, zero window, invalid `fs`, overlap
  clamp, and hop computation stay before traversal.
