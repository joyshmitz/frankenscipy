# PSD Welch SIMD Product-Lane Negative Result

Bead: `frankenscipy-65gu6`

## Verdict

Rejected. The safe portable-SIMD product-lane lever preserved the PSD golden
payload exactly, but the focused RCH Criterion gain was too small to clear the
Score>=2.0 gate. Production source is restored to the pre-trial PSD
implementation.

## Profile-Backed Target

Fresh crate-scoped RCH stats reprofile selected:

- `time_series/psd_welch/4096_w128_o64`: `382.47 us` median, top stats row.

Focused RCH baseline for the bead:

- `time_series/psd_welch/4096_w128_o64`: `[573.01 us, 574.37 us, 575.49 us]`.

## One Lever Tested

The DFT inner loop computed `s*cos` and `s*sin` products with safe
`std::simd::Simd<f64, 4>` lanes, then replayed scalar `re +=` and `im -=`
updates lane by lane in the original sample order.

## Behavior Proof

Golden before and after SHA-256:

`85048a3c06ab045815cbeb238fee9e1e07a05c27ceed3c3782ec0fd5ea97c6b1`

The normalized golden diff is empty.

Preserved surfaces:

- validation and early-return order
- Hann window and twiddle generation
- segment order
- frequency order
- sample order inside each segment-frequency DFT
- scalar addition order for `re` and `im`
- PSD output bin order and final averaging
- RNG absence and tie-breaking absence

## Benchmark Gate

Focused RCH after:

- `time_series/psd_welch/4096_w128_o64`: `[565.44 us, 568.64 us, 571.92 us]`.

Median comparison: `574.37 us -> 568.64 us`, about `1.01x`.

Score: `0.0`; the measured impact is too small for the keep gate.

## Restoration

The production source lever was removed after the failed gate. The next PSD
attempt must use a different primitive than SIMD product extraction plus scalar
replay.
