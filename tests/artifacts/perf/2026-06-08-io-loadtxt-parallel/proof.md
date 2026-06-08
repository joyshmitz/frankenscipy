# perf: parallelize loadtxt per-line f64 parsing (exact serial fallback)

Bead: frankenscipy-66uos

## Lever
`loadtxt` (crates/fsci-io/src/lib.rs) parsed `trimmed.split_whitespace().parse::<f64>()`
for every field in a serial line loop — and field parsing dominates on large numeric
files (numpy.loadtxt is famously slow for the same reason). Each data line maps to its
own contiguous output row with no cross-line state (the only shared value is `cols`, set
by the first data row), so split the line list into contiguous chunks, parse each chunk
on its own core into a local buffer, and concatenate the buffers in chunk order.

## Isomorphism / byte-identity argument
- `cols` is taken from the first non-comment line — exactly as the serial loop.
- Each chunk parses its lines in order with the same deterministic `f64::parse`, skips the
  same comment/blank lines, and validates `count == cols`; buffers are concatenated in
  chunk order, so the flat `data` and `rows` are identical to the serial loop.
- No floating-point reduction occurs (values are copied, not summed).
- MALFORMED INPUT: if any chunk sees a parse failure or a column-count mismatch, the
  function discards the parallel work and replays `loadtxt_serial` (the original loop,
  kept verbatim), so the returned `Err` — message text and which row/error it reports —
  is byte-for-byte the original behaviour.
- Gate: serial path for `< 4096` lines.

⇒ Output and error are bit-identical to the serial implementation.

## Proof (golden — serial baseline vs parallel NEW, identical)
Harness: `cargo run --profile release-perf -p fsci-io --bin perf_loadtxt`
(`data_xor_bits` folds every parsed f64's bits; rows/cols reported.)

```
req=(100x5)   rows=100   cols=5 data_xor_bits=ce6022f13851dae2
req=(5000x8)  rows=5000  cols=8 data_xor_bits=73b58002977bfaba
req=(20000x3) rows=20000 cols=3 data_xor_bits=43f8da462520568f
```
Identical bits/rows/cols in the stashed serial build and the parallel build.
sha256(golden payload file) = f68215a536e4ce9ff86e91f2109e56817319eb2a7ebdba6dc1da7ac1df029d8c

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| file              | serial (3x)         | new (3x)            | speedup |
|-------------------|---------------------|---------------------|---------|
| 300k rows × 8 col | 117.5/118.2/120.3 ms| 22.2/22.4/22.8 ms   | ~5.3x   |
| 800k rows × 4 col | 160.9/161.6/167.9 ms| 36.0/36.5/38.0 ms   | ~4.4x   |

## Validation
2 loadtxt unit tests pass; clippy clean (the only warning is in the fsci-fft dep).
