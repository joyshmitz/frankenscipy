# perf: parallelize read_csv per-row f64 parsing (exact serial fallback)

Bead: frankenscipy-7eizt

## Lever
`read_csv` (crates/fsci-io/src/lib.rs) parsed `split(delimiter).trim().parse::<f64>()` for
every field in a serial row loop; field parsing dominates on large CSVs. Each data row is
an independent `Vec<f64>` with no cross-row state beyond the column count (fixed by the
first data row / header), so split the data lines into contiguous chunks, parse each on its
own core, and concatenate the per-chunk row lists in order.

## Isomorphism / byte-identity argument
- The header is parsed identically (first line, split + trim). `expected_cols` is set by the
  first non-comment data row, exactly as the serial loop.
- Each chunk parses its rows in order with the same deterministic `f64::parse`, skips the
  same blank/`#` lines, and checks `row.len() == expected_cols`; chunks concatenate in order,
  so `data` is identical to the serial loop.
- No floating-point reduction occurs (values copied, not summed).
- MALFORMED / EDGE INPUT (empty-with-header, header/first-row column mismatch, any row
  column mismatch, parse error) replays the verbatim serial loop `read_csv_serial`, so the
  returned `(header, data)` and every `Err` (message + which row) are byte-for-byte the
  original behaviour.
- Gate: serial path for `< 4096` data lines.

⇒ Output and error are bit-identical to the serial implementation.

## Proof (golden — serial baseline vs parallel NEW, identical)
Harness: `cargo run --profile release-perf -p fsci-io --bin perf_read_csv`

```
req=(100x5,hdr=true)   rows=100   hcols=5 data_xor_bits=e2b9c88562b66a47
req=(5000x8,hdr=true)  rows=5000  hcols=8 data_xor_bits=832320568946efff
req=(20000x3,hdr=false)rows=20000 hcols=0 data_xor_bits=d4199d7a770d2fe2
```
Identical bits/rows/hcols in the stashed serial build and the parallel build.
sha256(golden payload file) = c52b85bf060eb8f0fe6822c952cf5eca06b05c93241aa451b801ade9fa5e9724

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| file              | serial (3x)          | new (3x)            | speedup |
|-------------------|----------------------|---------------------|---------|
| 300k rows × 8 col | 104.9/120.1/105.3 ms | 26.4/27.6/26.4 ms   | ~4.0x   |
| 800k rows × 4 col | 138.8/145.3/151.1 ms | 41.1/46.6/49.6 ms   | ~3.1x   |

(Lower than loadtxt's flat-buffer ~5x because read_csv allocates a `Vec<f64>` per row.)

## Validation
11 csv unit tests pass; clippy clean (the only warning is in the fsci-fft dep).
