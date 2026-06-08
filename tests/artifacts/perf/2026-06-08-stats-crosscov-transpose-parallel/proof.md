# perf: cross_cov transpose + parallel streamed dot (cov_matrix vein)

Bead: frankenscipy-a8jhx

## Lever
`cross_cov` (crates/fsci-stats/src/lib.rs) built the dx×dy cross-covariance by scatter-
writing the whole matrix once per observation:
```
for i in 0..n { for j in 0..dx { for k in 0..dy {
    cov[j][k] += (x[i][j]-mx[j])*(y[i][k]-my[k]); }}}
```
For dx·dy big enough to spill cache that is memory-bound. But
`cov[j][k] = Σ_obs (x[obs][j]-mx_j)·(y[obs][k]-my_k)` is exactly a dot product of two
centered series over observations in a FIXED order. Transpose+center each dataset into
contiguous per-variable series (`xt[j]`, `yt[k]`), turning every entry into a streamed,
cache-friendly, auto-vectorisable dot, and making the output rows independent so they fan
out across threads. (Same lever already shipped for `cov_matrix`.)

## Isomorphism / byte-identity argument
- For a fixed `(j,k)` the serial scatter accumulates over observations in ascending `i`;
  the transposed dot sums `xt[j][i]*yt[k][i]` over the same ascending `i`. Identical terms,
  identical order ⇒ identical f64 (no reduction reassociated).
- Centering `xt[j][i] = x[i][j]-mx[j]` precomputes the exact value the scatter formed inline.
- Means are accumulated in the same observation order and divided by `n`; each entry is
  divided once by `n-1`, as before.
- Small problems (`dx·dy·n < 2^22`) keep the original scatter verbatim (bit-identical
  reference). The transposed path is itself bit-identical to it.

⇒ Every matrix entry is bit-for-bit identical to the serial implementation.

## Proof (golden — serial scatter vs transpose+parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-stats --bin perf_cross_cov`
```
n=50  dx=6   dy=4  shape=(6,4)   xor_bits=5d8a47cd42f2623b
n=200 dx=80  dy=64 shape=(80,64) xor_bits=195f6164545cf667
n=500 dx=120 dy=90 shape=(120,90)xor_bits=3684d37030f9b75d
```
Identical bits in the stashed serial build and the transpose+parallel build.
sha256(golden payload file) = b30c0ee2f02327ca1456d7fb209a62d5a74cae5a7fd999ccf087583c72096e50

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| problem               | serial (3x)          | new (3x)           | speedup |
|-----------------------|----------------------|--------------------|---------|
| n=2000, dx=dy=160     | 51.1/49.0/52.0 ms    | 9.9/7.2/7.2 ms     | ~7x     |
| n=4000, dx=dy=200     | 151.3/158.1/154.6 ms | 16.6/14.8/17.6 ms  | ~9x     |

## Validation
fsci-stats builds; clippy: no new warning from cross_cov (the pre-existing fsci-stats
warnings are at lib.rs:34656 / 34921-2 (cov_matrix mirror loop) / 35842, none in cross_cov).
