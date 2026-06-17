# frankenscipy-psn7x.19 - Production Band-To-Tridiagonal Baseline

Date: 2026-06-17
Agent: RubyWaterfall
Target: `eig_banded(lower=true, eigvals_only=false)` and native symmetric-eigh reduction
Worker: local only, per ts1/RCH-offline override

## Reason

After `frankenscipy-psn7x.18` kept the compact diagonal-lane envelope update, `br ready --json` had no ready perf beads. This bead was created from fresh local profile evidence instead of waiting idle.

The next primitive must be production-facing: move the compact band work toward `eig_banded(..., eigvals_only=false)` so it avoids dense `symmetric_eigh_native` reduction for lower-band inputs. Do not spend another pass on benchmark-only transformed dense emission.

## Public Banded Baseline

Command:

```bash
env RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-psn7x17-target \
  hyperfine --warmup 1 --runs 5 --show-output \
  'cargo test -j 1 -p fsci-linalg --lib eig_banded_eigenvectors_perf_probe --release --locked -- --ignored --nocapture'
```

Transcript: `baseline_eig_banded_eigenvectors_local_hyperfine.txt`

Result:

- Wall: `210.6 ms +/- 5.6 ms`
- 128x128 bw32 candidate range: `3.321615-3.841108 ms`
- 256x256 bw32 candidate range: `13.419694-16.822633 ms`
- Residuals stayed `1.64845914696343243e-12` at 128 and `7.73070496506989002e-12` at 256
- Values digests stayed `0xd6dbb9200f65bd92` and `0x09ed4d367faab431`
- Vector digests stayed `0x6cf3573b5b50c275` and `0xc32797c0d224a75a`

## Stage Split

Command:

```bash
env RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenscipy-rubywaterfall-psn7x17-target \
  cargo test -j 1 -p fsci-linalg --lib symmetric_eigh_native_stage_breakdown_probe --release --locked -- --ignored --nocapture
```

Transcript: `profile_symmetric_eigh_stage_split_local.txt`

Result:

- 400x400: reduction `13.865564 ms`, tridiagonal eigen `10.344970 ms`, backtransform `6.895181 ms`, sort `0.917507 ms`
- 800x800: reduction `107.966982 ms`, tridiagonal eigen `40.287203 ms`, backtransform `40.239793 ms`, sort `3.232749 ms`
- 1200x1200: reduction `365.980855 ms`, tridiagonal eigen `90.414421 ms`, backtransform `119.430150 ms`, sort `6.562060 ms`
- Digests stayed `0x0dbbde75b75c8612`, `0x4461962827bdb038`, `0x2fc45e1f18ceb0ab`

## Next Lever Boundary

Allowed direction:

- implement a true production band-to-tridiagonal/eigenvector route for `eig_banded(lower=true, eigvals_only=false)` using compact lower-band storage and Q replay
- prove public values/vectors/residuals/orthogonality/order against the current dense oracle
- rebench with the same public `eig_banded_eigenvectors_perf_probe`

Do not retry:

- benchmark-only transformed dense materialization cleanup
- direct-index rotation packet wrappers
- full active lower-envelope storage
- fixed envelope width guesses
- full-reorthogonalized Lanczos eigenvectors
- shifted inverse iteration over band solves
- worker-count retuning
- raw/stale compact-WY panels
- scalar spelling or SIMD rank-2 vector spelling
