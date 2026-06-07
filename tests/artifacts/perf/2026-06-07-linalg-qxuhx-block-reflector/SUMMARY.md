# Keep: Parallel Left-Reflector Replay

Bead: `frankenscipy-qxuhx`

## Target

Continue the linalg no-gaps bidiagonal SVD campaign after exact-order replay
schedule changes, indexing cleanup, workspace lifetime, transposed right replay,
and compact-WY block reflector attempts failed to produce a Score `>= 2.0` win.

Fresh qxuhx baseline:

```text
worker=ts1
probe=thin_bidiag_factor_replay_perf_probe
shape=1024x512
reference_ms=489.466630
replay_ms=240.150780
reduction_digest=0x90cdd3f8f71ed2c1
replay_digest=0x8f521a39638fb520
```

## Lever

Replay the left Householder reflectors over disjoint column chunks of the thin
`U` factor with scoped worker threads. Each column still receives the same
reflector order, dot-product order, and update order as serial replay; columns
are independent, so chunk ownership does not change floating-point results.

The right reflector replay, sign canonicalization, singular-value ordering,
rank thresholds, certificate policy, public errors, and RNG behavior are
unchanged.

## Proof

RCH focused bit proof passed:

```text
test tests::thin_bidiag_parallel_left_replay_matches_serial_bits ... ok
```

The proof compares singular values, `U`, and `Vt` entries by `f64::to_bits`
for shapes that cross the parallel threshold.

Public SVD/lstsq/pinv golden payload SHA stayed unchanged:

```text
1cdd3658c6caef8dec9fc58fa7e12b8d5c90151e2f93df91ffe2fcf862c16225
```

## Rebench

RCH same-binary A/B on `ts1`:

```text
worker_count=64
serial_ms=249.231768
parallel_ms=73.028977
speedup=3.412779
digest=0x8f521a39638fb520
```

## Validation

Passed:

- `cargo fmt -p fsci-linalg --check`
- `git diff --check -- crates/fsci-linalg/src/lib.rs`
- RCH `cargo check -p fsci-linalg --all-targets --locked`
- RCH `cargo clippy -p fsci-linalg --all-targets --no-deps --locked -- -D warnings`
- `ubs crates/fsci-linalg/src/lib.rs`

UBS reported zero critical issues and the pre-existing broad linalg warning
inventory.

## Decision

Keep.

Score: `8.5 = Impact 3.4 * Confidence 5 / Effort 2`.

Next primitive remains the deeper bidiagonal-specialized SVD backend tracked by
`frankenscipy-8l8r1.47`.
