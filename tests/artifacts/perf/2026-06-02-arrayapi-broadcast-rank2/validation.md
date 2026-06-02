# Validation

## Passed

Focused remote test:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-arrayapi rch exec -- cargo test -p fsci-arrayapi broadcast_to_rank2_singletons_preserves_row_major_values --lib --locked -- --nocapture
```

Result: 1 passed.

Full remote fsci-arrayapi lib tests:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-arrayapi rch exec -- cargo test -p fsci-arrayapi --lib --locked -- --nocapture
```

Result: 54 passed, 0 failed.

Remote clippy:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-arrayapi rch exec -- cargo clippy -p fsci-arrayapi --all-targets --locked -- -D warnings
```

Result: passed.

UBS:

```text
ubs crates/fsci-arrayapi/src/backend.rs
```

Result: exit 0, no critical issues for the changed file.

## Caveat

Package fmt check:

```text
cargo fmt -p fsci-arrayapi --check
```

Result: failed on existing formatting drift in crates/fsci-arrayapi/src/creation.rs:333, outside this bead's edit surface. The changed file was not reformatted beyond the local patch.
