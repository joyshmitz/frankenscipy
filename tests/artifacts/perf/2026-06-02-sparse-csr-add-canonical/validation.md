# Validation

## Passed

Focused remote test:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-sparse rch exec -- cargo test -p fsci-sparse add_csr_direct_canonical_merge_preserves_sorted_rows_and_zero_elision --lib --locked -- --nocapture
```

Result: 1 passed, 303 filtered out.

Full remote fsci-sparse lib tests:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-sparse rch exec -- cargo test -p fsci-sparse --lib --locked -- --nocapture
```

Result: 304 passed, 0 failed.

Scoped format check:

```text
rustfmt --edition 2024 --check crates/fsci-sparse/src/ops.rs
```

Result: passed.

Adjusted remote clippy:

```text
RCH_FORCE_REMOTE=1 CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenscipy-olivesnow-sparse rch exec -- cargo clippy -p fsci-sparse --all-targets --locked -- -D warnings -A non_camel_case_types
```

Result: passed.

UBS:

```text
ubs crates/fsci-sparse/src/ops.rs
```

Result: exit 0, no critical issues for the changed file.

## Caveat

Strict remote clippy with `-D warnings` failed on pre-existing SciPy-compatible lowercase alias types in crates/fsci-sparse/src/lib.rs:23-37. The adjusted clippy run allowed only `non_camel_case_types` and denied all other warnings.
