# fsci-arrayapi promote_and_broadcast closeout

Bead: `frankenscipy-rr72u`

## Profile target

Fresh broad RCH Criterion profile:

- Artifact: `criterion_broad_rch.txt`
- Top rows:
  - `arrayapi_broadcast/promote_and_broadcast/10000`: `244.04 us`
  - `arrayapi_dtype_cast/astype_float32_to_complex128/10000`: `144.29 us`
  - `arrayapi_indexing/getitem_basic/10000`: `82.796 us`

Focused top-row baseline on `vmi1149989`:

- Artifact: `baseline_promote_broadcast_10000_rch.txt`
- `arrayapi_broadcast/promote_and_broadcast/10000`: `[311.19 us, 317.18 us, 323.82 us]`

Focused cast baseline on `vmi1149989`:

- Artifact: `baseline_astype_f32_complex128_10000_rch.txt`
- `arrayapi_dtype_cast/astype_float32_to_complex128/10000`: `[71.275 us, 71.715 us, 72.175 us]`

## Kept lever

`CoreArrayBackend::astype` now routes through `cast_values_to_dtype`, which specializes
`Float32|Float64 -> Complex128` casts by lifting the target-dtype branch out of the
per-element generic scalar conversion loop.

Score: `9.0 = impact 3.0 * confidence 4.5 / effort 1.5`.

Rejected lever: fused `astype` plus `broadcast_to` backend hook. It preserved the golden
output but regressed `promote_and_broadcast/10000` to `595.68 us` median, so it was removed.

## Performance result

Focused cast row:

- Before: `71.715 us` median
- After: `39.547 us` median
- Delta: `-44.9%`

Focused top row:

- Before: `317.18 us` median
- After: `244.17 us` median
- Delta: `-23.0%`

Post-change broad re-profile on `vmi1149989`:

- Artifact: `reprofile_after_astype_broad_rch.txt`
- Shifted top rows:
  - `arrayapi_indexing/getitem_basic/10000`: `67.586 us`
  - `arrayapi_broadcast/promote_and_broadcast/10000`: `63.534 us`
  - `arrayapi_dtype_cast/astype_float32_to_complex128/10000`: `12.763 us`

## Isomorphism proof

- Ordering preserved: yes. `astype` still iterates input values in slice order and emits one output per input value.
- Tie-breaking unchanged: N/A. There is no comparator, selection, or rank tie path.
- Floating-point preserved: yes. The specialized path uses the already stored real `f64` payload as the complex real part with `im = 0.0`, matching the old `scalar_to_complex_components` branch for `ScalarValue::F64`.
- RNG seeds unchanged: N/A. The array API cast and broadcast paths do not use RNG.
- Golden output: byte-identical before/after.
  - `golden_before.txt` sha256: `6ec9625d70f75ccde2bb8bce3d05f1a64b9e445fb0496139d7fe6a52cffa52c8`
  - `golden_after_astype.txt` matched `golden_before.txt` by `cmp -s`.

## Validation

- RCH `cargo test -p fsci-arrayapi --lib --locked`: `55 passed`
- RCH `cargo clippy -p fsci-arrayapi --all-targets --locked -- -D warnings`: passed
- `cargo fmt -p fsci-arrayapi --check`: passed
- `ubs crates/fsci-arrayapi/src/backend.rs crates/fsci-arrayapi/src/broadcast.rs`: exit `0`, no critical findings
