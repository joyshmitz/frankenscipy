# Argsort Non-NaN Fast Path Negative Result

Bead: `frankenscipy-8l8r1.8`

## Candidate

Tested a one-lever fast path for `argsort` when inputs contain no `NaN`: use
`sort_unstable_by` with an explicit index tie-break to preserve stable ordering
for equal values and signed-zero ties. Inputs containing `NaN` kept the prior
stable `sort_by` path to preserve existing `partial_cmp(...).unwrap_or(Equal)`
behavior.

## Behavior Proof

- Golden output before/after was byte-identical.
- Golden SHA stayed
  `0a5fa6391a2e9e64825488fc3ca4d991535a281f552e06ce1ca9813188a5ea31`.
- Same-binary A/B checksum matched in both replay runs.
- Test coverage passed for value ordering, stable ties, signed zero, and the
  existing `NaN` ordering behavior.

## Performance Evidence

Focused Criterion target: `ordering_and_bins/argsort/4096`.

| Run | Worker | Interval | Median |
| --- | --- | --- | --- |
| Baseline | `vmi1227854` | `[109.03 us, 110.66 us]` | `109.97 us` |
| First after | `vmi1293453` | `[52.682 us, 60.705 us]` | `56.981 us` |
| Repeat after | `vmi1156319` | `[224.71 us, 242.46 us]` | `231.61 us` |

Same-binary A/B checks:

| Run | Worker | Stable | Fast | Verdict |
| --- | --- | --- | --- | --- |
| `argsort-ab 2000` | `vmi1293453` | `122.648832 us` | `60.550539 us` | win |
| `argsort-ab 2001` | `vmi1153651` | `179.390537 us` | `287.844860 us` | regression |

The repeat Criterion result and second A/B replay do not prove a stable real
win. Per the campaign rule, the lever does not clear the keep gate.

## Validation Artifacts

- RCH focused argsort tests: passed.
- RCH `cargo check -p fsci-stats --all-targets --locked`: passed.
- RCH `cargo clippy -p fsci-stats --all-targets --locked -- -D warnings`:
  passed.
- `cargo fmt -p fsci-stats --check`: passed.
- Scoped UBS: exit 0, critical 0.

## Outcome

Score: `0.0` because the final performance evidence is negative.

Verdict: abandoned. No source change kept; `crates/fsci-stats/src/lib.rs` and
`crates/fsci-stats/src/bin/perf_stats.rs` match HEAD.
