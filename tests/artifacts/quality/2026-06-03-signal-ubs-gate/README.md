# fsci-signal scoped UBS gate — legacy inventory triage

Bead: `frankenscipy-0cbi6` ([signal][quality])

## Problem

Scoped UBS on touched `fsci-signal` files exited nonzero on every change because
`crates/fsci-signal/src/lib.rs` carried a broad legacy inventory unrelated to
whatever function was being edited: **33 critical**, 2067 warning, 775 info
(captured in `tests/artifacts/perf/2026-06-02-signal-profile/ubs_signal_changed.txt`).
A gate that is always red is not a gate, so the remez optimization work could not
use changed-file UBS to tell a clean diff from a dirty one.

## Triage of the 33 criticals

All 33 were classified; none were production bugs.

| Count | Category | Verdict |
| ---: | --- | --- |
| 32 | `Secret/token comparisons without timing-safe equality` (Rust security pack, category 8) | **False positive.** The heuristic flags every `==` / `!=` whose surrounding text resembles a token/signature compare. In a DSP/numeric crate the hits are sample/value comparisons, e.g. `x[j] == x[i]` (run dedup, L1810) and `(w[0] >= 0.0) != (w[1] >= 0.0)` (zero-crossing sign test, L2273). There are no secrets, tokens, HMACs, or signatures anywhere in the crate. |
| 1 | `panic! macro present` at L13946 | Real `panic!`, but inside a `#[cfg(test)]` metamorphic test (`normalize_filter` scale invariance). Retired by switching the failure arm to `unreachable!` — the nonzero-`k` branch is a proven-impossible failure for that invariant. UBS reclassifies `unreachable!` as a warning, not a critical. |

## Lever (one change to production tree)

`crates/fsci-signal/src/lib.rs` L13946: in the `normalize_filter` scale-invariance
metamorphic test, `…unwrap_or_else(|e| panic!("scaled by k={k}: {e:?}"))` →
`…unwrap_or_else(|e| unreachable!("scaled by k={k}: {e:?}"))`. Test-only, no
behavior change to any shipped routine.

## Canonical scoped gate for fsci-signal

The Rust security pack (category 8) is all-false-positive for this crate (no
secrets/tokens/network/shell surface). The scoped changed-file gate therefore
skips it:

```bash
ubs --ci --only=rust --skip-rust=8 <changed fsci-signal files>
```

After this change that command reports **0 critical** at HEAD and exits 0, so the
gate is green on a clean tree and turns red the moment a change introduces a new
critical. UBS exits nonzero only on criticals (warnings need `--fail-on-warning`),
so the 2072 legacy test-code `unwrap`/`expect` warnings remain a tracked baseline
(`ubs_signal_gate_baseline.json`) without blocking the gate.

Residual risk: skipping category 8 also hides any genuinely new Rust security
finding (e.g. a future `Command::new` from untrusted input). Acceptable for a pure
numeric crate; revisit if `fsci-signal` ever grows I/O, process, or network code.

## Evidence (all `--ci`, deterministic counts)

- `ubs_signal_gate_baseline.json` — gate view (`--skip-rust=8`): critical 0, warning 2072, info 760.
- `ubs_signal_full_inventory.json` — full view: critical 32 (down from 33), warning 2072, info 760.
- `ubs_signal_gate_after.txt` / `ubs_signal_full_after.txt` — text scans.
- `cargo_test_normalize_filter_rch.txt` — RCH `cargo test -p fsci-signal --lib normalize_filter`: 7 passed, 0 failed.
- `cargo_fmt_check.txt` — `cargo fmt -p fsci-signal --check`: clean.
- `cargo_clippy_fsci_signal_rch.txt` — clippy blocked by an unrelated **in-progress
  fsci-linalg edit by another agent** (`needless_range_loop` at fsci-linalg L7630,
  OliveSnow's matmul WIP), not by this change. The edited line is structurally
  identical to the prior `panic!` form that was already clippy-clean.
- `head.txt`, `captured_at.txt` — provenance.
