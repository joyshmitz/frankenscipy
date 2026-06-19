# frankenscipy-8l8r1.118 coherence gauntlet

Date: 2026-06-19
Agent: cod-b / MistyBirch
Decision: KEEP

## Lever

`coherence` stays on the fused Welch segment pass that accumulates `Pxy`,
`Pxx`, and `Pyy` together instead of composing three independent `csd` calls.
This gauntlet adds a durable Criterion group that measures:

- Rust fused coherence.
- Rust compositional `csd(x,y)` + `csd(x,x)` + `csd(y,y)`.
- Original `scipy.signal.coherence` when SciPy is importable.

## Commands

Remote build/check/test:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-signal --all-targets
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-signal coherence_matches --lib -- --nocapture
```

Remote Rust bench:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-signal --bench signal_bench coherence_gauntlet_scipy -- --noplot
```

Local same-host SciPy oracle bench, because rch worker `hz1` could not import
`scipy.signal`:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-signal --bench signal_bench coherence_gauntlet_scipy -- --noplot
```

## Results

Criterion mean point estimates from
`/data/projects/.rch-targets/frankenscipy-cod-b/criterion/coherence_gauntlet_scipy/*/new/estimates.json`:

| Workload | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence`, 65536 samples, window 1024, overlap 512 | 2.191980 ms | 8.65x faster than SciPy | win |
| Rust compositional triple-CSD route | 6.536569 ms | fused is 2.98x faster | internal win |
| SciPy `scipy.signal.coherence` | 18.961613 ms | 1.00x oracle | reference |

The remote rch Rust-only bench on worker `hz1` kept the same shape before the
SciPy row skipped:

| Workload | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence` | 4.3780 ms | 2.80x faster than compositional | internal win |
| Rust compositional triple-CSD route | 12.269 ms | 1.00x internal baseline | slower |

Win/loss/neutral summary for this scoped gauntlet: Rust fused coherence vs
SciPy is 1 win / 0 losses / 0 neutral; Rust fused coherence vs triple-CSD
composition is 1 win / 0 losses / 0 neutral.

## Guardrails

- PASS: `cargo check -p fsci-signal --all-targets` through rch worker `hz2`.
- PASS: focused coherence correctness tests through rch worker `hz2`:
  `coherence_matches_scipy_reference` and
  `coherence_matches_compositional_csd_formula`.
- PASS: changed benchmark file formatting via
  `rustfmt --edition 2024 --check crates/fsci-signal/benches/signal_bench.rs`.
- No revert. The fused route is a head-to-head SciPy win on the scoped
  coherence workload and a same-shape internal win against the triple-CSD
  composition it replaced.
