# Fuzz Seed Corpora

Per `/testing-fuzzing` hard rule 8 ("A corpus with 100K entries and the same
coverage as 500 entries is 200x slower, not 200x better") and the checklist
requirement ("Seed corpus created (empty + valid + boundary + adversarial,
min 5 entries)"), every fuzz target has a committed minimal seed corpus
here.

## Layout

```
fuzz/seeds/<target>/
    01_empty        — zero-byte input (exercises minimum-length guards)
    02_minimal      — single NUL byte (shortest non-empty decoding)
    03_typical      — 16 bytes of 0x00..0x0f (Arbitrary-friendly ascending)
    04_boundary     — 8 bytes of 0xff + 8 bytes of 0x00 (NaN / MAX vs 0.0)
    05_adversarial  — mixed 0xff/0x00/0x7f/0x80 bit-pattern stress
```

The corpus directory (`fuzz/corpus/`) is gitignored because libfuzzer
accumulates thousands of auto-mutated inputs there. Committed seeds live
under `fuzz/seeds/` and are copied into `fuzz/corpus/` before a campaign
via `scripts/prepare-fuzz-corpus.sh`.

## Usage

```bash
# Bootstrap corpus from committed seeds (idempotent, preserves auto-mutated entries).
scripts/prepare-fuzz-corpus.sh

# Run a target with the seeded corpus.
cd fuzz && cargo +nightly fuzz run p2c002_solve
```

## Adding or refining seeds

When a new bug is found, convert the minimized input into a permanent seed:

```bash
cargo +nightly fuzz tmin p2c002_solve fuzz/artifacts/p2c002_solve/crash-abc123
cp fuzz/corpus/p2c002_solve/<minimized-hash> fuzz/seeds/p2c002_solve/06_regression_abc123
git add fuzz/seeds/p2c002_solve/06_regression_abc123
```

This also satisfies hard rule 10 ("Every crash artifact becomes a regression
test or it WILL regress") — the seed is replayed on every campaign.

## Seed count per target

Minimum 5 per target, more for targets with known failure modes (sparse
format round-trip, special-function complex branches, stats fit degeneracies).
