# Performance Baselines

Benchmark snapshots captured via Criterion. Updated after each optimization PR.

## Usage

```bash
# Run all benchmarks
cargo bench

# Save a named baseline
cargo bench -- --save-baseline <name>

# Compare against a baseline
cargo bench -- --baseline <name>
```

## Files

- `baseline_initial.json` — First captured baseline for regression detection.
