# frankenscipy-8l8r1.117 sparse random rounded-empty gauntlet

Date: 2026-06-19
Agent: cod-b / MistyBirch
Decision: KEEP

## Lever

`fsci_sparse::random` now computes the rounded output cardinality before any
grid scan. If `round(density * rows * cols) == 0`, it returns an empty COO
matrix immediately after validating shape overflow. This matches SciPy's
rounded-size sparse random contract for sub-half expected nnz inputs and turns
the pathological huge-grid case into an output-sensitive constant-time path.

Implementation commit already present on `main` before this closeout:
`f037b1da frankenscipy-8l8r1.117 perf: rounded-empty sparse random, code-first
batch-test pending`.

## Commands

Remote Rust bench:

```bash
RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-sparse --bench sparse_bench -- \
  sparse_random_tiny_density --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Same-worker SciPy oracle:

```bash
RCH_WORKER=hz1 rch exec -- python3 -c '... scipy.sparse.random(..., format="coo", rng=reused_rng) ...'
```

Correctness guard:

```bash
RCH_WORKER=hz1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-sparse random_ -- --nocapture
```

## Results

RCH worker: `hz1`
SciPy: `1.17.1`
NumPy: `2.4.3`

| Workload | Rust Criterion point | SciPy median | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `random(1e9 x 1e9, density=1e-19)` | 39.405 ns | 46,453 ns | 1,179x faster | win |
| `random(2e9 x 2e9, density=1e-20)` | 39.408 ns | 55,310 ns | 1,403x faster | win |

SciPy p95 rows: 57,268 ns and 80,022 ns. Both SciPy and Rust produced
`nnz == 0` for the two huge rounded-empty cases.

Scorecard:

| Class | Count |
| --- | ---: |
| Win vs SciPy | 2 |
| Loss vs SciPy | 0 |
| Neutral vs SciPy | 0 |

## Guardrails

- PASS: `cargo test -p fsci-sparse random_ -- --nocapture` on `hz1`
  (`10 passed; 0 failed; 338 filtered out`).
- PASS: Criterion sparse tiny-density bench on `hz1`.
- No code change in this closeout. The measured code was already present in
  `f037b1da`.

## Negative Evidence

Do not retry dense Bernoulli scanning for rounded-empty sparse random. The
cardinality-first route is both SciPy-compatible and orders of magnitude faster
on the huge sub-half-expected-nnz case. Future random-constructor work should
target non-empty sparse sampling collisions, non-COO format construction, or
SciPy parity gaps outside this rounded-empty regime.
