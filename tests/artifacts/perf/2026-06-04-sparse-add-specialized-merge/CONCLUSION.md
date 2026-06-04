# CSR Add Specialized Merge Trial

Bead: `frankenscipy-odk6z`

Verdict: rejected; source restored.

## Baseline

Command:

```bash
RCH_WORKER=ts2 rch exec -- cargo bench -p fsci-sparse --bench sparse_bench --locked sparse_arithmetic/10000x10000_d0_add/10000 -- --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Result: `sparse_arithmetic/10000x10000_d0_add/10000` median `2.1090 ms` on RCH worker `ts2`.

## Lever

Split the metadata-canonical direct CSR add path from the generic `rhs_scale` merge so RHS-only and matched RHS entries avoid multiplying by `1.0`.

Isomorphism contract:

- CSR row order unchanged.
- Column-order merge and equal-column tie behavior unchanged.
- Overlap floating-point order unchanged: start at `0.0`, add lhs, add rhs.
- Zero elision unchanged.
- Canonical metadata tracking unchanged.
- No RNG or global state involved.

## Proof

Before and after RCH `perf_sparse add-csr-golden` payload SHA-256:

```text
a3a9d49d373b8d28f1aca881ab2b2322229b8befc0cb91c1e85a0820bc318da8
```

## Rebench

Same command and worker after the trial:

```text
sparse_arithmetic/10000x10000_d0_add/10000
                        time:   [2.0748 ms 2.0792 ms 2.0835 ms]
```

Delta: `2.1090 ms -> 2.0792 ms`, about `1.014x`.

Score: `1.5 = impact 1 * confidence 3 / effort 2`.

## Restore

The source lever was removed because the score was below the `2.0` keep gate.

Verification:

```text
git diff --quiet -- crates/fsci-sparse/src/ops.rs => 0
cargo fmt -p fsci-sparse --check => 0
```
