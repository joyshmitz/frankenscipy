# frankenscipy-8l8r1.110 rejection: native sparse LU diagonal split

Target: repeated `splu_solve` on 2D Laplacian factors after the MMD clique-update rejection.

Lever tested: split the native sparse LU upper diagonal into a separate `u_diagonal: Vec<f64>` and remove diagonal entries from `u_rows`, so back-substitution can skip the `col == row` branch and use a direct pivot load.

Verdict: reject. The first same-worker candidate run on `vmi1149989` looked promising, but two subsequent same-worker candidate repeats did not reproduce a broad win. Only the largest MMD case stayed faster; RCM and smaller MMD rows regressed. Score = 0.0 because this is not a stable all-row improvement and does not clear the Score>=2.0 keep gate.

Same-worker baseline, `vmi1149989`, no diagonal split:

```text
solve x 200 k=32: rcm=10.9888 ms  mmd= 3.5748 ms
solve x 200 k=45: rcm=24.5236 ms  mmd=10.9054 ms
solve x 200 k=64: rcm=128.9744 ms mmd=56.8391 ms
```

Candidate sample 1, `vmi1149989`, diagonal split:

```text
solve x 200 k=32: rcm= 7.7339 ms  mmd= 3.0930 ms
solve x 200 k=45: rcm=22.5264 ms  mmd= 9.0832 ms
solve x 200 k=64: rcm=69.0046 ms  mmd=20.2083 ms
```

Candidate confirmation, `vmi1149989`, diagonal split:

```text
solve x 200 k=32: rcm=11.3347 ms  mmd= 5.8829 ms
solve x 200 k=45: rcm=34.0259 ms  mmd=13.3379 ms
solve x 200 k=64: rcm=145.7886 ms mmd=47.6106 ms
```

Candidate repeat 2, `vmi1149989`, diagonal split:

```text
solve x 200 k=32: rcm=11.9175 ms  mmd= 4.4224 ms
solve x 200 k=45: rcm=44.0296 ms  mmd=16.3696 ms
solve x 200 k=64: rcm=147.5594 ms mmd=40.6983 ms
```

Behavior and restore proof:

- Candidate kept solve-output agreement between RCM and MMD at the harness tolerance surface: `max|dx|` was unchanged at `4.89e-12`, `1.69e-11`, and `6.87e-11`.
- No source is retained.
- Restored source SHA-256:
  - `crates/fsci-sparse/src/linalg.rs`: `970c13d153021c9d3eba7ef5885a81e031bd37dd9ce2b798e84cacc5108b96e2`
  - `crates/fsci-sparse/src/bin/perf_spsolve.rs`: `ce130cb695866f270780aa05b07c61c09091eea2f0aa2cb1d0146d4d7e86364e`
- `git diff --exit-code -- crates/fsci-sparse/src/linalg.rs crates/fsci-sparse/src/bin/perf_spsolve.rs` passed after restore.

Reroute contract:

- Do not repeat diagonal extraction, sorted-vector pairwise insertion, allocation/scratch reuse, threshold tuning, HashSet/BTreeSet/sorted-Vec spelling, or batched Vec-merge clique construction.
- Next sparse pass must attack a different primitive: elimination-tree/quotient-graph symbolic state, supernodal sparse LU/Cholesky panelization, or a Laplacian-preconditioned route for structured solves.
