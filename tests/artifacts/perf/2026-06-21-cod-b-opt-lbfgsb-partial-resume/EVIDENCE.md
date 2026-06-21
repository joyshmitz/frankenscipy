# 2026-06-21 cod-b opt L-BFGS-B partial-resume evidence

Bead: `frankenscipy-8l8r1.142`
Agent: cod-b / BlackThrush

Scope: one small per-crate `fsci-opt` benchmark row against a local SciPy
oracle. No library optimization was attempted in this turn. The only source
changes after measurement are gate cleanups in benchmark/helper-bin code:
`optimize_bench.rs` removes one clippy needless borrow and applies rustfmt line
wraps; `diff_leastsq.rs` applies rustfmt line wraps.

## Rust Criterion row

Command:

```bash
AGENT_NAME=BlackThrush RCH_REQUIRE_REMOTE=1 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
rch exec -- cargo bench -p fsci-opt --bench optimize_bench -- \
  lbfgsb/rosenbrock_unconstrained_fd/10 --noplot \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Worker: `vmi1152480`

Output:

```text
lbfgsb/rosenbrock_unconstrained_fd/10
                        time:   [125.99 us 134.04 us 142.95 us]
```

Criterion estimate file:

- slope point estimate: `134040.25587874596 ns` = `134.040 us`
- median point estimate: `138154.23644067795 ns` = `138.154 us`

## SciPy oracle

Command shape: local `scipy.optimize.minimize` on the same 10D Rosenbrock start
`[-1.2, 1.0, ...]`, `method="L-BFGS-B"`, no analytic Jacobian, `tol=1e-8`,
`options={"maxiter": 2000, "gtol": 1e-8, "eps": 1e-8}`.

Environment:

- SciPy `1.17.1`
- NumPy `2.4.3`

Timing:

- samples: `200`
- warmup: `5`
- median: `16537.314 us`
- p95: `19623.567 us`
- min/max: `15937.588 us` / `23228.543 us`
- result: `success=True`, `status=0`, `nit=71`, `nfev=913`,
  `fun=1.0899951963071803e-10`

## Ratio and decision

| Workload | Rust Criterion | SciPy oracle | Ratio vs SciPy | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | `134.040 us` | `16537.314 us` | Rust `123.38x` faster (`0.008105x` SciPy time) | measured win |

Decision: KEEP as measured evidence for the current end-to-end L-BFGS-B
finite-difference route. There was no performance source patch in this turn and
therefore no near-zero-gain optimization to revert.

Negative routing: do not spend the next turn on this end-to-end 10D L-BFGS-B
row unless a new conformance issue appears. The cod-a public finite-difference
helper bead `frankenscipy-8l8r1.141` has separate helper-only evidence; this
end-to-end L-BFGS-B row does not supersede that score.

## Gates

- PASS: focused opt tests via rch:
  `cargo test -p fsci-opt lbfgsb --lib -- --nocapture` = 8 passed.
- PASS: live SciPy conformance locally with required oracle:
  `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_lbfgsb_minimize -- --nocapture`
  = 1 passed.
- PASS: per-crate compile via rch:
  `cargo check -p fsci-opt --all-targets`.
- PASS: per-crate clippy via rch after the benchmark-file needless-borrow fix:
  `cargo clippy -p fsci-opt --all-targets --no-deps -- -D warnings`.
- PASS: `cargo fmt --check -p fsci-opt`.
- PASS: `git diff --check`.
- PASS/WARN: changed-file `ubs` exited 0 with 0 critical issues. It reported
  warning inventory in existing benchmark/helper-bin code, including `expect`
  in benchmark closures and direct indexing in `diff_leastsq.rs`.
