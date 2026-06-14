# Post-105 fsci-opt reprofile route

## Context

This is the required post-close reprofile after `frankenscipy-8l8r1.105`.

The RCH run was crate-scoped:

```text
rch exec -- cargo bench -p fsci-opt --bench optimize_bench --locked -- --sample-size 10
```

Artifact: `current_optimize_bench_rch.txt`

## Result

- Worker: `vmi1149989`
- Status: partial routing evidence. The broad optimize bench was SIGKILLed at `cg/quadratic/10`, after completing BFGS rows and CG Rosenbrock rows.
- Largest completed row after the fused CG evaluator keep:
  - `cg/rosenbrock/10`: `[260.81 us 270.39 us 282.03 us]`
- Next completed rows:
  - `cg/rosenbrock/5`: `[99.202 us 106.69 us 113.97 us]`
  - `bfgs/rosenbrock/10`: `[72.770 us 76.913 us 84.025 us]`

## Routing Decision

No ready or open `[perf]` bead existed after `frankenscipy-8l8r1.105` closed. I filed `frankenscipy-8l8r1.106` from this reprofile.

The next pass must not repeat accepted-gradient carry or scratch-only workspace micro-levers. First step is a focused same-worker baseline plus a stage/profile probe that separates:

- objective-call count
- finite-difference stencil cost
- line-search and zoom cost
- accepted-point materialization
- allocation traffic

Any source lever must preserve Wolfe ordering, finite-difference component order, evaluation counters, FP bits, and RNG absence, with golden SHA-256 proof and a same-worker keep gate.
