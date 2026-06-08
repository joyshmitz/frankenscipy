# perf: parallelize hyp1f1 + hyperu hypergeometric broadcast loops

Bead: frankenscipy-q5qur

## Lever
`hyp1f1_dispatch` (confluent hypergeometric 1F1 / Kummer M — backbone of noncentral
distributions, Coulomb wavefunctions) and `hyperu_dispatch` (Tricomi confluent U) in
crates/fsci-special/src/hyper.rs evaluated their vector broadcast loops serially
(`for i in 0..out_len { results.push(scalar(...broadcast at i...)?) }`). Each output index is
an independent series/asymptotic evaluation. Routed both (hyp1f1: real + complex branches;
hyperu: real branch) through the `par_map_indices` helper already in hyper.rs.

## Isomorphism / byte-identity argument
- Each output index `i` reads inputs at broadcast index `i` and computes the same scalar,
  written to slot `i`; chunks cover `0..out_len` contiguously and concatenate in index order
  ⇒ identical output `Vec`. No reduction.
- Error path: chunk results folded with `?` in chunk (=index) order ⇒ first failing index's
  `SpecialError`, exactly as the serial loop's `?`.
- Empty / broadcast-shape / scalar paths untouched. Gate: serial for `< 64` elements.

⇒ The returned value (and first error) is bit-identical to the serial implementation.

## Proof (golden — serial vs parallel, identical)
Harness: `cargo run --profile release-perf -p fsci-special --bin perf_hyp1f1u_array`
(a=1.5, b=2.5; hyp1f1 z∈(-2,2); hyperu x∈(0.1,8.1))
```
n=100   hyp1f1=5f96a3a475cbb7ba hyperu=0708f14d2ee1e14f
n=5000  hyp1f1=27f45154d44f296e hyperu=cbe88dca4b60e2b5
n=50000 hyp1f1=3f59105913c4c2fa hyperu=45abfc7157b0e7cd
timing acc: hyp1f1 500k=104b290ae90ad955  hyperu 100k=a05cb718176a2361
```
Identical in the stashed serial build and the parallel build.
sha256(golden payload file) = cded531741806ac3599b2aa19f59949d2da9d5d1e1470307d44738de2ce87261

## Timing (rch remote, release-perf, 3 back-to-back runs each)
| function / array | serial (3x)          | new (3x)             | speedup |
|------------------|----------------------|----------------------|---------|
| hyp1f1, 500k     | 31.90/37.11/31.88 ms | 6.84/7.71/7.86 ms    | ~4.3x   |
| hyperu, 100k     | 7.078/7.280/7.172 s  | 277.2/272.9/276.3 ms | ~26.1x  |

(hyperu's Tricomi U is ~72µs/element serial — hence the large ratio.)

## Validation
26 hyp1f1 unit tests pass; clippy: no warning in hyper.rs. With this, all the clean
broadcast-loop hyper dispatchers (hyp2f1, hyp1f1, hyperu) are parallel.
