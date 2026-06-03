# Conclusion: Rejected

The profile-backed target was `baseline_pinv/1000x500`, selected after the post-packed-B reprofile ranked it first among the scoped linalg benchmarks.

The one-lever trial replaced dense diagonal pseudoinverse materialization with direct scaling of `V` columns before the final multiply by `U^T`. Golden output was unchanged: the normalized RCH release `pinv` test output SHA-256 stayed `271c9ee685150a31f31ca47867f9b2264eaa254542b1ea49907242bb895bc1cc`.

The focused RCH Criterion benchmark failed the keep gate:

| benchmark | baseline median | after median |
| --- | ---: | ---: |
| `baseline_pinv/1000x500` | `316.20 ms` | `1.0569 s` |

Decision: reject and restore source. The source tree has no remaining diff for `crates/fsci-linalg/src/lib.rs`; only the rejection evidence is retained.
