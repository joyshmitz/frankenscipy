frankenscipy-8l8r1.48 rejected lever: parallel left-column Householder reduction

Target
- Profile-backed hotspot: `bidiag_large_reduction_perf_probe`, shape `1024x512`.
- Current-main RCH baseline: `elapsed_ms=521.955192`, digest `0x90cdd3f8f71ed2c1`, worker `vmi1264463`.
- Lever tested: split the Golub-Kahan left Householder trailing-column update across safe `std::thread::scope` column chunks, preserving each column's dot/update order.

Behavior proof
- RCH proof artifact: `proof_parallel_left_reduction_bits_rch.txt`.
- Remote worker: `vmi1156319`.
- Forced serial reducer versus forced 4-worker reducer passed bit-identical reduction checks for `256x160` and `384x192`.
- Digest policy preserved during the benchmark: serial digest `0x90cdd3f8f71ed2c1`, parallel digest `0x90cdd3f8f71ed2c1`.
- Floating-point order proof: each output column kept the exact same reflector-value iteration order for dot products and updates; only independent columns were partitioned. No tie-breaking, sign-choice, rank, certificate, error policy, or RNG surface changed.

Benchmark result
- RCH benchmark artifact: `after_parallel_left_reduction_rch.txt`.
- Remote worker: `vmi1293453`.
- Same-binary timing:
  - `serial_ms=180.445523`
  - `parallel_ms=463.392683`
  - `speedup=0.389401`
  - `worker_count=8`

Decision
- Rejected. The candidate preserved behavior but slowed the profiled reduction stage.
- Score: Impact `0.389401` x Confidence `5` / Effort `2` = `0.973503`, below the required `2.0`.
- Source was restored. Next direction should be a true communication-avoiding/two-stage bidiagonalization or packed DLABRD-style panel algorithm, not another independent-column scheduling tweak.
