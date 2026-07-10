# Median-null-gated A/B of two ISA builds of the SAME source, on this AVX2 box.
# Arms are two binaries; interleave base/avx2/base per iteration on one machine.
# null = base_i / base2_i (identical binary) -> must centre on 1.000; its range is the floor.
import subprocess, statistics as st, hashlib, platform, sys
SP="/data/tmp/claude-1000/-data-projects-frankenscipy/3bf5db44-e1fc-4823-8afc-8c8873739c5d/scratchpad"
ITERS_INNER = "150000"
def once(binp): return float(subprocess.run([binp, ITERS_INNER], capture_output=True, text=True).stdout.strip())
def sha(binp): return hashlib.sha256(open(binp,'rb').read()).hexdigest()[:16]
base, avx = f"{SP}/bench_sse2", f"{SP}/bench_avx2"
for b in (base, avx):   # warm
    for _ in range(3): once(b)
null_r, ab_r = [], []
K = 21
for _ in range(K):
    a = once(base); c = once(avx); a2 = once(base)   # interleaved, per-iteration paired
    null_r.append(a/a2)     # A/A: identical binary
    ab_r.append(a/c)        # base vs avx2
def med(v): return st.median(v)
nlo, nhi, nmed = min(null_r), max(null_r), med(null_r)
cmed = med(ab_r)
decidable = cmed > nhi or cmed < nlo
print(f"# host={platform.node()}  base_sha256={sha(base)}  avx2_sha256={sha(avx)}  K={K} inner={ITERS_INNER}")
print(f"# base(sse2) median {med([once(base) for _ in range(5)]):.4f} us/call  avx2 median {med([once(avx) for _ in range(5)]):.4f} us/call")
print(f"NULL(A/A base) median {nmed:.3f}x  range [{nlo:.3f}, {nhi:.3f}]")
print(f"CAND(sse2/avx2) median {cmed:.3f}x  {'DECIDED' if decidable else 'IN-FLOOR'}  "
      f"(candidate median {'outside' if decidable else 'inside'} null range)")
print(f"# interpretation: {cmed:.2f}x means the AVX2 build runs the SAME source {cmed:.2f}x faster, bit-identically")
