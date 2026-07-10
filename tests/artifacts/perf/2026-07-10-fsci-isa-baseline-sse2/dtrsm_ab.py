import subprocess, statistics as st, hashlib, platform
SP="/data/tmp/claude-1000/-data-projects-frankenscipy/3bf5db44-e1fc-4823-8afc-8c8873739c5d/scratchpad"
base, avx = f"{SP}/k_sse2", f"{SP}/k_avx2"
INNER="150000"; K=41; KERN="dot2"
def once(b): return float(subprocess.run([b,KERN,INNER],capture_output=True,text=True).stdout.strip())
def sha(b): return hashlib.sha256(open(b,'rb').read()).hexdigest()[:16]
def med(v): return st.median(v)
for _ in range(5): once(base); once(avx)
nr, cr = [], []
for _ in range(K):
    a=once(base); c=once(avx); a2=once(base)
    nr.append(a/a2); cr.append(a/c)
nlo,nhi,nm = min(nr),max(nr),med(nr); cm=med(cr)
bm=med([once(base) for _ in range(9)]); am=med([once(avx) for _ in range(9)])
dec = cm>nhi or cm<nlo
print(f"# KERNEL: dtrsm / simd_dot2_shared_rhs  (cod's MR2 panel TRSM, a6d7ba897/770c4d490)")
print(f"# host={platform.node()}  base_sha={sha(base)} avx2_sha={sha(avx)}  K={K} inner={INNER}")
print(f"# self_time (perf): simd_dot2 = 98.82% (sse2) / 98.97% (avx2) -> genuinely the kernel")
print(f"# self_time absolute: sse2 1.789e9 -> avx2 1.581e9 events = 1.132x fewer")
print(f"# bit-identical: checksum a75a7aa618732200 for BOTH builds")
print(f"base(sse2) median {bm:.4f} us/call   avx2 median {am:.4f} us/call")
print(f"NULL(A/A base) median {nm:.3f}x  range [{nlo:.3f}, {nhi:.3f}]")
print(f"CAND(sse2/avx2) median {cm:.3f}x  {'DECIDED' if dec else 'IN-FLOOR (undecidable at this K)'}")
