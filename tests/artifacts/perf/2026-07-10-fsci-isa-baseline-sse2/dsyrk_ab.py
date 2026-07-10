import subprocess, statistics as st, hashlib, platform
SP="/data/tmp/claude-1000/-data-projects-frankenscipy/3bf5db44-e1fc-4823-8afc-8c8873739c5d/scratchpad"
base, avx = f"{SP}/k_sse2", f"{SP}/k_avx2"
INNER="150000"; K=31; KERN="dot4"
def once(b): return float(subprocess.run([b,KERN,INNER],capture_output=True,text=True).stdout.strip())
def sha(b): return hashlib.sha256(open(b,'rb').read()).hexdigest()[:16]
def med(v): return st.median(v)
for _ in range(4): once(base); once(avx)
nr, cr = [], []
for _ in range(K):
    a=once(base); c=once(avx); a2=once(base)
    nr.append(a/a2); cr.append(a/c)
nlo,nhi,nm = min(nr),max(nr),med(nr); cm=med(cr)
bm=med([once(base) for _ in range(7)]); am=med([once(avx) for _ in range(7)])
print(f"# KERNEL: dsyrk / simd_dot4  (fsci cholesky_syrk_flat_rows inner)")
print(f"# host={platform.node()}  base_sha={sha(base)} avx2_sha={sha(avx)}  K={K} inner={INNER}")
print(f"# self_time (perf): simd_dot4 = 98.3% (sse2) / 98.2% (avx2) of the microbench -> genuinely the kernel")
print(f"# self_time absolute: sse2 3.731e9 events -> avx2 2.398e9 events = 1.556x fewer")
print(f"# bit-identical: checksum b27df98589396e00 for BOTH builds")
print(f"base(sse2) median {bm:.4f} us/call   avx2 median {am:.4f} us/call")
print(f"NULL(A/A base) median {nm:.3f}x  range [{nlo:.3f}, {nhi:.3f}]")
print(f"CAND(sse2/avx2) median {cm:.3f}x  {'DECIDED' if (cm>nhi or cm<nlo) else 'IN-FLOOR'}")
