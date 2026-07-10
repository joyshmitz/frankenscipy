import subprocess, statistics as st, hashlib, platform
SP="/data/tmp/claude-1000/-data-projects-frankenscipy/3bf5db44-e1fc-4823-8afc-8c8873739c5d/scratchpad"
base, avx = f"{SP}/k_sse2", f"{SP}/k_avx2"
INNER="150000"; K=21
def once(binp,kern): return float(subprocess.run([binp,kern,INNER],capture_output=True,text=True).stdout.strip())
def sha(b): return hashlib.sha256(open(b,'rb').read()).hexdigest()[:16]
def med(v): return st.median(v)
print(f"# host={platform.node()}  base_sha={sha(base)} avx2_sha={sha(avx)}  K={K} inner={INNER}")
print(f"# {'kernel':20s} {'base us':>9s} {'avx2 us':>9s} {'CAND med':>9s} {'NULL med':>9s} {'null range':>18s}  verdict")
for kern,label in [("dot","simd_dot (trsm/gemm)"),("dot2","simd_dot2 (MR2 trsm)"),("dot4","simd_dot4 (syrk)"),("axpy","syrk_axpy (trailing)")]:
    for _ in range(3): once(base,kern); once(avx,kern)
    nr,cr=[],[]
    for _ in range(K):
        a=once(base,kern); c=once(avx,kern); a2=once(base,kern)
        nr.append(a/a2); cr.append(a/c)
    nlo,nhi=min(nr),max(nr); nm=med(nr); cm=med(cr)
    bm=med([once(base,kern) for _ in range(5)]); am=med([once(avx,kern) for _ in range(5)])
    verdict="DECIDED" if (cm>nhi or cm<nlo) else "IN-FLOOR"
    print(f"  {label:20s} {bm:9.4f} {am:9.4f} {cm:9.3f} {nm:9.3f}   [{nlo:.3f},{nhi:.3f}]  {verdict}")
