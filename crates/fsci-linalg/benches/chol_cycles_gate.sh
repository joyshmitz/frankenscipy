#!/usr/bin/env bash
# Cholesky-wall CYCLES self-time gate (frankenscipy-64wo0; prior art d97283534).
#
# Wraps `bin/perf_chol_cycles_gate` in `perf stat -e cycles` and runs an
# interleaved base/cand A/B plus an A/A (base-vs-base) null. CPU cycles count
# RETIRED work across all threads, immune to OS scheduling, so the null floor is
# ~±2-4% vs wall-clock's ~±5% at n=1000 — decidable for the sub-floor kernel /
# data-movement levers (pack fusion, scratch reuse) that wall-clock cannot resolve.
#
# Usage:  bash crates/fsci-linalg/benches/chol_cycles_gate.sh
#   env:  N (default 1000)  REPS (16)  K (21 interleaved rounds)
# Run under rch so the build offloads and perf runs on the same worker:
#   rch exec -- bash crates/fsci-linalg/benches/chol_cycles_gate.sh
#
# The default arms isolate the trailing-SYRK kernel (both share TRSM_ROWS2 +
# chol_nb): base = mul+add MR4xNR8, cand = fused mul_add MR4xNR8. This reproduces
# the FMA-SYRK lever (a VALIDATION of the gate against its known verdict). To gate
# a NEW lever, point the two arms in perf_chol_cycles_gate.rs at the new baseline
# / candidate factor fns and re-run.
set -euo pipefail

N="${N:-1000}"; REPS="${REPS:-16}"; K="${K:-21}"

BIN=$(cargo build --release -p fsci-linalg --bin perf_chol_cycles_gate \
        --features chol-wall-bench --message-format=json 2>/dev/null \
      | grep -o '"executable":"[^"]*perf_chol_cycles_gate"' | head -1 \
      | sed 's/.*"executable":"//;s/"//')
[ -x "$BIN" ] || { echo "gate binary not built"; exit 1; }
echo "BIN=$BIN  N=$N REPS=$REPS K=$K"

# Execution proof: full-array checksums MUST differ (arm switch really flipped).
DB=$("$BIN" base "$N" 1 full); DC=$("$BIN" cand "$N" 1 full)
echo "EXEC_PROOF base:[$DB]"
echo "EXEC_PROOF cand:[$DC]"
[ "${DB##*digest=}" != "${DC##*digest=}" ] || { echo "FAIL: identical digests — arm switch dead"; exit 3; }

cyc(){ perf stat -e cycles -x, -- "$BIN" "$1" "$N" "$REPS" 2>/tmp/psgate >/dev/null; \
       awk -F, '/cycles/{print $1; exit}' /tmp/psgate; }

"$BIN" base "$N" 2 >/dev/null; "$BIN" cand "$N" 2 >/dev/null   # warm caches/pages
: > /tmp/chol_gate_samples
for _ in $(seq 1 "$K"); do
  echo "base $(cyc base)"   >> /tmp/chol_gate_samples
  echo "cand $(cyc cand)"   >> /tmp/chol_gate_samples
  echo "nullb $(cyc base)"  >> /tmp/chol_gate_samples
done

python3 - <<'PY'
import statistics
base=[];cand=[];nullb=[]
for line in open("/tmp/chol_gate_samples"):
    p=line.split()
    if len(p)!=2: continue
    try: v=float(p[1])
    except ValueError: continue
    {"base":base,"cand":cand,"nullb":nullb}.get(p[0],[]).append(v)
mb=statistics.median(base); mc=statistics.median(cand)
print("  base cycles median=%.4e cv=%.2f%%"%(mb,statistics.pstdev(base)/mb*100))
print("  cand cycles median=%.4e cv=%.2f%%"%(mc,statistics.pstdev(cand)/mc*100))
print("  LEVER base/cand = %.4f x  (>1 = cand faster / fewer cycles)"%(mb/mc))
pr=sorted(b/nn for b,nn in zip(base,nullb))
print("  NULL  base/nullb median=%.4f  range=[%.4f, %.4f]"%(statistics.median(pr),pr[0],pr[-1]))
sep = (mb/mc) > pr[-1]
print("  VERDICT: %s (lever %.4f vs null_hi %.4f)"%("DECIDED" if sep else "IN-FLOOR", mb/mc, pr[-1]))
PY
