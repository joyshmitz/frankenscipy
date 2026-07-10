#!/bin/bash
# Median SELF-TIME gate via perf stat cycles. Interleaved base/avx2/base per iteration.
SP="$1"; KERN="$2"; INNER="$3"; K="${4:-15}"
cyc(){ perf stat -e cycles -x, -- "$1" "$KERN" "$INNER" 2>&1 | awk -F, '/cycles/{print $1}'; }
# warm
for i in 1 2; do cyc "$SP/k_sse2" >/dev/null; cyc "$SP/k_avx2" >/dev/null; done
bn=(); an=(); nn=()
for i in $(seq 1 $K); do
  b=$(cyc "$SP/k_sse2"); a=$(cyc "$SP/k_avx2"); b2=$(cyc "$SP/k_sse2")
  bn+=($b); an+=($a); nn+=($(python3 -c "print($b/$b2)")); 
  # cand ratio base/avx2 stored via python below
  echo "$b $a $b2"
done
