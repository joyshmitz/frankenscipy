#!/usr/bin/env bash
# Copy committed seeds into fuzz/corpus/<target>/ before launching a campaign.
# Seeds live under fuzz/seeds/<target>/ (tracked in git); corpus is gitignored.
# Per /testing-fuzzing hard rule 8: every target needs a bootstrap corpus.

set -euo pipefail

FUZZ_DIR="${FUZZ_DIR:-$(dirname "$0")/../fuzz}"
FUZZ_DIR="$(cd "$FUZZ_DIR" && pwd)"

if [[ ! -d "$FUZZ_DIR/seeds" ]]; then
  echo "error: $FUZZ_DIR/seeds does not exist" >&2
  exit 1
fi

count=0
for target_dir in "$FUZZ_DIR/seeds"/*/; do
  target="$(basename "$target_dir")"
  corpus_dir="$FUZZ_DIR/corpus/$target"
  mkdir -p "$corpus_dir"
  for seed in "$target_dir"*; do
    [[ -f "$seed" ]] || continue
    cp --update=none "$seed" "$corpus_dir/" 2>/dev/null || cp -n "$seed" "$corpus_dir/" 2>/dev/null || true
  done
  count=$((count + 1))
done

echo "Seeded $count targets under $FUZZ_DIR/corpus/"
