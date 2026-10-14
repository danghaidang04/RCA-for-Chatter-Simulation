#!/usr/bin/env bash
# Builds a clean, self-contained folder for a Hugging Face Docker Space (or any Docker host) in build/space.
# Usage: ./deploy/make_space.sh        then follow DEPLOY.md
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; OUT="$ROOT/build/space"
rm -rf "$OUT"; mkdir -p "$OUT/prototype_kg_bn_llm/report" "$OUT/prototype_kg_bn_llm/documents" "$OUT/cutting_sim/output"
P="$ROOT/prototype_kg_bn_llm"
cp "$ROOT/Dockerfile" "$OUT/Dockerfile"; cp "$ROOT/deploy/space_README.md" "$OUT/README.md"
cp -R "$P/app" "$OUT/prototype_kg_bn_llm/app"; rm -rf "$OUT/prototype_kg_bn_llm/app/__pycache__" "$OUT/prototype_kg_bn_llm/app/"*.log
cp "$P"/*.py "$OUT/prototype_kg_bn_llm/"; cp -R "$P/kg" "$OUT/prototype_kg_bn_llm/kg"
cp "$P/documents/corpus_demo.json" "$OUT/prototype_kg_bn_llm/documents/"
cp "$P/report/cond24_results_k10.json" "$OUT/prototype_kg_bn_llm/report/"
cp "$ROOT/cutting_sim/"*.py "$OUT/cutting_sim/"
REAL="$ROOT/cutting_sim/output/real_windows.npz"
if [ -f "$REAL" ]; then   # Hugging Face rejects non-LFS files above 10 MiB, so ship it in 3 parts; the Dockerfile joins them
  split -b 7000000 "$REAL" "$OUT/cutting_sim/output/real_windows.npz.part_"
  (cd "$ROOT/cutting_sim/output" && shasum -a 256 real_windows.npz | cut -d' ' -f1) > "$OUT/cutting_sim/output/real_windows.sha256"
else echo "WARNING: $REAL is missing: the backend cannot start without it."; fi
cp "$ROOT/cutting_sim/output/rca_gain_effect_k10.json" "$OUT/cutting_sim/output/"
echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
