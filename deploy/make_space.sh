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
[ -f "$REAL" ] && cp "$REAL" "$OUT/cutting_sim/output/" || echo "WARNING: $REAL is missing: the backend cannot start without it."
cp "$ROOT/cutting_sim/output/rca_gain_effect_k10.json" "$OUT/cutting_sim/output/"
echo "Built $OUT ($(du -sh "$OUT" | cut -f1))"
