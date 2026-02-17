#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│     xordb vs GPTCache — Semantic Cache Benchmark         │"
echo "│     75 queries · matches + negatives + edge cases      │"
echo "└──────────────────────────────────────────────────────────┘"
echo ""

echo "Building containers (first run downloads ONNX model + runtime)..."
docker build -f benchmarks/Dockerfile.xordb    -t xordb-bench    . -q
docker build -f benchmarks/Dockerfile.gptcache -t gptcache-bench . -q
echo "Done."
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ROUND 1: xordb (n-gram HDC · zero dependencies)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
docker run --rm xordb-bench go test -v -run TestXorDB_NGram_Report -count=1 -timeout=60s .

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ROUND 2: xordb (MiniLM encoder · local ONNX inference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
docker run --rm xordb-bench go test -v -run TestXorDB_MiniLM_Report -count=1 -timeout=120s .

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ROUND 3: GPTCache (ONNX + FAISS + SQLite)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
docker run --rm gptcache-bench

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All 3 rounds ran on the same 75-query dataset."
echo "  Compare: accuracy, false positives, latency, memory."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
