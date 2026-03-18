#!/usr/bin/env python3
"""GPTCache benchmark using the same dataset as xordb.

Uses ONNX embeddings + FAISS index — GPTCache's recommended setup
for local semantic caching.
"""

import time
import resource
import tracemalloc

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache, get, put
from gptcache.embedding import Onnx as OnnxEmbedding
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from dataset import DATASET


def run_benchmark():
    # Initialize GPTCache with ONNX embeddings (local, no API calls).
    onnx = OnnxEmbedding()
    cache = Cache()
    init_similar_cache(
        cache_obj=cache,
        data_manager=manager_factory(
            "sqlite,faiss",
            vector_params={"dimension": onnx.dimension},
        ),
        embedding=onnx,
        evaluation=SearchDistanceEvaluation(),
    )

    # Populate cache (only using the unique cached keys).
    print("Populating cache...")
    seen = set()
    pop_start = time.perf_counter()
    for cached, _, answer, _, _ in DATASET:
        if cached not in seen:
            put(cached, answer, cache_obj=cache)
            seen.add(cached)
    pop_elapsed = time.perf_counter() - pop_start
    print(f"  Populated {len(seen)} unique entries in {pop_elapsed*1000:.1f}ms")

    # Measure lookups.
    print("Running lookups...")
    tracemalloc.start()

    results = []
    start = time.perf_counter()

    for cached, lookup, expected, expect_hit, category in DATASET:
        result = get(lookup, cache_obj=cache)
        got_hit = result is not None
        results.append((lookup, expect_hit, got_hit, category))

    elapsed = time.perf_counter() - start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # RSS from the OS — includes C libraries (ONNX, FAISS), mmap'd files, etc.
    # resource.getrusage returns maxrss in KB on Linux, bytes on macOS.
    import platform
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        rss_mb = rss_raw / 1024 / 1024  # bytes → MB
    else:
        rss_mb = rss_raw / 1024  # KB → MB

    n = len(results)
    avg_latency_ms = (elapsed / n) * 1000

    # Classification metrics.
    tp = sum(1 for _, e, g, _ in results if e and g)
    tn = sum(1 for _, e, g, _ in results if not e and not g)
    fp = sum(1 for _, e, g, _ in results if not e and g)
    fn = sum(1 for _, e, g, _ in results if e and not g)
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    # Category breakdown.
    categories = ["match", "neg", "hard-neg"]
    cat_total = {c: 0 for c in categories}
    cat_correct = {c: 0 for c in categories}
    for _, expect, got, cat in results:
        cat_total[cat] += 1
        if expect == got:
            cat_correct[cat] += 1

    # Print report.
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            GPTCache — ONNX + FAISS + SQLite             ║")
    print("╠══════════════════════════════════════════════════════════╣")

    def row(label, value):
        print(f"║  {label:<15}  {value:<39} ║")

    row(
        "Dataset:",
        f"{n} queries ({cat_total['match']} match, {cat_total['neg']} neg, {cat_total['hard-neg']} hard-neg)",
    )
    row("Precision:", f"{precision:.1f}% ({tp}/{tp+fp} hits correct)")
    row("Recall:", f"{recall:.1f}% ({tp}/{tp+fn} matches found)")
    row("F1 Score:", f"{f1:.1f}%")
    row("FP Rate:", f"{fpr:.1f}% ({fp}/{fp+tn} wrong hits)")
    row("False neg:", f"{fn}  (should hit, got miss)")
    row("Total time:", f"{elapsed*1000:.1f}ms")
    row("Avg latency:", f"{avg_latency_ms:.2f}ms / query")
    row("Heap (py):", f"{peak_mem / 1024 / 1024:.2f} MB")
    row("RSS (process):", f"{rss_mb:.2f} MB")
    row("Dependencies:", "gptcache, faiss, onnxruntime, numpy")

    print("╠══════════════════════════════════════════════════════════╣")
    for cat in categories:
        if cat_total[cat] > 0:
            row(f"{cat}:", f"{cat_correct[cat]}/{cat_total[cat]} correct")

    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    run_benchmark()
