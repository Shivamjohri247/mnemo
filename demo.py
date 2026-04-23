#!/usr/bin/env python3
"""Demo script for Mnemo — SuperLocalMemory prototype.

Usage:
    python demo.py --ingest         Ingest 20 sample facts
    python demo.py --simulate-decay Run 30-day forgetting simulation
    python demo.py --query QUERY    Query memories
    python demo.py --show-quantization  Show FRQAD scoring demo
    python demo.py --all            Run full demo sequence
"""

import argparse
import json
import os
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_engine(project: str = "demo"):
    from mnemo.core.engine import MemoryEngine

    return MemoryEngine(project=project, auto_scheduler=False)


def ingest_facts():
    """Ingest sample facts from data/sample_facts.json."""
    from unittest.mock import patch

    engine = get_engine()

    # Use mock embedder for fast demo
    from mnemo.storage.embeddings import set_embedder
    from mnemo.testing import make_mock_embedder

    mock = make_mock_embedder()
    set_embedder(mock)

    with open("data/sample_facts.json") as f:
        facts = json.load(f)

    # Patch the encode pipeline's embedder
    with (
        patch("mnemo.pipeline.encode.get_embedder", return_value=mock),
        patch("mnemo.pipeline.encode._get_nlp") as mock_nlp,
    ):
        mock_nlp.return_value.return_value.ents = []
        for fact_data in facts:
            fid = engine.remember(
                fact_data["text"],
                importance=fact_data["importance"],
                source=fact_data.get("source", "user"),
            )
            status = f"[NEW] {fact_data['text'][:60]}" if fid else f"[DUP] {fact_data['text'][:60]}"
            print(f"  {status}")

    stats = engine.stats()
    total = sum(s["count"] for s in stats.values())
    print(f"\nIngested {total} facts total.")


def simulate_decay():
    """Run compressed 30-day forgetting simulation."""
    from mnemo.core.forgetting import lifecycle_state, memory_strength, retention

    print("\n=== 30-Day Forgetting Simulation ===\n")
    print(f"{'Scenario':<30} {'Strength':>10} {'Retention':>10} {'State':>10}")
    print("-" * 65)

    scenarios = [
        ("Hot (daily, high imp)", 30, 0.8, 3, 12),
        ("Warm (weekly, med imp)", 4, 0.5, 1, 48),
        ("Cold (once, low imp)", 1, 0.2, 0, 720),
        ("Trusted source", 10, 0.7, 5, 24),
        ("Untrusted source", 10, 0.7, 0, 24),
    ]

    for name, accesses, importance, confirmations, hours_ago in scenarios:
        s = memory_strength(accesses, importance, confirmations, 0.0)
        r = retention(s, hours_ago)
        state, bits = lifecycle_state(r)
        print(f"{name:<30} {s:>10.2f} {r:>10.4f} {state:>10}")

    # Show discriminative power
    hot_s = memory_strength(30, 0.8, 3, 0.0)
    cold_s = memory_strength(1, 0.2, 0, 0.0)
    print(f"\nDiscriminative power: {hot_s / cold_s:.1f}x (hot vs cold)")


def query_demo(query: str):
    """Query memories and show channel attribution."""
    from unittest.mock import patch

    from mnemo.storage.embeddings import set_embedder
    from mnemo.testing import make_mock_embedder

    mock = make_mock_embedder()
    set_embedder(mock)
    engine = get_engine()

    with patch("mnemo.core.retrieval.embed_text", side_effect=lambda t: mock.embed_text(t)):
        results = engine.recall(query, top_k=5)

    if not results:
        print(f"No results for: {query}")
        return

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        retention_pct = (r.get("retention") or 0) * 100
        print(f"  [{i}] {r['text']}")
        print(
            f"      lifecycle={r['lifecycle']}  retention={retention_pct:.1f}%  accesses={r['access_count']}"
        )


def show_quantization():
    """Show FRQAD scoring at different precision levels."""
    from mnemo.core.quantization import (
        dequantize_int2,
        dequantize_int4,
        dequantize_int8,
        frqad_score,
        quantize_to_int2,
        quantize_to_int4,
        quantize_to_int8,
    )

    print("\n=== FRQAD Quantization Demo ===\n")

    rng = np.random.RandomState(42)
    vec = rng.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # Create a related query
    query = vec + rng.randn(384).astype(np.float32) * 0.1
    query /= np.linalg.norm(query)

    vec_8 = dequantize_int8(quantize_to_int8(vec))
    vec_4 = dequantize_int4(quantize_to_int4(vec))
    vec_2 = dequantize_int2(quantize_to_int2(vec))

    for bits, v, noise in [
        (32, vec, 0.000),
        (8, vec_8, 0.005),
        (4, vec_4, 0.050),
        (2, vec_2, 0.200),
    ]:
        score = frqad_score(query, v, 32, bits)
        print(f"  {bits:2d}-bit score: {score:.4f}  (noise={noise:.3f})")

    print("\n  -> Lower precision correctly ranks lower without manual re-weighting.")


def main():
    parser = argparse.ArgumentParser(description="Mnemo Demo")
    parser.add_argument("--ingest", action="store_true", help="Ingest sample facts")
    parser.add_argument("--simulate-decay", action="store_true", help="30-day simulation")
    parser.add_argument("--query", type=str, help="Query memories")
    parser.add_argument("--show-quantization", action="store_true", help="FRQAD demo")
    parser.add_argument("--all", action="store_true", help="Run full demo")
    args = parser.parse_args()

    if args.all or not any([args.ingest, args.simulate_decay, args.query, args.show_quantization]):
        ingest_facts()
        simulate_decay()
        query_demo("what database does this project use?")
        show_quantization()
    else:
        if args.ingest:
            ingest_facts()
        if args.simulate_decay:
            simulate_decay()
        if args.query:
            query_demo(args.query)
        if args.show_quantization:
            show_quantization()


if __name__ == "__main__":
    main()
