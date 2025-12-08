#!/usr/bin/env python3
"""
Validate Graph Construction Output

Quick script to verify the graph was built correctly.

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz

# Project root (works from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths to graph files
ADJACENCY_PATH = PROJECT_ROOT / "data" / "processed" / "adjacency_matrix.npz"
NODE_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "node_map.json"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "graph_stats.json"


def validate_graph():
    """Validate the constructed graph."""
    print("\n" + "=" * 70)
    print("  GRAPH VALIDATION")
    print("=" * 70)

    # Check if files exist
    print("\n1. Checking files exist...")
    if not ADJACENCY_PATH.exists():
        print(f"  ❌ Adjacency matrix not found: {ADJACENCY_PATH}")
        return
    if not NODE_MAP_PATH.exists():
        print(f"  ❌ Node map not found: {NODE_MAP_PATH}")
        return
    if not STATS_PATH.exists():
        print(f"  ❌ Stats file not found: {STATS_PATH}")
        return

    print(f"  ✓ All files exist")

    # Load adjacency matrix
    print("\n2. Loading adjacency matrix...")
    adjacency = load_npz(ADJACENCY_PATH)
    print(f"  ✓ Shape: {adjacency.shape}")
    print(f"  ✓ Type: {type(adjacency)}")
    print(f"  ✓ Non-zero elements: {adjacency.nnz:,}")

    # Load node mapping
    print("\n3. Loading node mapping...")
    with NODE_MAP_PATH.open('r') as f:
        node_map = json.load(f)
    print(f"  ✓ Number of nodes: {len(node_map):,}")
    print(f"  ✓ Sample nodes: {list(node_map.keys())[:3]}")

    # Load statistics
    print("\n4. Loading statistics...")
    with STATS_PATH.open('r') as f:
        stats = json.load(f)
    print(f"  ✓ Statistics:")
    for key, value in stats.items():
        print(f"     - {key}: {value}")

    # Validate consistency
    print("\n5. Validating consistency...")

    # Check if adjacency is square
    if adjacency.shape[0] != adjacency.shape[1]:
        print(f"  ❌ Adjacency is not square: {adjacency.shape}")
        return
    print(f"  ✓ Adjacency is square")

    # Check if number of nodes matches
    if adjacency.shape[0] != len(node_map):
        print(f"  ❌ Node count mismatch: adjacency={adjacency.shape[0]}, node_map={len(node_map)}")
        return
    print(f"  ✓ Node counts match")

    # Check if symmetric
    is_symmetric = (adjacency != adjacency.T).nnz == 0
    print(f"  ✓ Graph is symmetric: {is_symmetric}")

    # Check degree statistics
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(f"\n6. Degree statistics:")
    print(f"  ✓ Min degree: {degrees.min():.2f}")
    print(f"  ✓ Max degree: {degrees.max():.2f}")
    print(f"  ✓ Mean degree: {degrees.mean():.2f}")
    print(f"  ✓ Median degree: {np.median(degrees):.2f}")

    # Check edge weight statistics
    edge_weights = adjacency.data
    print(f"\n7. Edge weight statistics:")
    print(f"  ✓ Min weight: {edge_weights.min():.4f}")
    print(f"  ✓ Max weight: {edge_weights.max():.4f}")
    print(f"  ✓ Mean weight: {edge_weights.mean():.4f}")
    print(f"  ✓ Median weight: {np.median(edge_weights):.4f}")

    print("\n" + "=" * 70)
    print("✓ GRAPH VALIDATION PASSED!")
    print("=" * 70)
    print(f"\nGraph files located at:")
    print(f"  {ADJACENCY_PATH}")
    print(f"  {NODE_MAP_PATH}")
    print(f"  {STATS_PATH}")
    print()


if __name__ == "__main__":
    validate_graph()
