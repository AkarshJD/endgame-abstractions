"""
Compression analysis for the KPK decision tree model.

Computes:
  - Raw WDL storage cost (uncompressed)
  - Serialised model file size
  - Tree structure metrics (nodes, leaves, depth)
  - Compression ratio and bits-per-position
  - Syzygy tablebase file size (if available)

Usage:
    python scripts/kpk/compression_analysis.py
"""

import json
import math
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from endgame.utils.logger import make_run_dir, save_json


SYZYGY_PATH = "storage/syzygy/3_4_5"
MODEL_BASE = "logs/kpk"
FEATURES_PATH = "data/processed/kpk/features.csv"


def find_latest_model(base=MODEL_BASE):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib found under {base}")
    return runs[-1]


def tree_node_bytes(tree):
    """
    Estimate the minimal serialised size of the tree structure.
    Each internal node stores: feature index, threshold, left/right child pointers.
    Each leaf stores: class label.
    Uses int16 for feature index, float32 for threshold, int32 for pointers.
    """
    n_nodes = tree.node_count
    n_leaves = tree.n_node_samples[tree.children_left == -1].size
    n_internal = n_nodes - n_leaves

    bytes_per_internal = 2 + 4 + 4 + 4   # feature(int16) + threshold(f32) + left(int32) + right(int32)
    bytes_per_leaf = 1                    # class label: 3 classes fits in 2 bits, use 1 byte

    return n_internal * bytes_per_internal + n_leaves * bytes_per_leaf


def syzygy_file_sizes(syzygy_dir, endgame="KPK"):
    """Return WDL and DTZ file sizes in bytes if they exist."""
    sizes = {}
    for ext in ("rtbw", "rtbz"):
        name = f"{endgame}.{ext}"
        path = Path(syzygy_dir) / name
        if path.exists():
            sizes[ext] = path.stat().st_size
    return sizes


def main():
    model_path = find_latest_model()
    print(f"Model: {model_path}")

    model = joblib.load(model_path)
    tree = model.tree_

    meta_path = Path(model_path).parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    n_positions = meta["n_samples"]
    n_classes = 3  # WIN, DRAW, LOSS

    # ── Raw storage cost ───────────────────────────────────────────────────────
    bits_per_pos_raw = math.ceil(math.log2(n_classes))          # 2 bits
    raw_bytes = math.ceil(n_positions * bits_per_pos_raw / 8)

    # ── Model file size ────────────────────────────────────────────────────────
    model_file_bytes = os.path.getsize(model_path)

    # ── Tree structure size (theoretical minimum) ─────────────────────────────
    tree_min_bytes = tree_node_bytes(tree)

    # ── Compression ratios ────────────────────────────────────────────────────
    ratio_vs_raw    = raw_bytes / model_file_bytes
    ratio_vs_theory = raw_bytes / tree_min_bytes

    bits_per_pos_model  = (model_file_bytes  * 8) / n_positions
    bits_per_pos_theory = (tree_min_bytes    * 8) / n_positions

    # ── Leaf statistics ───────────────────────────────────────────────────────
    leaf_mask = tree.children_left == -1
    n_leaves = int(leaf_mask.sum())
    samples_per_leaf = tree.n_node_samples[leaf_mask]

    # ── Syzygy file size ──────────────────────────────────────────────────────
    syzygy_sizes = syzygy_file_sizes(SYZYGY_PATH)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n=== KPK Compression Analysis ===\n")
    print(f"Positions            : {n_positions:>12,}")
    print(f"Tree depth           : {tree.max_depth:>12}")
    print(f"Tree nodes           : {tree.node_count:>12,}")
    print(f"Tree leaves          : {n_leaves:>12,}")
    print(f"Avg positions/leaf   : {n_positions / n_leaves:>12.1f}")
    print(f"Max positions/leaf   : {int(samples_per_leaf.max()):>12,}")
    print()
    print(f"Raw WDL storage      : {raw_bytes:>12,} bytes  ({raw_bytes/1024:.1f} KB)")
    print(f"Model file size      : {model_file_bytes:>12,} bytes  ({model_file_bytes/1024:.1f} KB)")
    print(f"Tree min structure   : {tree_min_bytes:>12,} bytes  ({tree_min_bytes/1024:.1f} KB)")
    print()
    print(f"Compression vs raw (model file) : {ratio_vs_raw:.1f}x")
    print(f"Compression vs raw (tree only)  : {ratio_vs_theory:.1f}x")
    print()
    print(f"Bits/position (raw)    : {bits_per_pos_raw:.4f}")
    print(f"Bits/position (model)  : {bits_per_pos_model:.4f}")
    print(f"Bits/position (tree)   : {bits_per_pos_theory:.4f}")

    if syzygy_sizes:
        print()
        for ext, sz in syzygy_sizes.items():
            ratio = sz / model_file_bytes
            print(f"Syzygy KPK.{ext}  : {sz:>10,} bytes  ({sz/1024:.1f} KB)  → model is {ratio:.1f}x {'larger' if ratio < 1 else 'smaller'}")

    # ── Save results ──────────────────────────────────────────────────────────
    run_dir = make_run_dir(base="logs/kpk_compression")

    results = {
        "model": str(model_path),
        "n_positions": n_positions,
        "tree": {
            "depth": int(tree.max_depth),
            "nodes": int(tree.node_count),
            "leaves": n_leaves,
            "avg_positions_per_leaf": float(n_positions / n_leaves),
        },
        "storage_bytes": {
            "raw_wdl_uncompressed": raw_bytes,
            "model_file": model_file_bytes,
            "tree_minimum": tree_min_bytes,
        },
        "compression_ratio": {
            "model_vs_raw": ratio_vs_raw,
            "tree_vs_raw": ratio_vs_theory,
        },
        "bits_per_position": {
            "raw": bits_per_pos_raw,
            "model_file": bits_per_pos_model,
            "tree_minimum": bits_per_pos_theory,
        },
        "syzygy_file_bytes": syzygy_sizes,
    }

    save_json(results, f"{run_dir}/compression.json")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
