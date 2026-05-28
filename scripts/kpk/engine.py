"""
KPK 1-ply engine.

Loads the trained decision tree, plays from every legal KPK position using
a 1-ply lookahead, and measures how often it achieves tablebase-optimal play.

Usage:
    python scripts/kpk/engine.py
"""

import json
import os
import chess
import joblib
import pandas as pd
from pathlib import Path

from endgame.features.kpk_geom import extract_kpk_features
from endgame.utils.logger import make_run_dir, save_json


def find_latest_model(base="logs/kpk"):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib found under {base}")
    return runs[-1]


def load_feature_cols(model_path):
    meta_path = Path(model_path).parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["features"]


def build_fen_lookup(full_path, features_path):
    """
    Returns dict mapping stripped FEN → true WDL.
    FEN is stripped to first 4 fields (position + side to move + castling + ep)
    so that move-counter differences don't break lookups.
    """
    full_df = pd.read_csv(full_path)
    feat_df = pd.read_csv(features_path)

    lookup = {}
    for i in range(len(full_df)):
        fen = full_df.at[i, "fen"]
        wdl = int(feat_df.at[i, "wdl"])
        key = " ".join(fen.split()[:4])
        lookup[key] = wdl
    return lookup


def fen_key(board):
    return " ".join(board.fen().split()[:4])


def predict_wdl(model, feature_cols, board):
    """Predict WDL. Returns None if the board is no longer a KPK position."""
    try:
        raw = extract_kpk_features(board)
    except ValueError:
        return None
    X = pd.DataFrame([raw]).reindex(columns=feature_cols)
    return int(model.predict(X)[0])


def engine_move(model, feature_cols, board):
    """
    1-ply: pick the move whose successor position has the worst WDL
    for the opponent (= best for side to move).
    Pawn promotions are treated as immediate wins (score = 2).
    Returns (best_move, predicted_wdl_for_current_side).
    """
    best_score = -999
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        succ_wdl = predict_wdl(model, feature_cols, board)
        if succ_wdl is None:
            # Pawn promoted — this is a win for the side that promoted
            score = 2
        else:
            score = -succ_wdl
        if score > best_score:
            best_score = score
            best_move = move
        board.pop()

    return best_move, best_score


def main():
    model_path = find_latest_model()
    print(f"Model: {model_path}")

    model = joblib.load(model_path)
    feature_cols = load_feature_cols(model_path)
    print(f"Features: {len(feature_cols)}")

    print("Building FEN lookup table...")
    fen_lookup = build_fen_lookup(
        "data/processed/kpk/full.csv",
        "data/processed/kpk/features.csv",
    )
    print(f"Positions in lookup: {len(fen_lookup)}")

    full_df = pd.read_csv("data/processed/kpk/full.csv")
    feat_df = pd.read_csv("data/processed/kpk/features.csv")

    run_dir = make_run_dir(base="logs/kpk_engine")

    total = 0
    optimal = 0
    missed_lookup = 0

    # Per-class tracking: key = true WDL of current position
    by_class = {
        2:  {"total": 0, "optimal": 0},
        0:  {"total": 0, "optimal": 0},
        -2: {"total": 0, "optimal": 0},
    }

    print("Running engine on all positions...")

    for i in range(len(full_df)):
        if i % 20000 == 0:
            print(f"  {i:,} / {len(full_df):,}")

        fen = full_df.at[i, "fen"]
        true_wdl = int(feat_df.at[i, "wdl"])
        board = chess.Board(fen)

        legal = list(board.legal_moves)
        if not legal:
            continue

        move, _ = engine_move(model, feature_cols, board)
        if move is None:
            continue

        # Look up true WDL of the chosen successor
        board.push(move)
        key = fen_key(board)
        board.pop()

        if key not in fen_lookup:
            missed_lookup += 1
            continue

        succ_true_wdl_for_opponent = fen_lookup[key]
        # From current side's perspective
        succ_true_wdl_for_us = -succ_true_wdl_for_opponent

        # Optimal: result we achieve equals the theoretical best
        # WIN  (2): must find a move where opponent is in LOSS (-2 from their view)
        # DRAW (0): must find a move where opponent is in DRAW
        # LOSS (-2): any move is "as good as it gets"
        if true_wdl == -2:
            is_optimal = True
        else:
            is_optimal = succ_true_wdl_for_us >= true_wdl

        total += 1
        by_class[true_wdl]["total"] += 1

        if is_optimal:
            optimal += 1
            by_class[true_wdl]["optimal"] += 1

    overall_rate = optimal / total if total > 0 else 0.0

    print("\n=== Engine Results ===")
    print(f"Total positions evaluated : {total:,}")
    print(f"Missed lookup             : {missed_lookup:,}")
    print(f"Optimal play rate         : {overall_rate:.6f}  ({optimal:,}/{total:,})")
    print()

    labels = {2: "WIN", 0: "DRAW", -2: "LOSS"}
    for wdl_val, s in by_class.items():
        if s["total"] > 0:
            rate = s["optimal"] / s["total"]
            print(f"  {labels[wdl_val]:5s}: {rate:.6f}  ({s['optimal']:,}/{s['total']:,})")

    summary = {
        "model": str(model_path),
        "n_features": len(feature_cols),
        "total_positions": total,
        "missed_lookup": missed_lookup,
        "optimal_moves": optimal,
        "optimal_rate": overall_rate,
        "by_class": {
            labels[k]: {
                "total": v["total"],
                "optimal": v["optimal"],
                "rate": v["optimal"] / v["total"] if v["total"] > 0 else 0.0,
            }
            for k, v in by_class.items()
        },
    }

    save_json(summary, f"{run_dir}/summary.json")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
