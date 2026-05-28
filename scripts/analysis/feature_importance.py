"""
Feature importance analysis across all trained endgame classifiers.

Loads the latest model for each endgame, ranks features by importance,
and prints a clean summary. The top features should correspond to named
chess concepts — this is the primary interpretability evidence.

Usage:
    python scripts/analysis/feature_importance.py
"""

import json
import joblib
import numpy as np
from pathlib import Path


# Known chess concept labels for common feature names
CHESS_CONCEPTS = {
    # Universal
    "turn":                       "whose move (tempo)",
    "king_opposition":            "direct opposition",
    "d_wk_bk":                   "WK–BK Chebyshev distance",
    "m_wk_bk":                   "WK–BK Manhattan distance",
    "bk_edge":                   "BK distance from edge",
    "bk_on_edge":                "BK is on edge",
    "bk_to_corner":              "BK distance from any corner",
    "bk_in_corner":              "BK is in corner",
    "bk_mobility":               "BK legal move count (zugzwang detector)",
    "bk_geo_mobility":           "BK geometric mobility",
    # KRK rook
    "d_wr_bk":                   "rook–BK distance",
    "rook_on_bk_rank":           "rook cuts BK's rank",
    "rook_on_bk_file":           "rook cuts BK's file",
    "rook_protected":            "rook defended by WK",
    # KPK pawn
    "pawn_rank":                 "pawn's rank (progress toward promotion)",
    "d_bk_p":                    "BK–pawn Chebyshev distance",
    "bk_vs_p_rank":              "BK rank relative to pawn (ahead/behind)",
    "bk_p_dr":                   "BK–pawn rank difference (signed)",
    "bk_p_df":                   "BK–pawn file difference (signed)",
    "file_gap":                  "file distance between kings and pawn",
    "wk_dist_to_nearest_key":    "WK distance from pawn's key squares",
    "e2_bk_p":                   "BK inside pawn square (rule of the square)",
    "e2_wk_p":                   "WK inside pawn square",
    # KBNK corner geometry
    "bishop_color":              "bishop square color",
    "bk_to_correct_corner":      "BK dist from correct-color corner",
    "bk_to_wrong_corner":        "BK dist from wrong-color corner",
    "bk_in_wrong_corner":        "BK in wrong corner (stalemate risk)",
    "bk_near_wrong_corner":      "BK near wrong corner",
    "corner_color_advantage":    "wrong-corner dist minus correct-corner dist",
    "bk_same_color_as_bishop":   "BK on same color as bishop",
    "d_wb_bk":                   "bishop–BK distance",
    "d_wn_bk":                   "knight–BK distance",
    "d_wk_wn":                   "WK–knight distance",
    "d_wk_wb":                   "WK–bishop distance",
    "d_wn_wb":                   "knight–bishop distance",
    "m_wn_bk":                   "knight–BK Manhattan distance",
    "m_wb_bk":                   "bishop–BK Manhattan distance",
    "knight_covers_bk_neighbors":"knight controls BK escape squares",
    "knight_near_bk":            "knight within 2 squares of BK",
    "bishop_attacks_bk_color":   "bishop attacks BK's square color",
    "bishop_on_bk_diagonal":     "bishop on same diagonal as BK",
    # KBBK bishop parity
    "bishops_same_color":        "both bishops same color (insufficient material)",
    "d_light_b_bk":              "light bishop–BK distance",
    "d_dark_b_bk":               "dark bishop–BK distance",
    "d_b_min":                   "closer bishop–BK distance",
    "d_b_max":                   "farther bishop–BK distance",
    "d_wk_light_b":              "WK–light bishop distance",
    "d_wk_dark_b":               "WK–dark bishop distance",
    "bk_neighbors_covered_union":"BK escape squares covered by bishops",
    # KNNK knight coordination
    "bk_neighbors_covered_by_near": "near knight covers BK escape squares",
    "bk_neighbors_covered_union":   "both knights cover BK escape squares",
    "d_wn_near_bk":              "closer knight–BK distance",
    "d_wn_far_bk":               "farther knight–BK distance",
    "d_wk_wn_near":              "WK–closer knight distance",
    "d_wk_wn_far":               "WK–farther knight distance",
    "d_nn":                      "inter-knight distance",
    "d_wn_max":                  "farther knight–BK distance (max)",
    "m_wn_far_bk":               "farther knight–BK Manhattan distance",
    "nn_coverage":               "total squares both knights attack",
    "nn_overlap":                "squares both knights attack simultaneously",
    "knights_same_color":        "both knights on same square color",
}


ENDGAMES = {
    "KRK":  "logs/krk_classifier",
    "KPK":  "logs/kpk",
    "KBNK": "logs/kbnk_classifier",
    "KNNK": "logs/knnk_classifier",
    "KBBK": "logs/kbbk_classifier",
}

TOP_N = 10


def find_latest_run(base):
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        return None
    return runs[-1].parent


def load_endgame(name, base):
    run_dir = find_latest_run(base)
    if run_dir is None:
        print(f"  [{name}] no model found — skipping")
        return None

    model = joblib.load(run_dir / "model.joblib")
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)

    features = meta["features"]
    importances = model.feature_importances_

    ranked = sorted(
        zip(features, importances),
        key=lambda x: -x[1]
    )
    return {
        "run_dir": run_dir,
        "depth": model.get_depth(),
        "leaves": model.get_n_leaves(),
        "n_used": int((importances > 0).sum()),
        "ranked": ranked,
    }


def concept(name):
    return CHESS_CONCEPTS.get(name, "—")


def main():
    print("=" * 70)
    print("FEATURE IMPORTANCE ACROSS ENDGAMES")
    print("=" * 70)

    all_results = {}

    for endgame, base in ENDGAMES.items():
        result = load_endgame(endgame, base)
        if result is None:
            continue
        all_results[endgame] = result

        print(f"\n{'─' * 70}")
        print(f"  {endgame}   depth={result['depth']}  leaves={result['leaves']}  "
              f"features used={result['n_used']}/{len(result['ranked'])}")
        print(f"{'─' * 70}")
        print(f"  {'Rank':<5} {'Importance':>10}  {'Feature':<35} {'Chess concept'}")
        print(f"  {'─'*4} {'─'*10}  {'─'*35} {'─'*25}")

        for rank, (feat, imp) in enumerate(result["ranked"][:TOP_N], 1):
            c = concept(feat)
            print(f"  {rank:<5} {imp:>10.4f}  {feat:<35} {c}")

        # Show which features were NOT used at all
        unused = [f for f, i in result["ranked"] if i == 0]
        if unused:
            print(f"\n  Unused features ({len(unused)}): {', '.join(unused[:8])}"
                  + (" ..." if len(unused) > 8 else ""))

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY — TOP FEATURE PER ENDGAME")
    print(f"{'=' * 70}")
    print(f"  {'Endgame':<8} {'Top feature':<35} {'Chess concept'}")
    print(f"  {'─'*7} {'─'*35} {'─'*25}")
    for endgame, result in all_results.items():
        top_feat, top_imp = result["ranked"][0]
        print(f"  {endgame:<8} {top_feat:<35} {concept(top_feat)}")

    # Compression summary (raw vs tree minimum)
    print(f"\n{'=' * 70}")
    print("COMPRESSION — LEAF COUNT vs POSITION COUNT")
    print(f"{'=' * 70}")
    POSITIONS = {
        "KRK": 900_000, "KPK": 331_000,
        "KBNK": 24_536_088, "KNNK": 12_579_944, "KBBK": 11_891_488,
    }
    print(f"  {'Endgame':<8} {'Positions':>12}  {'Leaves':>8}  {'Pos/Leaf':>10}  {'Bits/pos (raw)':>15}")
    print(f"  {'─'*7} {'─'*12}  {'─'*8}  {'─'*10}  {'─'*15}")
    import math
    for endgame, result in all_results.items():
        n = POSITIONS.get(endgame, 0)
        leaves = result["leaves"]
        if n and leaves:
            ratio = n / leaves
            bits = math.log2(leaves) / math.log2(n) if n > 1 else 0
            print(f"  {endgame:<8} {n:>12,}  {leaves:>8}  {ratio:>10,.0f}  "
                  f"{'~' + str(round(math.ceil(math.log2(3)), 2)) + ' (raw)':>15}")

    print()


if __name__ == "__main__":
    main()
