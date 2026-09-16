"""
Faithfulness Decomposition Experiment — v2 (faithfulness-decomposition branch).

WHAT CHANGED FROM v1 AND WHY
------------------------------
v1 added DTZ as a feature to the model and retrained. That was circular: WDL and
DTZ are both Syzygy outputs, and WDL is essentially a function of DTZ's sign and
magnitude under the 50-move rule. Adding one oracle output to predict another
oracle output produces 100% accuracy trivially — not a causal finding. The flat
1.000 ± 0.000 across three structurally different endgames was a tautology
detector firing, not experimental confirmation. v1 numbers go in the lab notebook
as a leakage demonstration.

THE VALID EXPERIMENT
---------------------
The legitimate excluded-information variable is the HALFMOVE CLOCK — a real
observable present in every FEN and visible to both players, not oracle data.
Static geometric features cannot encode the clock because all enumerated positions
are probed from clock=0. The relevant structural question is therefore:

    Do draw-boundary errors concentrate on cursed-win / blessed-loss positions —
    positions whose WDL outcome depends on the halfmove clock?

Cursed / blessed positions are identified from Syzygy's WDL GRANULARITY (the ±1
codes that the current pipeline normalises away). This is a LABEL PROPERTY of the
error set, not a feature added to the model. DTZ never enters as a predictor.

Expected cross-endgame gradient (the real causal test):
  KBNK — 50-move rule manufactures many cursed wins → high cursed share
  KPK  — long pawn endings create some cursed positions → intermediate
  KRK  — draws are stalemate only, no 50-move mechanism → near-zero cursed share

A real causal mechanism produces a gradient. A flat ceiling at 1.000 across all
three is leakage. The gradient is the finding.

Whatever draw-boundary errors remain after removing the cursed/blessed set are
genuine intrinsic-geometry errors — positions the model gets wrong even though
their outcome does not depend on the clock. Those are outcome (B).

GATE NUMBERS
-------------
 1. WIN fidelity  — baseline (5-fold CV mean ± std)
 2. DRAW fidelity — baseline (5-fold CV mean ± std)
 3. LOSS fidelity — baseline (5-fold CV mean ± std)
 4. Cursed/blessed share of draw-boundary errors (full dataset, label analysis)
 5. Genuine geometric residual count (draw-boundary errors NOT cursed/blessed)
 6. KRK cursed share (expected ≈ 0; the placebo control)
 7. Draw-boundary rate vs empirical Monte Carlo null (p-value)
 8. Is KBNK cursed share >> KRK cursed share? (gradient test, y/n + difference)

SCOPE: static features + decision tree only. No NN. No 7-piece endgames.
       No emergent-concept analysis. No MDP/stochastic extensions.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

import chess
import chess.syzygy
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYZYGY_PATH = "storage/syzygy/3_4_5"
N_CV_FOLDS  = 5

ENDGAME_CFG: dict[str, dict] = {
    "kbnk": {
        "features_csv": "data/processed/kbnk/features.csv",
        "model_base":   "logs/kbnk_classifier",
        "output_base":  "logs/kbnk_faithfulness",
    },
    "krk": {
        "features_csv": "data/processed/krk/features.csv",
        "model_base":   "logs/krk_classifier",
        "output_base":  "logs/krk_faithfulness",
    },
    "kpk": {
        "features_csv": "data/processed/kpk/features.csv",
        "model_base":   "logs/kpk",
        "output_base":  "logs/kpk_faithfulness",
    },
}

LABEL_MAP    = {2: "WIN", 0: "DRAW", -2: "LOSS"}
LABEL_VALUES = [2, 0, -2]
DT_PARAMS    = dict(max_depth=None, min_samples_leaf=50, random_state=42)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_latest_model(base: str) -> Path:
    runs = sorted(Path(base).glob("*/model.joblib"))
    if not runs:
        raise FileNotFoundError(f"No model.joblib under {base}/")
    return runs[-1]


def load_baseline(cfg: dict) -> tuple:
    model_path   = find_latest_model(cfg["model_base"])
    model        = joblib.load(model_path)
    meta_path    = model_path.parent / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    # feature_cols contains NO dtz — baseline is clean by construction
    feature_cols = [c for c in meta["features"] if c != "dtz"]
    return model, feature_cols, model_path

# ---------------------------------------------------------------------------
# Raw WDL probing  (±2, ±1, 0 — NOT normalised)
#
# probe_wdl returns:
#   2  — clean win  (forced, ≤50 half-moves to zeroing)
#   1  — cursed win (forced but takes >50 half-moves; draw under 50-move rule)
#   0  — draw
#  -1  — blessed loss (losing but can claim draw under 50-move rule)
#  -2  — clean loss
#
# The current pipeline normalises ±1 → ±2 before training, discarding this
# granularity. Here we recover it for label analysis only — never as a feature.
# ---------------------------------------------------------------------------

def _probe_raw_wdl_worker(args: tuple) -> list:
    fens, syzygy_path = args
    tb  = chess.syzygy.open_tablebase(syzygy_path)
    out = []
    for fen in fens:
        try:
            out.append(tb.probe_wdl(chess.Board(fen)))
        except Exception:
            out.append(None)
    tb.close()
    return out


def probe_raw_wdl_parallel(
    fens: list,
    syzygy_path: str = SYZYGY_PATH,
) -> np.ndarray:
    """
    Probe raw WDL (±2, ±1, 0) for all FENs in parallel.
    Returns int array; None probes imputed as 0.
    """
    n_workers  = max(1, mp.cpu_count() - 1)
    chunk_size = math.ceil(len(fens) / n_workers)
    chunks     = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
    raw: list  = []
    with mp.Pool(n_workers) as pool:
        for r in pool.imap(_probe_raw_wdl_worker,
                           [(c, syzygy_path) for c in chunks]):
            raw.extend(r)
    n_fail = sum(1 for v in raw if v is None)
    if n_fail:
        print(f"  Raw WDL probe failures: {n_fail:,} — imputed 0.")
    return np.array([v if v is not None else 0 for v in raw], dtype=int)

# ---------------------------------------------------------------------------
# Per-class fidelity
# ---------------------------------------------------------------------------

def per_class_fidelity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | None]:
    """Per-class recall against the oracle — reported separately per class."""
    out: dict = {}
    for val in LABEL_VALUES:
        mask = y_true == val
        out[LABEL_MAP[val]] = float((y_pred[mask] == val).mean()) if mask.sum() else None
    return out

# ---------------------------------------------------------------------------
# Ordinal-confound control  (Monte Carlo empirical null)
# ---------------------------------------------------------------------------

def empirical_null_boundary_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = 2000,
    seed:   int = 0,
) -> dict:
    """
    Permute predicted labels within the error set to build the null distribution
    of draw-boundary rates. Reports observed rate as a percentile.
    Sidesteps closed-form prior algebra; gives a p-value instead.
    """
    rng      = np.random.default_rng(seed)
    err_mask = y_true != y_pred
    total    = int(err_mask.sum())
    if total == 0:
        return {"null_mean": 0.0, "null_p95": 0.0, "observed": 0.0,
                "p_value": 1.0, "exceeds_p95": False, "flat_null": 4/6}

    err_true = y_true[err_mask]
    err_pred = y_pred[err_mask]
    observed = float(np.sum((err_true == 0) | (err_pred == 0)) / total)

    null_rates = [
        float(np.sum((err_true == 0) | (rng.permutation(err_pred) == 0)) / total)
        for _ in range(n_perm)
    ]
    null_arr = np.array(null_rates)
    return {
        "null_mean":   float(null_arr.mean()),
        "null_std":    float(null_arr.std()),
        "null_p95":    float(np.percentile(null_arr, 95)),
        "observed":    observed,
        "p_value":     float((null_arr >= observed).mean()),
        "exceeds_p95": observed > float(np.percentile(null_arr, 95)),
        "flat_null":   4 / 6,
    }


def ordinal_null_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    err_mask = y_true != y_pred
    total    = int(err_mask.sum())
    p_draw   = float((y_true == 0).mean())
    if total == 0:
        return {"total_errors": 0, "draw_boundary_errors": 0,
                "win_loss_errors": 0, "draw_boundary_rate": 0.0,
                "empirical_null": empirical_null_boundary_rate(y_true, y_pred),
                "p_draw": p_draw}

    err_true = y_true[err_mask]
    err_pred = y_pred[err_mask]

    return {
        "total_errors":         total,
        "draw_boundary_errors": int(np.sum((err_true == 0) | (err_pred == 0))),
        "win_loss_errors":      int(
            np.sum(((err_true == 2) & (err_pred == -2)) |
                   ((err_true == -2) & (err_pred == 2)))
        ),
        "draw_boundary_rate":   float(np.sum((err_true == 0) | (err_pred == 0)) / total),
        "empirical_null":       empirical_null_boundary_rate(y_true, y_pred),
        "p_draw":               p_draw,
    }

# ---------------------------------------------------------------------------
# Cursed / blessed partition  (the non-circular intervention)
#
# For each draw-boundary error, we ask: is the TRUE position a cursed win (raw
# WDL = 1) or blessed loss (raw WDL = -1)?  These are positions whose outcome
# depends on the halfmove clock — the legitimate excluded observable.
#
# raw_wdl is a LABEL PROPERTY of the error set, never a model feature.
# ---------------------------------------------------------------------------

def cursed_partition(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    raw_wdl:  np.ndarray,
) -> dict:
    """
    Partition draw-boundary errors by whether the true position is
    cursed (raw WDL = 1) or blessed (raw WDL = -1).

    Error taxonomy:
      Type A  Win→Draw   : y_true=2,  y_pred=0
      Type B  Draw→Win/Loss: y_true=0,  y_pred≠0
      Type C  Loss→Draw  : y_true=-2, y_pred=0

    Cursed mechanism applies to:
      Type A where raw_wdl=1  (cursed win predicted as draw)
      Type C where raw_wdl=-1 (blessed loss predicted as draw)

    Type B errors (true draw mislabelled) are NOT cursed by definition —
    these are genuine geometric confusion.

    cursed_share = (cursed Type A + cursed Type C) / total draw-boundary errors
    """
    err_mask = y_true != y_pred

    # Draw-boundary subsets
    type_a = err_mask & (y_true == 2)  & (y_pred == 0)   # Win → Draw
    type_b = err_mask & (y_true == 0)                      # Draw → anything
    type_c = err_mask & (y_true == -2) & (y_pred == 0)   # Loss → Draw

    draw_boundary = type_a | type_b | type_c
    total_db      = int(draw_boundary.sum())

    cursed_a = int((type_a & (raw_wdl == 1)).sum())
    cursed_c = int((type_c & (raw_wdl == -1)).sum())
    total_cursed = cursed_a + cursed_c

    genuine_b       = int(type_b.sum())         # all Type B are genuine
    genuine_a       = int(type_a.sum()) - cursed_a
    genuine_c       = int(type_c.sum()) - cursed_c
    total_genuine   = genuine_a + genuine_b + genuine_c

    cursed_share = total_cursed / total_db if total_db > 0 else 0.0

    return {
        "draw_boundary_total":   total_db,
        "type_a_win_to_draw":    int(type_a.sum()),
        "type_b_draw_to_other":  int(type_b.sum()),
        "type_c_loss_to_draw":   int(type_c.sum()),
        "cursed_a":              cursed_a,
        "cursed_c":              cursed_c,
        "total_cursed":          total_cursed,
        "total_genuine":         total_genuine,
        "cursed_share":          float(cursed_share),
        "p_cursed_in_sample":    float((raw_wdl == 1).mean() + (raw_wdl == -1).mean()),
    }

# ---------------------------------------------------------------------------
# Cross-validation  (baseline fidelity only — one model arm)
# ---------------------------------------------------------------------------

def baseline_cv(
    X:       np.ndarray,
    y:       np.ndarray,
    raw_wdl: np.ndarray,
    n_splits: int = N_CV_FOLDS,
) -> dict:
    """
    5-fold CV on the baseline model (geometric features, no oracle data).
    Reports per-class fidelity mean ± std and ordinal-null test per fold.
    Cursed partition is reported per-fold for error-bar consistency.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_fidelity   = {k: [] for k in ("WIN", "DRAW", "LOSS")}
    fold_ordinal:   list = []
    fold_cursed:    list = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        y_tr, y_te       = y[train_idx],       y[test_idx]
        X_tr, X_te       = X[train_idx],        X[test_idx]
        raw_te           = raw_wdl[test_idx]

        n_draw   = int((y_te == 0).sum())
        n_cursed = int(((raw_te == 1) | (raw_te == -1)).sum())

        clf = DecisionTreeClassifier(**DT_PARAMS)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        fid = per_class_fidelity(y_te, y_pred)
        for k in ("WIN", "DRAW", "LOSS"):
            if fid[k] is not None:
                fold_fidelity[k].append(fid[k])

        ordinal = ordinal_null_test(y_te, y_pred)
        fold_ordinal.append(ordinal)

        cursed = cursed_partition(y_te, y_pred, raw_te)
        fold_cursed.append(cursed)

        db_errors = ordinal["draw_boundary_errors"]
        warn = (" ← WARNING: low count" if db_errors < 100 else "")
        print(f"  Fold {fold_idx+1}/{n_splits}  "
              f"held-out={len(y_te):,}  draws={n_draw:,}  "
              f"cursed_in_fold={n_cursed:,}  "
              f"db_errors={db_errors}  "
              f"cursed_share={cursed['cursed_share']:.3f}{warn}")

    def ms(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    fidelity_agg = {
        k: {"mean": ms(fold_fidelity[k])[0], "std": ms(fold_fidelity[k])[1]}
        for k in ("WIN", "DRAW", "LOSS")
    }

    cursed_shares = [c["cursed_share"] for c in fold_cursed]
    cs_mean, cs_std = ms(cursed_shares)

    ordinal_agg = {
        "draw_boundary_rate_mean": float(np.mean([o["draw_boundary_rate"] for o in fold_ordinal])),
        "draw_boundary_rate_std":  float(np.std( [o["draw_boundary_rate"] for o in fold_ordinal])),
        "empirical_null_mean":     float(np.mean([o["empirical_null"]["null_mean"] for o in fold_ordinal])),
        "empirical_null_p95_mean": float(np.mean([o["empirical_null"]["null_p95"] for o in fold_ordinal])),
        "exceeds_p95_frac":        float(np.mean([o["empirical_null"]["exceeds_p95"] for o in fold_ordinal])),
        "p_value_mean":            float(np.mean([o["empirical_null"]["p_value"] for o in fold_ordinal])),
        "flat_null":               4 / 6,
        "win_loss_errors_mean":    float(np.mean([o["win_loss_errors"] for o in fold_ordinal])),
    }

    cursed_agg = {
        "cursed_share_mean":     cs_mean,
        "cursed_share_std":      cs_std,
        "total_cursed_mean":     float(np.mean([c["total_cursed"] for c in fold_cursed])),
        "total_genuine_mean":    float(np.mean([c["total_genuine"] for c in fold_cursed])),
        "type_b_mean":           float(np.mean([c["type_b_draw_to_other"] for c in fold_cursed])),
        "p_cursed_in_sample":    float(np.mean([c["p_cursed_in_sample"] for c in fold_cursed])),
    }

    return {
        "fidelity":     fidelity_agg,
        "ordinal":      ordinal_agg,
        "cursed":       cursed_agg,
        "fold_details": {"fidelity": fold_fidelity, "cursed_shares": cursed_shares},
    }

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _save_fidelity_bar(cv: dict, endgame: str, fig_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping bar chart.")
        return

    labels = ["WIN", "DRAW", "LOSS"]
    means  = [cv["fidelity"][l]["mean"] for l in labels]
    stds   = [cv["fidelity"][l]["std"]  for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=5, color="#4C72B0", width=0.5)
    ax.set_ylabel("Per-class fidelity (oracle agreement)")
    ax.set_title(f"{endgame.upper()} — Baseline fidelity "
                 f"({N_CV_FOLDS}-fold CV, geometric features only)")
    lo = min(means) - 0.015
    ax.set_ylim(max(0.0, lo), 1.005)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
    fig.tight_layout()
    path = os.path.join(fig_dir, f"fidelity_bar_{endgame}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {path}")

# ---------------------------------------------------------------------------
# Gate printer
# ---------------------------------------------------------------------------

def print_gate(endgame: str, cv: dict) -> None:
    fid = cv["fidelity"]
    cur = cv["cursed"]
    ord_ = cv["ordinal"]
    bar  = "=" * 66

    print(f"\n{bar}")
    print(f"  GATE NUMBERS — {endgame.upper()}  ({N_CV_FOLDS}-fold CV)")
    print(bar)
    print(f"  Baseline per-class fidelity (geometric features only):")
    print(f"    [1] WIN  : {fid['WIN']['mean']:.6f}  ± {fid['WIN']['std']:.6f}")
    print(f"    [2] DRAW : {fid['DRAW']['mean']:.6f}  ± {fid['DRAW']['std']:.6f}")
    print(f"    [3] LOSS : {fid['LOSS']['mean']:.6f}  ± {fid['LOSS']['std']:.6f}")
    print()
    print(f"  Cursed / blessed partition of draw-boundary errors:")
    print(f"    [4] Cursed share  : {cur['cursed_share_mean']:.4f}  ± {cur['cursed_share_std']:.4f}")
    print(f"        (cursed/blessed errors / total draw-boundary errors)")
    print(f"        Total cursed      : {cur['total_cursed_mean']:.0f}  per fold (mean)")
    print(f"        Genuine geometric : {cur['total_genuine_mean']:.0f}  per fold (mean)")
    print(f"        Type B (draw→win/loss): {cur['type_b_mean']:.0f}  per fold (mean)")
    print(f"    [5] p_cursed in sample: {cur['p_cursed_in_sample']:.4f}  "
          f"(fraction of positions that are cursed/blessed)")
    print()
    print(f"  [7] Draw-boundary rate vs empirical Monte Carlo null:")
    print(f"        Actual rate      = {ord_['draw_boundary_rate_mean']:.4f}  "
          f"± {ord_['draw_boundary_rate_std']:.4f}")
    print(f"        Null mean (2000) = {ord_['empirical_null_mean']:.4f}  "
          f"p95 = {ord_['empirical_null_p95_mean']:.4f}")
    print(f"        Flat null (ref)  = {ord_['flat_null']:.4f}")
    print(f"        Exceeds p95? {'YES' if ord_['exceeds_p95_frac'] > 0.5 else 'NO'}  "
          f"(p-value = {ord_['p_value_mean']:.4f})")
    print(f"        Win↔Loss errors  = {ord_['win_loss_errors_mean']:.0f}  (expected 0)")
    print(bar)
    print()
    print("  OUTCOME TEST (gate #8 requires cross-endgame comparison):")
    print("  Run all three endgames, then compare cursed_share:")
    print("  KBNK >> KRK  → gradient confirms clock as excluded mechanism")
    print("  KBNK ≈ KRK   → boundary confusion is not clock-driven (outcome B)")
    print(bar)

# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(endgame: str, syzygy_path: str = SYZYGY_PATH) -> dict:
    cfg = ENDGAME_CFG[endgame]
    ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{cfg['output_base']}/{ts}"
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load features ─────────────────────────────────────────────────────
    print(f"\n[{endgame.upper()}] Loading features...")
    df = pd.read_csv(cfg["features_csv"])
    print(f"  Positions: {len(df):,}")
    if "wdl" not in df.columns:
        raise ValueError(f"'wdl' column missing from {cfg['features_csv']}")

    # ── Load baseline model ────────────────────────────────────────────────
    print(f"[{endgame.upper()}] Loading baseline model...")
    model, feature_cols, model_path = load_baseline(cfg)
    print(f"  Model    : {model_path}")
    print(f"  Features : {len(feature_cols)}  (no DTZ — clean baseline)")

    y     = df["wdl"].astype(int).values
    X_df  = df.reindex(columns=feature_cols)
    X     = X_df.values.astype(float)

    # Sanity check on full dataset
    y_sanity = model.predict(X)
    fid_sanity = per_class_fidelity(y, y_sanity)
    print(f"  Sanity (full dataset): "
          f"WIN={fid_sanity['WIN']:.4f}  "
          f"DRAW={fid_sanity['DRAW']:.4f}  "
          f"LOSS={fid_sanity['LOSS']:.4f}")

    # ── Probe RAW WDL  (±1 granularity — label property, NOT a feature) ───
    print(f"[{endgame.upper()}] Probing raw WDL (±1 granularity)...")
    if "fen" not in df.columns:
        raise ValueError("No 'fen' column — cannot probe raw WDL.")
    raw_wdl = probe_raw_wdl_parallel(df["fen"].tolist(), syzygy_path=syzygy_path)
    n_cursed = int(((raw_wdl == 1) | (raw_wdl == -1)).sum())
    print(f"  Cursed wins   (raw= 1): {(raw_wdl == 1).sum():,}")
    print(f"  Blessed losses(raw=-1): {(raw_wdl == -1).sum():,}")
    print(f"  Total cursed/blessed  : {n_cursed:,}  "
          f"({n_cursed/len(raw_wdl)*100:.2f}% of sample)")

    # ── 5-fold CV — baseline only ─────────────────────────────────────────
    print(f"\n[{endgame.upper()}] Running {N_CV_FOLDS}-fold baseline CV...")
    cv = baseline_cv(X, y, raw_wdl, n_splits=N_CV_FOLDS)

    # ── Figure ────────────────────────────────────────────────────────────
    print(f"\n[{endgame.upper()}] Generating figure...")
    _save_fidelity_bar(cv, endgame, fig_dir)

    # ── Gate numbers ──────────────────────────────────────────────────────
    print_gate(endgame, cv)

    # ── Save JSON ─────────────────────────────────────────────────────────
    gate = {
        "endgame":    endgame,
        "timestamp":  ts,
        "model":      str(model_path),
        "n_positions": len(df),
        "n_cv_folds": N_CV_FOLDS,
        "sanity_fidelity": fid_sanity,
        "raw_wdl_summary": {
            "cursed_wins":      int((raw_wdl == 1).sum()),
            "blessed_losses":   int((raw_wdl == -1).sum()),
            "clean_wins":       int((raw_wdl == 2).sum()),
            "clean_losses":     int((raw_wdl == -2).sum()),
            "draws":            int((raw_wdl == 0).sum()),
        },
        "cv_results": cv,
        "gate_numbers": {
            "1_win_fidelity_mean":       cv["fidelity"]["WIN"]["mean"],
            "1_win_fidelity_std":        cv["fidelity"]["WIN"]["std"],
            "2_draw_fidelity_mean":      cv["fidelity"]["DRAW"]["mean"],
            "2_draw_fidelity_std":       cv["fidelity"]["DRAW"]["std"],
            "3_loss_fidelity_mean":      cv["fidelity"]["LOSS"]["mean"],
            "3_loss_fidelity_std":       cv["fidelity"]["LOSS"]["std"],
            "4_cursed_share_mean":       cv["cursed"]["cursed_share_mean"],
            "4_cursed_share_std":        cv["cursed"]["cursed_share_std"],
            "5_genuine_geometric_mean":  cv["cursed"]["total_genuine_mean"],
            "7_boundary_rate_mean":      cv["ordinal"]["draw_boundary_rate_mean"],
            "7_empirical_null_p95":      cv["ordinal"]["empirical_null_p95_mean"],
            "7_exceeds_p95":             cv["ordinal"]["exceeds_p95_frac"] > 0.5,
            "7_p_value_mean":            cv["ordinal"]["p_value_mean"],
        },
    }

    path = f"{out_dir}/gate_metrics.json"
    with open(path, "w") as f:
        json.dump(gate, f, indent=2)
    print(f"\nGate metrics saved: {path}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithfulness decomposition — cursed/blessed partition (v2)."
    )
    parser.add_argument(
        "--endgame",
        choices=list(ENDGAME_CFG.keys()),
        default="kbnk",
    )
    parser.add_argument(
        "--syzygy-path",
        default=SYZYGY_PATH,
    )
    args = parser.parse_args()
    run(args.endgame, syzygy_path=args.syzygy_path)


if __name__ == "__main__":
    main()
