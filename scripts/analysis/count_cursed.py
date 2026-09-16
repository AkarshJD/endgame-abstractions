"""Count cursed-win (raw WDL=1) and blessed-loss (raw WDL=-1) positions
in the full KBNK dataset. Reports exact count and population base rate."""

import chess
import chess.syzygy
import math
import multiprocessing as mp
import pandas as pd

import argparse

SYZYGY = "storage/syzygy/3_4_5"
RAW_CSVS = {
    "kbnk": "data/raw/kbnk/full.csv",
    "krk":  "data/raw/krk/full.csv",
    "kpk":  "data/raw/kpk/full.csv",
}


def _worker(args):
    fens, path = args
    tb = chess.syzygy.open_tablebase(path)
    counts = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}
    for fen in fens:
        try:
            v = tb.probe_wdl(chess.Board(fen))
            counts[v] = counts.get(v, 0) + 1
        except Exception:
            pass
    tb.close()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endgame", choices=list(RAW_CSVS), default="kbnk")
    args = parser.parse_args()
    RAW_CSV = RAW_CSVS[args.endgame]
    print(f"Endgame: {args.endgame.upper()}  ({RAW_CSV})")
    print("Loading FENs...")
    df   = pd.read_csv(RAW_CSV, usecols=["fen"])
    fens = df["fen"].tolist()
    N    = len(fens)
    print(f"Total positions: {N:,}")

    n_workers  = max(1, mp.cpu_count() - 1)
    chunk_size = math.ceil(N / n_workers)
    chunks     = [fens[i : i + chunk_size] for i in range(0, N, chunk_size)]
    print(f"Probing with {n_workers} workers...")

    totals = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}
    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(
            pool.imap(_worker, [(c, SYZYGY) for c in chunks])
        ):
            for k, v in result.items():
                totals[k] = totals.get(k, 0) + v
            print(f"  chunk {i + 1}/{len(chunks)} done — "
                  f"cursed so far: {totals.get(1,0)+totals.get(-1,0):,}")

    labels = {
        2:  "Clean win",
        1:  "Cursed win  (clock-dependent)",
        0:  "Draw",
        -1: "Blessed loss (clock-dependent)",
        -2: "Clean loss",
    }
    print("\n=== KBNK Full Dataset WDL Distribution ===")
    for v in [2, 1, 0, -1, -2]:
        n = totals.get(v, 0)
        print(f"  {v:+d}  {labels[v]:38s}: {n:>10,}  ({n / N * 100:.6f}%)")

    cursed = totals.get(1, 0) + totals.get(-1, 0)
    print(f"\nTotal cursed/blessed : {cursed:,}")
    print(f"Population base rate : {cursed / N * 100:.6f}%  "
          f"(1 in {int(N / cursed) if cursed else 'inf'})")


if __name__ == "__main__":
    main()
