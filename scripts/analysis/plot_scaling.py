#!/usr/bin/env python3
"""Plot strong/weak scaling and solver comparison from solver_bench CSVs.

Usage:
  plot_scaling.py compare <csv> [<csv> ...] -o solver_compare.png
      Bar chart: time/solve and iters per solver, grouped by N.

  plot_scaling.py strong <csv> -o strong.png
      Time-vs-ranks line plot. CSV must include a 'ranks' column or the file
      naming convention strong_r<N>.csv (one rank count per file).

  plot_scaling.py weak <csv-glob> -o weak.png
      Time-vs-ranks where N grows w/ ranks.

CSV columns produced by solver_bench:
  tag,solver,N,nnz_per_row,alpha,rtol,max_iter,iters,converged,stagnated,diverged,rresid,err,t_gen,t_solve
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


def load_csvs(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        for p in glob.glob(path):
            df = pd.read_csv(p)
            df["__file"] = os.path.basename(p)
            m = re.search(r"_r(\d+)", df["__file"].iloc[0])
            if m:
                df["ranks"] = int(m.group(1))
            frames.append(df)
    if not frames:
        sys.exit("no CSVs matched")
    return pd.concat(frames, ignore_index=True)


def cmd_compare(args):
    df = load_csvs(args.csvs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for n_val, sub in df.groupby("N"):
        ax1.bar(sub["solver"] + f"\nN={n_val}", sub["t_solve"])
    ax1.set_ylabel("solve time (s)")
    ax1.set_yscale("log")
    ax1.tick_params(axis="x", rotation=30)

    for n_val, sub in df.groupby("N"):
        ax2.bar(sub["solver"] + f"\nN={n_val}", sub["iters"])
    ax2.set_ylabel("iterations")
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print("wrote", args.o)


def cmd_strong(args):
    df = load_csvs(args.csvs)
    if "ranks" not in df.columns:
        sys.exit("strong: CSVs lack 'ranks' column (fix filenames or add to CSV)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, sub in df.groupby("solver"):
        sub = sub.sort_values("ranks")
        ax.plot(sub["ranks"], sub["t_solve"], marker="o", label=solver)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("solve time (s)")
    ax.set_title("Strong scaling")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print("wrote", args.o)


def cmd_weak(args):
    df = load_csvs(args.csvs)
    if "ranks" not in df.columns:
        sys.exit("weak: CSVs lack 'ranks' column")
    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, sub in df.groupby("solver"):
        sub = sub.sort_values("ranks")
        ax.plot(sub["ranks"], sub["t_solve"], marker="s", label=solver)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks (N proportional)")
    ax.set_ylabel("solve time (s)")
    ax.set_title("Weak scaling")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.o, dpi=150)
    print("wrote", args.o)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True, dest="cmd")
    for name, fn in [("compare", cmd_compare), ("strong", cmd_strong), ("weak", cmd_weak)]:
        s = sub.add_parser(name)
        s.add_argument("csvs", nargs="+")
        s.add_argument("-o", default=f"{name}.png")
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
