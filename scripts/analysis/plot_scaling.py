#!/usr/bin/env python3
"""Plot generators for the P6 scaling study + MAC writeup.

Subcommands:
  strong  <csv>     -> strong scaling (time vs ranks, log-log) + ideal speedup
  weak    <csv>     -> weak scaling (time vs ranks, linear) + ideal flat
  comm    <csv>     -> stacked bar of solve time decomposition vs ranks
  mac     <csv>     -> per-step time-series (constraints / iters / rresid / t_solve)

Each subcommand also accepts -o / --out for the output png.

Expects solver_bench's `mode=mpi` rows (with t_alltoallv etc.) for strong/weak/comm,
and mac_sim's metrics.csv for mac.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# -------- strong scaling --------
def cmd_strong(args):
    df = _load(args.csv)
    df = df[df["mode"] == "mpi"].copy()
    if df.empty:
        sys.exit("no mpi rows in csv")

    fig, axes = plt.subplots(1, df["N"].nunique(), figsize=(6 * df["N"].nunique(), 5),
                              squeeze=False)
    axes = axes.flatten()

    for ax, (N, sub) in zip(axes, df.groupby("N")):
        for solver, ssub in sub.groupby("solver"):
            ssub = ssub.sort_values("ranks")
            ax.plot(ssub["ranks"], ssub["t_solve"], marker="o", label=str(solver))

        # Ideal speedup line: t_ideal(p) = t(1) / p, anchored at smallest rank.
        anchor = sub.sort_values("ranks").iloc[0]
        ranks = sorted(sub["ranks"].unique())
        ideal = [anchor["t_solve"] * anchor["ranks"] / r for r in ranks]
        ax.plot(ranks, ideal, "k--", alpha=0.4, label="ideal")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("solve time (s)")
        ax.set_title(f"strong scaling, N={N:,}")
        ax.grid(True, which="both", ls=":")
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


# -------- weak scaling --------
def cmd_weak(args):
    df = _load(args.csv)
    df = df[df["mode"] == "mpi"].copy()
    if df.empty:
        sys.exit("no mpi rows")

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver, sub in df.groupby("solver"):
        sub = sub.sort_values("ranks")
        ax.plot(sub["ranks"], sub["t_solve"], marker="s", label=str(solver))

    anchor = df.sort_values("ranks").iloc[0]
    ranks = sorted(df["ranks"].unique())
    ax.axhline(anchor["t_solve"], color="k", ls="--", alpha=0.4, label="ideal (flat)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("MPI ranks (N proportional to ranks)")
    ax.set_ylabel("solve time (s)")
    ax.set_title("weak scaling")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


# -------- comm-vs-compute breakdown --------
def cmd_comm(args):
    df = _load(args.csv)
    df = df[df["mode"] == "mpi"].copy()
    if df.empty:
        sys.exit("no mpi rows")

    # One panel per (solver, N) combo. Stacked bars of t_alltoallv / t_allreduce
    # / t_local_spmv / t_pack across rank counts.
    groups = list(df.groupby(["solver", "N"]))
    n = len(groups)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    parts = ["t_local_spmv", "t_alltoallv", "t_allreduce", "t_pack"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for ax, ((solver, N), sub) in zip(axes, groups):
        sub = sub.sort_values("ranks")
        x_labels = [str(r) for r in sub["ranks"]]
        x = list(range(len(x_labels)))
        bottom = [0.0] * len(x)
        for part, color in zip(parts, colors):
            vals = sub[part].fillna(0).tolist()
            ax.bar(x, vals, bottom=bottom, label=part, color=color)
            bottom = [b + v for b, v in zip(bottom, vals)]
        # Total t_solve overlay.
        ax.plot(x, sub["t_solve"].tolist(), "ko-", label="t_solve", markersize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("seconds")
        ax.set_title(f"{solver}, N={int(N):,}")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, axis="y", ls=":")

    for ax in axes[len(groups):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


# -------- mac time series --------
def cmd_mac(args):
    df = _load(args.csv)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(df["step"], df["n_constraints"], color="C0")
    axes[0].set_ylabel("# constraints")

    axes[1].plot(df["step"], df["iters"], color="C1")
    axes[1].set_ylabel("solver iters")

    axes[2].semilogy(df["step"], df["rresid"].clip(lower=1e-20), color="C2")
    axes[2].set_ylabel("rresid")

    axes[3].plot(df["step"], df["t_solve"], color="C3", alpha=0.8, label="t_solve")
    axes[3].plot(df["step"], df["t_assemble"], color="C4", alpha=0.8, label="t_assemble")
    axes[3].set_ylabel("seconds / step")
    axes[3].set_xlabel("step")
    axes[3].legend()

    # Mark binding events.
    bind_steps = df.loc[df["bind_this_step"] > 0, "step"]
    for ax in axes:
        for s in bind_steps:
            ax.axvline(s, color="grey", lw=0.3, alpha=0.4)
        ax.grid(True, ls=":")

    fig.suptitle(os.path.basename(os.path.dirname(args.csv)) or args.csv, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True, dest="cmd")
    for name, fn, default_out in [
        ("strong", cmd_strong, "strong_scaling.png"),
        ("weak",   cmd_weak,   "weak_scaling.png"),
        ("comm",   cmd_comm,   "comm_breakdown.png"),
        ("mac",    cmd_mac,    "mac_timeseries.png"),
    ]:
        s = sub.add_parser(name)
        s.add_argument("csv")
        s.add_argument("-o", "--out", default=default_out)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
