#!/usr/bin/env python3
"""Visualize a mac_sim trajectory.

Subcommands:
  snapshots <traj.xyz> -o snapshots.png         # multi-panel grid of frames
  anim       <traj.xyz> -o anim.gif             # animated GIF
  3d_snap    <traj.xyz> -o snap3d.png           # one big 3D snapshot of last frame

The XYZ format mac_sim emits:
    <n_bodies>
    step=K nbind=B
    C5b x y z
    C6  x y z
    ...

Bonds are inferred from center-to-center distance: any two bodies whose
distance is below `--bond_thresh` are drawn as a line. With unit-radius
bodies and a target weld-distance of 2 (each body presents its surface
to the other), bond_thresh=2.5 catches all welds.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)


KIND_COLORS = {
    "C5b": "#1f77b4",  # blue
    "C6":  "#2ca02c",  # green
    "C7":  "#ff7f0e",  # orange
    "C8":  "#9467bd",  # purple
    "C9":  "#d62728",  # red
}


def parse_xyz(path):
    """Yield (step, nbind, kinds, coords) tuples per frame."""
    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                return
            n = int(header.strip())
            comment = f.readline().strip()
            # comment looks like "step=K nbind=B"
            step = nbind = -1
            for part in comment.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k == "step":  step  = int(v)
                    if k == "nbind": nbind = int(v)
            kinds = []
            coords = np.zeros((n, 3), dtype=float)
            for i in range(n):
                tok = f.readline().split()
                kinds.append(tok[0])
                coords[i] = [float(tok[1]), float(tok[2]), float(tok[3])]
            yield step, nbind, kinds, coords


def find_bonds(coords, thresh):
    """Return list of (i, j) pairs with dist < thresh."""
    n = len(coords)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < thresh:
                bonds.append((i, j))
    return bonds


def draw_frame_3d(ax, kinds, coords, bonds, title=None, axis_lim=None):
    ax.clear()
    colors = [KIND_COLORS.get(k, "#888888") for k in kinds]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=120, edgecolors="k", linewidths=0.5,
               depthshade=True)
    for i, j in bonds:
        seg = coords[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="k", lw=1.4, alpha=0.7)
    if axis_lim is not None:
        ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_zlim(axis_lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=10)


def auto_axis_limits(frames, pad=1.0):
    all_coords = np.vstack([f[3] for f in frames])
    lo = all_coords.min() - pad
    hi = all_coords.max() + pad
    return (lo, hi)


def cmd_snapshots(args):
    frames = list(parse_xyz(args.xyz))
    if not frames:
        sys.exit("no frames")
    n_show = min(args.n_panels, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n_show).astype(int)
    selected = [frames[i] for i in idxs]
    axis_lim = auto_axis_limits(selected)

    cols = min(n_show, args.cols)
    rows = (n_show + cols - 1) // cols
    fig = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for k, (step, nbind, kinds, coords) in enumerate(selected):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        bonds = find_bonds(coords, args.bond_thresh)
        draw_frame_3d(ax, kinds, coords, bonds,
                      title=f"step={step}  bonds={nbind}",
                      axis_lim=axis_lim)
    # Legend (one for the whole figure).
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=10, label=k)
               for k, c in KIND_COLORS.items()]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=10)
    fig.suptitle(os.path.basename(args.xyz), fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


def cmd_anim(args):
    frames = list(parse_xyz(args.xyz))
    if not frames:
        sys.exit("no frames")
    axis_lim = auto_axis_limits(frames)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=10, label=k)
               for k, c in KIND_COLORS.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)

    def update(idx):
        step, nbind, kinds, coords = frames[idx]
        bonds = find_bonds(coords, args.bond_thresh)
        draw_frame_3d(ax, kinds, coords, bonds,
                      title=f"step={step}  bonds={nbind}",
                      axis_lim=axis_lim)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000.0 / args.fps, blit=False)
    if args.out.endswith(".gif"):
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    elif args.out.endswith(".mp4"):
        anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps))
    else:
        sys.exit("--out must be .gif or .mp4")
    print("wrote", args.out, f"({len(frames)} frames)")


def cmd_3d_snap(args):
    frames = list(parse_xyz(args.xyz))
    if not frames:
        sys.exit("no frames")
    last = frames[-1]
    axis_lim = auto_axis_limits(frames)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    step, nbind, kinds, coords = last
    bonds = find_bonds(coords, args.bond_thresh)
    draw_frame_3d(ax, kinds, coords, bonds,
                  title=f"final state — step={step}  bonds={nbind}",
                  axis_lim=axis_lim)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=10, label=k)
               for k, c in KIND_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshots", help="grid of evenly-spaced frame snapshots")
    s.add_argument("xyz")
    s.add_argument("-o", "--out", default="snapshots.png")
    s.add_argument("--n_panels", type=int, default=6)
    s.add_argument("--cols", type=int, default=3)
    s.add_argument("--bond_thresh", type=float, default=2.5)
    s.set_defaults(fn=cmd_snapshots)

    s = sub.add_parser("anim", help="animated 3D trajectory (gif/mp4)")
    s.add_argument("xyz")
    s.add_argument("-o", "--out", default="anim.gif")
    s.add_argument("--fps", type=int, default=8)
    s.add_argument("--bond_thresh", type=float, default=2.5)
    s.set_defaults(fn=cmd_anim)

    s = sub.add_parser("3d_snap", help="single 3D snapshot of final frame")
    s.add_argument("xyz")
    s.add_argument("-o", "--out", default="snap3d.png")
    s.add_argument("--bond_thresh", type=float, default=2.5)
    s.set_defaults(fn=cmd_3d_snap)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
