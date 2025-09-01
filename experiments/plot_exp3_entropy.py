#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grouped bar chart for normalized entropy:
- Each method is a group (gap between groups)
- Each guidance is one bar within the group
- Bars = mean; error bars = ±1 SD; optional "μ±σ" annotations
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_entropy_bars(
    csv_paths,
    concept: str = None,
    guidance_order = None,   # e.g., [3.0, 5.0, 7.5]
    method_order = None,     # e.g., ["ours", "baselineA", "baselineB"]
    title: str = None,
    save_path: str = None,
    bar_width: float = 0.22,
    show_error: bool = True,
    annotate: bool = True,
    annot_fmt: str = "{:.3f}±{:.3f}",
    debug: bool = False,
):
    # 1) load & merge
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["__src__"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # 2) normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # allow entropy_std or entropy_sd
    if "entropy_std" not in cols and "entropy_sd" in cols:
        df = df.rename(columns={cols["entropy_sd"]: "entropy_std"})
        cols["entropy_std"] = "entropy_std"
    # basic sanity
    need = ["method","concept","guidance","entropy_mean"]
    for k in need:
        if k not in cols:
            raise ValueError(f"CSV missing required column: {k}")
    # coerce types
    df["guidance"] = df[cols["guidance"]].astype(float)
    df["entropy_mean"] = df[cols["entropy_mean"]].astype(float)
    if "entropy_std" in cols or "entropy_std" in df.columns:
        df["entropy_std"] = df["entropy_std"].astype(float)
    else:
        df["entropy_std"] = 0.0

    # 3) optional filter
    if concept is not None:
        df = df[df[cols["concept"]].astype(str).str.lower() == str(concept).lower()]
    if df.empty:
        raise ValueError("No data after filtering; check concept/guidance or CSV contents.")

    # 4) choose orders
    methods = sorted(df[cols["method"]].unique().tolist()) if method_order is None else [m for m in method_order if m in set(df[cols["method"]])]
    guidances = sorted(df["guidance"].unique().tolist()) if guidance_order is None else [float(g) for g in guidance_order]

    # debug prints
    if debug:
        print("[DEBUG] methods in data:", sorted(df[cols["method"]].unique()))
        print("[DEBUG] guidances in data:", sorted(df["guidance"].unique()))
        print("[DEBUG] concept in data:", sorted(df[cols["concept"]].unique()))
        # check duplicates
        dup = df.duplicated(subset=[cols["method"], "guidance"])
        if dup.any():
            print("[WARN] duplicated rows for (method,guidance):")
            print(df.loc[dup, [cols["method"], "guidance", "__src__"]])

    # 5) prepare layout
    n_m = len(methods)
    n_g = len(guidances)
    x_centers = np.arange(n_m, dtype=float)
    total_width = n_g * bar_width
    start = -0.5 * total_width + 0.5 * bar_width
    offsets = [start + i * bar_width for i in range(n_g)]

    fig, ax = plt.subplots(figsize=(max(6, 1.8*n_m), 4.8), dpi=140)

    # 6) draw bars by guidance
    for gi, g in enumerate(guidances):
        xs_all, ys_all, es_all = [], [], []
        missing_pairs = []
        for mi, m in enumerate(methods):
            sub = df[(df[cols["method"]] == m) & (df["guidance"] == float(g))]
            x_pos = x_centers[mi] + offsets[gi]
            if sub.empty:
                missing_pairs.append((m, g))
                continue
            # 如果有多行（不应发生），取均值
            ent_mu = float(sub["entropy_mean"].mean())
            ent_sd = float(sub["entropy_std"].mean()) if "entropy_std" in sub.columns else 0.0
            xs_all.append(x_pos); ys_all.append(ent_mu); es_all.append(ent_sd)

        if missing_pairs and debug:
            for m,gx in missing_pairs:
                print(f"[WARN] missing combo -> method={m}, guidance={gx}")

        if not xs_all:
            # 该 guidance 完全没有数据
            continue

        bars = ax.bar(xs_all, ys_all, width=bar_width, label=f"guidance={g}")
        if show_error:
            ax.errorbar(xs_all, ys_all, yerr=es_all, fmt="none",
                        ecolor="black", elinewidth=1, capsize=3, capthick=1)
        if annotate:
            for x, y, s in zip(xs_all, ys_all, es_all):
                ax.text(x, y + 0.02, annot_fmt.format(y, s),
                        ha="center", va="bottom", fontsize=9)

    # 7) axes
    ax.set_xticks(x_centers)
    ax.set_xticklabels(methods)
    ax.set_xlim(-0.6, n_m - 0.4)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("normalized entropy (H / log K), ±1 SD")
    ax.set_ylim(0.0, 1.08)
    if title is None:
        title = "Entropy per method" + (f" | concept={concept}" if concept else "")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=min(len(guidances), 3))
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
        plt.close(fig)
    else:
        plt.show()

def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="one or more exp3_entropy_{concept}.csv paths")
    ap.add_argument("--concept", type=str, default=None, help="optional concept filter")
    ap.add_argument("--guidances", type=str, default=None, help="comma list order, e.g., '3.0,5.0,7.5'")
    ap.add_argument("--methods", type=str, default=None, help="comma list order")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_error", action="store_true")
    ap.add_argument("--no_annot", action="store_true")
    ap.add_argument("--bar_width", type=float, default=0.22)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    guidance_order = [float(x) for x in args.guidances.split(",")] if args.guidances else None
    method_order = [s.strip() for s in args.methods.split(",")] if args.methods else None

    plot_entropy_bars(
        csv_paths=args.csv,
        concept=args.concept,
        guidance_order=guidance_order,
        method_order=method_order,
        title=args.title,
        save_path=args.out,
        show_error=not args.no_error,
        annotate=not args.no_annot,
        bar_width=args.bar_width,
        debug=args.debug,
    )

if __name__ == "__main__":
    _main()