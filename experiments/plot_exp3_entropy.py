#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grouped bar chart for normalized entropy (auto path version):
- Auto-load CSVs from: {outputs_root}/{method}_{concept}/eval/exp3_entropy_{concept}.csv
- Each method is a group; each guidance is a bar within the group
- Bars = entropy_mean; error bars = ±entropy_std (optional)

Usage example:
python plot_entropy_bars_auto.py \
  --concept truck \
  --methods dpp,pg,cads \
  --guidances 3.0,5.0,7.5 \
  --outputs_root /mnt/data/flow_grpo/flow_base/outputs \
  --out /tmp/entropy_truck.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _read_csv_norm(path: Path) -> pd.DataFrame:
    """Read a CSV and normalize column names & types."""
    df = pd.read_csv(path)
    # columns -> lowercase+strip
    df.columns = [c.strip().lower() for c in df.columns]
    # alias entropy_sd -> entropy_std
    if "entropy_sd" in df.columns and "entropy_std" not in df.columns:
        df = df.rename(columns={"entropy_sd": "entropy_std"})
    # ensure required columns
    need = {"method","concept","guidance","entropy_mean"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    # types
    df["guidance"] = df["guidance"].astype(float)
    df["entropy_mean"] = df["entropy_mean"].astype(float)
    if "entropy_std" in df.columns:
        df["entropy_std"] = df["entropy_std"].astype(float)
    else:
        df["entropy_std"] = 0.0
    return df

def _build_entropy_csv(outputs_root: Path, method: str, concept: str) -> Path:
    return outputs_root / f"{method}_{concept}" / "eval" / f"exp3_entropy_{concept}.csv"

def plot_entropy_bars_auto(
    concept: str,
    methods: list[str],
    guidances: list[float],
    outputs_root: str | Path = "/mnt/data/flow_grpo/flow_base/outputs",
    title: str | None = None,
    save_path: str | None = None,
    bar_width: float = 0.22,
    show_error: bool = True,
    annotate: bool = True,
    annot_fmt: str = "{:.3f}±{:.3f}",
    debug: bool = False,
):
    outputs_root = Path(outputs_root)
    if debug:
        print(f"[DEBUG] outputs_root = {outputs_root}")

    # 读取每个 method 的单一 CSV
    frames = []
    missing_methods = []
    for m in methods:
        csv_path = _build_entropy_csv(outputs_root, m, concept)
        if not csv_path.is_file():
            missing_methods.append((m, str(csv_path)))
            continue
        df = _read_csv_norm(csv_path)
        # 仅保留该 concept（以防 CSV 里混了别的 concept）
        df = df[df["concept"].astype(str).str.lower() == concept.lower()]
        if df.empty:
            if debug:
                print(f"[WARN] {csv_path} has no rows for concept={concept}")
            continue
        df["__src__"] = str(csv_path)
        frames.append(df)

    if missing_methods:
        for m, p in missing_methods:
            print(f"[WARN] CSV not found for method={m}: {p}")

    if not frames:
        raise FileNotFoundError("No CSVs loaded. Check outputs_root/methods/concept.")

    df = pd.concat(frames, ignore_index=True)

    # 只保留目标 guidances；严格按用户给的顺序画
    gset = set(float(g) for g in guidances)
    df = df[df["guidance"].isin(gset)]
    if df.empty:
        raise ValueError("After filtering by guidances, no data remains.")

    # 为稳妥起见：若同一 (method,guidance) 多行，聚合为均值
    df = (df.groupby(["method","concept","guidance"], as_index=False)
            .agg(entropy_mean=("entropy_mean","mean"),
                 entropy_std =("entropy_std","mean")))

    # 布局
    methods_order = [m for m in methods if m in set(df["method"])]
    guidances_order = [float(g) for g in guidances]

    n_m = len(methods_order)
    n_g = len(guidances_order)
    if n_m == 0:
        raise ValueError("None of the specified methods appear in the data.")
    x_centers = np.arange(n_m, dtype=float)
    total_width = n_g * bar_width
    start = -0.5 * total_width + 0.5 * bar_width
    offsets = [start + i * bar_width for i in range(n_g)]

    fig, ax = plt.subplots(figsize=(max(6, 1.8*n_m), 4.8), dpi=140)

    # 逐 guidance 画柱
    for gi, g in enumerate(guidances_order):
        xs_all, ys_all, es_all = [], [], []
        missing_pairs = []
        for mi, m in enumerate(methods_order):
            sub = df[(df["method"] == m) & (df["guidance"] == float(g))]
            x_pos = x_centers[mi] + offsets[gi]
            if sub.empty:
                missing_pairs.append((m, g))
                continue
            ent_mu = float(sub["entropy_mean"].iloc[0])
            ent_sd = float(sub["entropy_std"].iloc[0]) if "entropy_std" in sub.columns else 0.0
            xs_all.append(x_pos); ys_all.append(ent_mu); es_all.append(ent_sd)

        if missing_pairs and debug:
            for m,gx in missing_pairs:
                print(f"[WARN] missing combo -> method={m}, guidance={gx}")

        if not xs_all:
            # 该 guidance 完全无数据，跳过
            continue

        bars = ax.bar(xs_all, ys_all, width=bar_width, label=f"guidance={g}")
        if show_error:
            ax.errorbar(xs_all, ys_all, yerr=es_all, fmt="none",
                        ecolor="black", elinewidth=1, capsize=3, capthick=1)
        if annotate:
            for x, y, s in zip(xs_all, ys_all, es_all):
                ax.text(x, y + 0.02, annot_fmt.format(y, s),
                        ha="center", va="bottom", fontsize=9)

    # 坐标轴与样式
    ax.set_xticks(x_centers)
    ax.set_xticklabels(methods_order)
    ax.set_xlim(-0.6, n_m - 0.4)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("normalized entropy (H / log K), ±1 SD")
    ax.set_ylim(0.0, 1.08)
    if title is None:
        title = f"Entropy per method | concept={concept}"
    ax.set_title(title)
    ax.legend(frameon=False, ncols=min(n_g, 3))
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
        plt.close(fig)
    else:
        plt.show()

def _parse_list(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def _parse_float_list(s: str) -> list[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, type=str, help="concept name, e.g., truck")
    ap.add_argument("--methods", required=True, type=str, help="comma-separated methods, e.g., dpp,pg,cads")
    ap.add_argument("--guidances", required=True, type=str, help="comma-separated guidance values, e.g., 3.0,5.0,7.5")
    ap.add_argument("--outputs_root", type=str, default="/mnt/data/flow_grpo/flow_base/outputs")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--bar_width", type=float, default=0.22)
    ap.add_argument("--no_error", action="store_true")
    ap.add_argument("--no_annot", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    methods = _parse_list(args.methods)
    guidances = _parse_float_list(args.guidances)

    plot_entropy_bars_auto(
        concept=args.concept,
        methods=methods,
        guidances=guidances,
        outputs_root=args.outputs_root,
        title=args.title,
        save_path=args.out,
        bar_width=args.bar_width,
        show_error=not args.no_error,
        annotate=not args.no_annot,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()