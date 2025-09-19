#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

# ---------- Font (Times with fallback) ----------
def _pick_font(preferred_list):
    names = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    for want in preferred_list:
        for k, v in names.items():
            if want.lower() == k or want.lower() in k:
                return v
    return None

def _ensure_font(font_path=None):
    picked = None
    if font_path:
        fp = Path(font_path)
        if fp.exists():
            try:
                fm.fontManager.addfont(str(fp))
                picked = fm.FontProperties(fname=str(fp)).get_name()
            except Exception as e:
                print(f"[WARN] failed to load font from --font-path: {e}", file=sys.stderr)
    if picked is None:
        picked = _pick_font([
            "Times New Roman", "Times", "Nimbus Roman No9 L", "Liberation Serif", "DejaVu Serif"
        ]) or "DejaVu Serif"
    mpl.rcParams["font.family"]  = picked
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["axes.unicode_minus"] = False
    print(f"[INFO] Using font family: {picked}")

# ---------- Label normalization ----------
def _norm_method_label(s: str) -> str:
    lo = s.strip().lower()
    if lo == "pg": return "PG"
    if lo == "dpp": return "DPP"
    if lo == "cads": return "CADS"
    if lo == "ourmethod": return "Ourmethod"
    return s

# ---------- CSV I/O ----------
def _read_csv_norm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "entropy_sd" in df.columns and "entropy_std" not in df.columns:
        df = df.rename(columns={"entropy_sd": "entropy_std"})
    need = {"method","concept","guidance","entropy_mean"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["guidance"]     = df["guidance"].astype(float)
    df["entropy_mean"] = df["entropy_mean"].astype(float)
    df["entropy_std"]  = df.get("entropy_std", 0.0).astype(float)
    return df

def _build_entropy_csv(outputs_root: Path, method: str, concept: str) -> Path:
    return outputs_root / f"{method}_{concept}" / "eval" / f"exp3_entropy_{concept}.csv"

def _load_all(concept, methods, guidances, outputs_root, debug=False):
    outputs_root = Path(outputs_root)
    frames, missing_methods = [], []
    for m in methods:
        csv_path = _build_entropy_csv(outputs_root, m, concept)
        if not csv_path.is_file():
            missing_methods.append((m, str(csv_path))); continue
        df = _read_csv_norm(csv_path)
        df = df[df["concept"].astype(str).str.lower() == concept.lower()]
        if df.empty:
            if debug: print(f"[WARN] {csv_path} has no rows for concept={concept}")
            continue
        df["__src__"] = str(csv_path)
        frames.append(df)
    if missing_methods:
        for m, p in missing_methods:
            print(f"[WARN] CSV not found for method={m}: {p}", file=sys.stderr)
    if not frames:
        raise FileNotFoundError("No CSVs loaded. Check outputs_root/methods/concept.")

    df = pd.concat(frames, ignore_index=True)
    gids = [float(g) for g in guidances]
    df = df[df["guidance"].isin(set(gids))]
    if df.empty:
        raise ValueError("After filtering by guidances, no data remains.")
    df = (df.groupby(["method","concept","guidance"], as_index=False)
            .agg(entropy_mean=("entropy_mean","mean"),
                 entropy_std =("entropy_std","mean")))
    return df

# ---------- Facet bars ----------
def _facet_bars(
    df, concept, methods, guidances,
    title=None,
    show_error=True,
    annotate=True,
    annot_fmt="{:.3f}±{:.3f}",
    show_legend=True,
    legend_loc="upper right",
    annot_pad=0.02,
):
    methods_order   = [m for m in methods if m in set(df["method"])]
    guidances_order = [float(g) for g in guidances]
    n_g = len(guidances_order)

    fig, axes = plt.subplots(1, n_g, figsize=(3.0*n_g + 1.8, 3.2), sharey=True)
    if n_g == 1: axes = [axes]

    colors = plt.get_cmap("tab10").colors
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods_order)}
    legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=_norm_method_label(m)) for m in methods_order]

    global_top_needed = 0.0

    for ax, g in zip(axes, guidances_order):
        xs, ys, es, cols = [], [], [], []
        x_centers = np.arange(len(methods_order), dtype=float)

        for i, m in enumerate(methods_order):
            row = df[(df["method"]==m) & (df["guidance"]==g)]
            if len(row):
                mu = float(row["entropy_mean"].iloc[0])
                sd = float(row["entropy_std"].iloc[0])
                xs.append(x_centers[i]); ys.append(mu); es.append(sd); cols.append(color_map[m])
                global_top_needed = max(global_top_needed, mu + (sd if show_error else 0.0) + annot_pad + 0.015)

        ax.bar(xs, ys, color=cols, width=0.65)
        if show_error and len(xs):
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="black", elinewidth=1, capsize=3, capthick=1)

        if annotate and len(xs):
            for x, y, s in zip(xs, ys, es):
                y_text = y + (s if show_error else 0.0) + annot_pad
                ax.text(x, y_text, f"{y:.3f}", ha="center", va="bottom", fontsize=8, clip_on=False)

        ax.set_title(f"CFG={g:.1f}", fontsize=11)  # <-- changed here
        ax.set_xticks(x_centers)
        ax.set_xticklabels([_norm_method_label(m) for m in methods_order], rotation=0, ha="center")  # <-- changed here
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    y_top = max(1.08, global_top_needed + 0.01) if global_top_needed > 0 else 1.08
    for ax in axes: ax.set_ylim(0.0, min(1.12, y_top))

    axes[0].set_ylabel("Normalized Entropy ($H/\\log K$)")
    if title is None:
        title = f"Normalized Entropy for the '{concept}' Concept"
    fig.suptitle(title, y=0.94, fontsize=14)

    if show_legend and legend_handles:
        fig.legend(
            handles=legend_handles,
            loc=legend_loc,
            ncol=min(4, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.88),
            bbox_transform=fig.transFigure
        )

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig

# ---------- Heatmap ----------
def _heatmap(df, concept, methods, guidances, title=None, annotate=True, annot_fmt="{:.3f}±{:.3f}"):
    methods_order   = [m for m in methods if m in set(df["method"])]
    guidances_order = [float(g) for g in guidances]
    piv_mu = df.pivot(index="method", columns="guidance", values="entropy_mean").reindex(index=methods_order, columns=guidances_order)
    piv_sd = df.pivot(index="method", columns="guidance", values="entropy_std").reindex(index=methods_order, columns=guidances_order)

    fig, ax = plt.subplots(figsize=(1.6*len(guidances_order)+2.6, 0.55*len(methods_order)+2.6))
    im = ax.imshow(piv_mu.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(guidances_order)))
    ax.set_xticklabels([f"{g:.1f}" if abs(g-int(g))>1e-6 else f"{int(g)}" for g in guidances_order])
    ax.set_yticks(np.arange(len(methods_order)))
    ax.set_yticklabels([_norm_method_label(m) for m in methods_order])  # normalize names
    ax.set_xlabel("CFG")   # <-- changed here
    ax.set_ylabel("Method")

    if annotate:
        for i in range(len(methods_order)):
            for j in range(len(guidances_order)):
                mu = piv_mu.values[i, j]; sd = piv_sd.values[i, j]
                if np.isnan(mu): continue
                ax.text(j, i, annot_fmt.format(mu, sd),
                        ha="center", va="center",
                        color=("white" if mu>0.5 else "black"), fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Mean entropy", rotation=90)

    ax.set_title(title or f"Normalized entropy heatmap | concept={concept}", fontsize=12)
    fig.tight_layout()
    return fig

# ---------- Dumbbell ----------
def _dumbbell(df, concept, methods, guidances, title=None, show_error=True, annotate=True, annot_fmt="{:.3f}±{:.3f}"):
    methods_order   = [m for m in methods if m in set(df["method"])]
    guidances_order = [float(g) for g in guidances]
    colors = plt.get_cmap("tab10").colors
    g_colors = {g: colors[i % len(colors)] for i, g in enumerate(guidances_order)}
    markers  = ["o","s","^","D","v","P","X","*"]
    g_mark   = {g: markers[i % len(markers)] for i, g in enumerate(guidances_order)}
    y_pos = np.arange(len(methods_order), dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 0.7*len(methods_order)+2.8))
    for i, m in enumerate(methods_order):
        sub = df[df["method"]==m].set_index("guidance").reindex(guidances_order)
        xs = sub["entropy_mean"].astype(float).values
        es = sub["entropy_std"].astype(float).values
        xs_valid = [x for x in xs if not np.isnan(x)]
        if len(xs_valid) >= 2:
            ax.plot(xs, [y_pos[i]]*len(xs), lw=1.8, color="#666666", alpha=0.7, zorder=1)
        for g, x, sd in zip(guidances_order, xs, es):
            if np.isnan(x): continue
            ax.scatter(x, y_pos[i], color=g_colors[g], marker=g_mark[g], s=40, zorder=3)
            if show_error and sd>0:
                ax.errorbar(x, y_pos[i], xerr=sd, fmt="none", ecolor=g_colors[g],
                            elinewidth=1.6, capsize=3, capthick=1.6, zorder=2)
            if annotate:
                ax.text(x + (sd if show_error else 0) + 0.015, y_pos[i],
                        annot_fmt.format(x, sd), va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos); ax.set_yticklabels([_norm_method_label(m) for m in methods_order])  # normalize names
    ax.set_xlabel("Normalized entropy (H / log K)")
    ax.set_title(title or f"Per-method entropy (dumbbell) | concept={concept}")
    ax.grid(axis="x", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(0.0, 1.05)
    handles = [plt.Line2D([0],[0], color=g_colors[g], marker=g_mark[g], lw=0, label=f"CFG={g:.1f}") for g in guidances_order]  # <-- changed here
    ax.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    return fig

# ---------- API ----------
def plot_entropy_auto(
    concept: str,
    methods: list[str],
    guidances: list[float],
    outputs_root: str | Path = "/mnt/data/flow_grpo/flow_base/outputs",
    title: str | None = None,
    save_path: str | None = None,
    style: str = "facetbars",
    show_error: bool = True,
    annotate: bool = True,
    annot_fmt: str = "{:.3f}±{:.3f}",
    show_legend: bool = True,
    legend_loc: str = "upper center",
    annot_pad: float = 0.02,
    debug: bool = False,
):
    df = _load_all(concept, methods, guidances, outputs_root, debug=debug)

    if style == "facetbars":
        fig = _facet_bars(
            df, concept, methods, guidances,
            title=title, show_error=show_error, annotate=annotate,
            annot_fmt=annot_fmt, show_legend=show_legend,
            legend_loc=legend_loc, annot_pad=annot_pad
        )
    elif style == "heatmap":
        fig = _heatmap(df, concept, methods, guidances, title=title, annotate=annotate, annot_fmt=annot_fmt)
    elif style == "dumbbell":
        fig = _dumbbell(df, concept, methods, guidances, title=title, show_error=show_error, annotate=annotate, annot_fmt=annot_fmt)
    else:
        raise ValueError(f"Unknown style: {style}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"[SAVE] {save_path}")
        pdf_path = Path(save_path).with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[SAVE] {pdf_path}")
        plt.close(fig)
    else:
        plt.show()

# ---------- CLI ----------
def _parse_list(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def _parse_float_list(s: str) -> list[float]:
    return [float(t.strip()) for t in s.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept",     required=True, type=str)
    ap.add_argument("--methods",     required=True, type=str, help="comma-separated methods, e.g., dpp,pg,cads")
    ap.add_argument("--guidances",   required=True, type=str, help="comma-separated values, e.g., 3.0,5.0,7.5")
    ap.add_argument("--outputs_root",type=str, default="/mnt/data/flow_grpo/flow_base/outputs")
    ap.add_argument("--title",       type=str, default=None)
    ap.add_argument("--out",         type=str, default=None)
    ap.add_argument("--style",       type=str, default="facetbars", choices=["facetbars","heatmap","dumbbell"])
    ap.add_argument("--no_error",    action="store_true")
    ap.add_argument("--no_annot",    action="store_true")
    ap.add_argument("--no_legend",   action="store_true", help="hide global legend (facetbars)")
    ap.add_argument("--legend_loc",  type=str, default="upper center")
    ap.add_argument("--annot_pad",   type=float, default=0.02)
    ap.add_argument("--debug",       action="store_true")
    ap.add_argument("--font-path",   type=str, default="", help="Times New Roman .ttf path (optional)")
    args = ap.parse_args()

    _ensure_font(args.font_path)

    methods   = _parse_list(args.methods)
    guidances = _parse_float_list(args.guidances)

    plot_entropy_auto(
        concept=args.concept,
        methods=methods,
        guidances=guidances,
        outputs_root=args.outputs_root,
        title=args.title,
        save_path=args.out,
        style=args.style,
        show_error=not args.no_error,
        annotate=not args.no_annot,
        annot_fmt="{:.3f}±{:.3f}",
        show_legend=not args.no_legend,
        legend_loc=args.legend_loc,
        annot_pad=args.annot_pad,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
