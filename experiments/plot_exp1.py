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

# -------------------------
# Font helpers (Times fallback)
# -------------------------
def _pick_font(preferred_list):
    names = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    for want in preferred_list:
        for k, v in names.items():
            if want.lower() == k or want.lower() in k:
                return v
    return None

def _ensure_font(font_path=None, verbose=True):
    preferred = [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    loaded_from_path = None
    if font_path:
        fpath = Path(font_path)
        if fpath.exists():
            try:
                fm.fontManager.addfont(str(fpath))
                prop = fm.FontProperties(fname=str(fpath))
                loaded_from_path = prop.get_name()
            except Exception as e:
                print(f"[WARN] failed to load font from --font-path: {e}", file=sys.stderr)

    picked = loaded_from_path or _pick_font(preferred) or mpl.rcParams.get("font.family", ["DejaVu Serif"])[0]
    mpl.rcParams["font.family"] = picked
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["axes.unicode_minus"] = False
    
    # 增加字体大小设置
    mpl.rcParams["font.size"] = 12        # 全局默认字体大小
    mpl.rcParams["axes.titlesize"] = 25   # 子图标题大小
    mpl.rcParams["axes.labelsize"] = 20   # 坐标轴标签大小
    mpl.rcParams["xtick.labelsize"] = 15  # x轴刻度标签大小
    mpl.rcParams["ytick.labelsize"] = 15  # y轴刻度标签大小
    mpl.rcParams["legend.fontsize"] = 15  # 图例字体大小
    
    if verbose:
        print(f"[INFO] Using font family: {picked}")

# -------------------------
# IO helpers
# -------------------------
def _try_csv_candidates(eval_dir: Path, concept: str, g_str: str):
    cand = []
    try:
        g_val = float(g_str)
        variants = list(dict.fromkeys([g_str, f"{g_val:g}", f"{g_val:.1f}", f"{g_val:.2f}"]))
    except:
        variants = [g_str]
    for v in variants:
        cand.append(eval_dir / f"exp1_{v}_{concept}.csv")
    for p in cand:
        if p.exists():
            return p
    for p in sorted(eval_dir.glob(f"exp1_*_{concept}.csv")):
        return p
    return None

def _load_one_csv(csv_path: Path, method: str, concept: str, guidance: float):
    df = pd.read_csv(csv_path)
    if "method" in df.columns:
        df = df[df["method"] == method]
    if "concept" in df.columns:
        df = df[df["concept"] == concept]
    if "guidance" in df.columns:
        df = df[(df["guidance"] - guidance).abs() < 1e-6]
    if df.empty:
        return None
    if "recall" in df.columns:
        df = df.sort_values(by=["recall"]).reset_index(drop=True)

    def _first(col, default=np.nan):
        return df[col].iloc[0] if col in df.columns and len(df[col]) else default

    summary = {
        "auc_prd_mu": _first("auc_prd_mu"),
        "rec_at_p0.60_mu": _first("rec_at_p0.60_mu"),
        "rec_at_p0.70_mu": _first("rec_at_p0.70_mu"),
        "rec_at_p0.76_mu": _first("rec_at_p0.76_mu"),
    }
    return df, summary

def _resolve_sources(outputs_root: Path, methods_csv: str, files_csv: str, concept: str, labels_csv: str):
    files_or_roots, methods, labels = [], [], []
    if files_csv:
        toks = [t.strip() for t in files_csv.split(",") if t.strip()]
        files_or_roots = [Path(t) for t in toks]
        if methods_csv:
            methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
            if len(methods) != len(files_or_roots):
                print("[ERROR] when using --files, the number of --methods must match.", file=sys.stderr)
                sys.exit(1)
        else:
            for p in files_or_roots:
                base = p.parent if p.is_file() else p
                par  = base.parent
                meth = par.name.split("_")[0]
                methods.append(meth)
        labels = [s.strip() for s in labels_csv.split(",")] if labels_csv else []
        if not labels or len(labels) != len(files_or_roots):
            labels = methods[:]
    else:
        if outputs_root is None or not methods_csv:
            print("[ERROR] without --files, you must provide --outputs_root and --methods", file=sys.stderr)
            sys.exit(1)
        methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
        files_or_roots = [outputs_root / f"{m}_{concept}" / "eval" for m in methods]
        labels = [s.strip() for s in labels_csv.split(",")] if labels_csv else methods[:]
        if len(labels) != len(files_or_roots):
            labels = methods[:]
    return files_or_roots, methods, labels

# -------------------------
# Label normalization for legend
# -------------------------
def _legend_label(label: str):
    lower = label.strip().lower()
    if lower == "pg": return "PG"
    if lower == "dpp": return "DPP"
    if lower == "cads": return "CADS"
    if lower == "ourmethod": return "OSCAR"
    return label

# -------------------------
# Plot
# -------------------------
def plot_guidances_panel(
    files_or_roots,
    methods,
    labels,
    concept,
    guidances,
    save_dir: Path,
    title_suffix="",
    xlim=None,
    ylim=None,
    legend_loc="lower left",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    Gs = [float(g) for g in guidances]
    n = len(Gs)

    base_colors = plt.get_cmap("tab10").colors
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(methods)}

    # 增加图形高度以适应更大的字体
    fig, axes = plt.subplots(1, n, figsize=(3.3*n + 2.0, 3.2), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)

    summary_rows = []

    for ax, g in zip(axes, Gs):
        handles, leg_labels = [], []

        for i, src in enumerate(files_or_roots):
            method = methods[i]
            label  = labels[i] if labels and i < len(labels) and labels[i] else method

            if src.is_dir():
                csv_path = _try_csv_candidates(src, concept, str(g))
                if csv_path is None:
                    print(f"[WARN] no CSV under {src} concept={concept}, g={g}", file=sys.stderr)
                    continue
            else:
                csv_path = src

            loaded = _load_one_csv(csv_path, method, concept, float(g))
            if loaded is None:
                print(f"[WARN] {csv_path} has no rows for method={method}, concept={concept}, g={g}", file=sys.stderr)
                continue
            df, summary = loaded

            x = df["recall"].values
            y = df["precision_mu"].values
            sd = df["precision_sd"].values if "precision_sd" in df.columns else None

            ln, = ax.plot(x, y, lw=2.2, color=color_map[method])
            if sd is not None:
                ax.fill_between(x, y - sd, y + sd, color=color_map[method], alpha=0.18, lw=0)

            handles.append(ln)
            leg_labels.append(f"{_legend_label(label)} (AUC={summary['auc_prd_mu']:.3f})")

            summary_rows.append({
                "guidance": float(g),
                "method": method,
                "label": _legend_label(label),
                "auc_prd_mu": summary["auc_prd_mu"],
                "rec_at_p0.60_mu": summary["rec_at_p0.60_mu"],
                "rec_at_p0.70_mu": summary["rec_at_p0.70_mu"],
                "rec_at_p0.76_mu": summary["rec_at_p0.76_mu"],
                "source": str(csv_path),
            })
            
        ax.set_yticks(np.arange(0.0, 1.0, 0.3))

        ax.grid(True, ls="--", alpha=0.35)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(f"CFG={float(g):.1f}", fontsize=18)  # 增加子图标题字号

        if handles:
            leg = ax.legend(handles, leg_labels, loc=legend_loc, frameon=True, fontsize=10)
            for lh in leg.legend_handles:
                lh.set_linewidth(2)

    fig.supxlabel("Recall", fontsize=15, y=0.08)  # 增加y值减小标签与图的距离
    fig.supylabel("Precision", fontsize=15, x=0.02)  # 增加x值减小标签与图的距离
    
    ttl = f"PRD in class-conditional task on Concept '{concept}'"
    if title_suffix:
        ttl += f" {title_suffix}"
    fig.suptitle(ttl, y=0.90, fontsize=20)  # 增加总标题字号
    
    # 增加tight_layout的padding以适应更大的字体
    fig.tight_layout(pad=1.0)

    gs_tag = "_".join([f"{float(g):g}" for g in Gs])
    out_png = save_dir / f"prd_{concept}_panel_cfg{gs_tag}.png"
    out_pdf = save_dir / f"prd_{concept}_panel_cfg{gs_tag}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = save_dir / f"summary_{concept}_panel_cfg{gs_tag}.csv"
        df_sum.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

# -------------------------
# CLI
# -------------------------
def _parse_pair(s):
    if not s: return None
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if len(toks) != 2: return None
    try:
        return (float(toks[0]), float(toks[1]))
    except:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=Path, default=None,
                    help="like outputs/; expects {method}_{concept}/eval/exp1_<g>_<concept>.csv")
    ap.add_argument("--methods", type=str, default="",
                    help="comma-separated method names (prefix in {method}_{concept})")
    ap.add_argument("--files", type=str, default="",
                    help="comma-separated CSV paths or eval dirs; pairs with --labels")
    ap.add_argument("--labels", type=str, default="",
                    help="comma-separated legend labels; default uses method names")

    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--guidances", type=str, required=True,
                    help="comma-separated, e.g. '3.0,5.0,7.5'")
    ap.add_argument("--save_dir", type=Path, default=None,
                    help="output dir; default is outputs_root/{concept}_exp1/plots or the parent of --files plus /plots")
    ap.add_argument("--xlim", type=str, default="", help="e.g. '0,0.25' (optional)")
    ap.add_argument("--ylim", type=str, default="", help="e.g. '0.6,1.0' (optional)")
    ap.add_argument("--title_suffix", type=str, default="", help="optional title suffix")
    ap.add_argument("--legend_loc", type=str, default="lower left",
                    help="legend location per subplot")
    ap.add_argument("--font-path", type=str, default="",
                    help="path to a .ttf for Times New Roman (optional)")

    args = ap.parse_args()
    _ensure_font(args.font_path, verbose=True)

    concept = args.concept
    guidances = [t.strip() for t in args.guidances.split(",") if t.strip()]

    files_or_roots, methods, labels = _resolve_sources(
        outputs_root=args.outputs_root,
        methods_csv=args.methods,
        files_csv=args.files,
        concept=concept,
        labels_csv=args.labels
    )

    if args.save_dir is None:
        if args.outputs_root is not None:
            save_dir = args.outputs_root / f"{concept}_exp1" / "plots"
        else:
            base = files_or_roots[0] if files_or_roots else Path("./")
            save_dir = (base.parent if base.is_file() else base) / "plots"
    else:
        save_dir = args.save_dir

    xlim = _parse_pair(args.xlim)
    ylim = _parse_pair(args.ylim)

    plot_guidances_panel(
        files_or_roots=files_or_roots,
        methods=methods,
        labels=labels,
        concept=concept,
        guidances=guidances,
        save_dir=save_dir,
        title_suffix=args.title_suffix,
        xlim=xlim,
        ylim=ylim,
        legend_loc=args.legend_loc,
    )

if __name__ == "__main__":
    main()