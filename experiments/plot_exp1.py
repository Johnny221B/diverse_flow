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

# =========================
# 字体：Times New Roman 优先 + 可靠回退
# =========================
def _pick_font(preferred_list):
    names = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    for want in preferred_list:
        # 既支持精确匹配，也支持包含式匹配
        for k, v in names.items():
            if want.lower() == k or want.lower() in k:
                return v
    return None

def _ensure_font(font_path=None, verbose=True):
    preferred = [
        "Times New Roman",   # Windows / macOS 常见
        "Times",             # macOS 别名
        "Nimbus Roman No9 L",# TeX/ghostscript 常见
        "Liberation Serif",  # 常见替代
        "DejaVu Serif",      # Matplotlib 自带
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

    picked = loaded_from_path or _pick_font(preferred)
    if picked is None:
        picked = mpl.rcParams.get("font.family", ["DejaVu Serif"])[0]

    mpl.rcParams["font.family"] = picked
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["axes.unicode_minus"] = False
    if verbose:
        print(f"[INFO] Using font family: {picked}")

# -------------------------
# 在 eval 目录里找 CSV
# -------------------------
def _try_csv_candidates(eval_dir: Path, concept: str, g_str: str):
    cand = []
    try:
        g_val = float(g_str)
        variants = list(dict.fromkeys([
            g_str,
            f"{g_val:g}",       # 3
            f"{g_val:.1f}",     # 3.0
            f"{g_val:.2f}",     # 3.00
        ]))
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

# -------------------------
# 解析输入来源（根目录/文件）
# -------------------------
def _resolve_sources(outputs_root: Path, methods_csv: str, files_csv: str, concept: str, labels_csv: str):
    files_or_roots, methods, labels = [], [], []

    if files_csv:
        toks = [t.strip() for t in files_csv.split(",") if t.strip()]
        files_or_roots = [Path(t) for t in toks]
        if methods_csv:
            methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
            if len(methods) != len(files_or_roots):
                print("[ERROR] 当使用 --files 时，若提供 --methods，二者数量必须一致。", file=sys.stderr)
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
            print("[ERROR] 未提供 --files 时，需要 --outputs_root 与 --methods", file=sys.stderr)
            sys.exit(1)
        methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
        files_or_roots = [outputs_root / f"{m}_{concept}" / "eval" for m in methods]
        labels = [s.strip() for s in labels_csv.split(",")] if labels_csv else methods[:]
        if len(labels) != len(files_or_roots):
            labels = methods[:]

    return files_or_roots, methods, labels

# -------------------------
# 三联图：每子图一个 guidance，子图内各自图例
# -------------------------
def plot_guidances_panel(
    files_or_roots,
    methods,
    labels,
    concept,
    guidances,          # list[str or float]
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
            leg_labels.append(f"{label} (AUC={summary['auc_prd_mu']:.3f})")

            summary_rows.append({
                "guidance": float(g),
                "method": method,
                "label": label,
                "auc_prd_mu": summary["auc_prd_mu"],
                "rec_at_p0.60_mu": summary["rec_at_p0.60_mu"],
                "rec_at_p0.70_mu": summary["rec_at_p0.70_mu"],
                "rec_at_p0.76_mu": summary["rec_at_p0.76_mu"],
                "source": str(csv_path),
            })

        ax.grid(True, ls="--", alpha=0.35)
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(f"guidance={float(g):.1f}", fontsize=11)

        if handles:
            leg = ax.legend(handles, leg_labels, loc=legend_loc, frameon=True, fontsize=9)
            for lh in leg.legend_handles:
                lh.set_linewidth(2)

    fig.supxlabel("Recall")
    fig.supylabel("Precision")
    ttl = f"PRD on '{concept}': method comparison under identical settings"
    if title_suffix:
        ttl += f" {title_suffix}"
    fig.suptitle(ttl, y=1.05, fontsize=13)
    fig.tight_layout()

    gs_tag = "_".join([f"{float(g):g}" for g in Gs])
    out_png = save_dir / f"prd_{concept}_panel_g{gs_tag}.png"
    out_pdf = save_dir / f"prd_{concept}_panel_g{gs_tag}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = save_dir / f"summary_{concept}_panel_g{gs_tag}.csv"
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
                    help="如 outputs/（会寻找 {method}_{concept}/eval/exp1_<g>_<concept>.csv）")
    ap.add_argument("--methods", type=str, default="",
                    help="逗号分隔的方法名（与 {method}_{concept} 前缀一致）")
    ap.add_argument("--files", type=str, default="",
                    help="逗号分隔的 CSV 路径或 eval 目录路径；与 --labels 一一对应")
    ap.add_argument("--labels", type=str, default="",
                    help="逗号分隔的图例名称，留空则用方法名/文件名")

    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--guidances", type=str, required=True,
                    help="逗号分隔，如 '3.0,5.0,7.5'")
    ap.add_argument("--save_dir", type=Path, default=None,
                    help="输出目录；默认 outputs_root/{concept}_exp1/plots 或 files 的上层/plots")
    ap.add_argument("--xlim", type=str, default="", help="如 '0,0.25'（可选）")
    ap.add_argument("--ylim", type=str, default="", help="如 '0.6,1.0'（可选）")
    ap.add_argument("--title_suffix", type=str, default="", help="标题后缀（可选）")
    ap.add_argument("--legend_loc", type=str, default="lower left",
                    help="每个子图图例位置，例如 'lower left' / 'upper right' 等")
    ap.add_argument("--font-path", type=str, default="",
                    help="Times New Roman 的 .ttf 路径（可选）")

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
