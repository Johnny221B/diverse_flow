#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt

def _try_csv_candidates(eval_dir: Path, concept: str, g_str: str):
    """
    为了兼容 3 / 3.0 / 3.00 等命名差异，这里尝试多种字符串。
    返回第一个存在的 CSV 路径，否则 None。
    """
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
    # 最后兜底：扫描该目录所有 exp1_*_{concept}.csv
    for p in sorted(eval_dir.glob(f"exp1_*_{concept}.csv")):
        return p
    return None

def _load_one_csv(csv_path: Path, method: str, concept: str, guidance: float):
    df = pd.read_csv(csv_path)
    # 过滤：方法、concept、guidance
    if "method" in df.columns:
        df = df[df["method"] == method]
    if "concept" in df.columns:
        df = df[df["concept"] == concept]
    if "guidance" in df.columns:
        # 容忍少量浮点误差
        df = df[ (df["guidance"] - guidance).abs() < 1e-6 ]
    if df.empty:
        return None

    # 排序
    if "recall" in df.columns:
        df = df.sort_values(by=["recall"]).reset_index(drop=True)

    # 抽取一次性标量
    def _first(col, default=None):
        return df[col].iloc[0] if col in df.columns and len(df[col]) else default

    summary = {
        "auc_prd_mu": _first("auc_prd_mu"),
        "rec_at_p0.60_mu": _first("rec_at_p0.60_mu"),
        "rec_at_p0.70_mu": _first("rec_at_p0.70_mu"),
        "rec_at_p0.76_mu": _first("rec_at_p0.76_mu"),
    }
    return df, summary

def plot_guidance(
    files_or_roots,
    methods,
    labels,
    concept,
    guidance,
    save_dir: Path,
    title_suffix="",
):
    """
    files_or_roots: list[Path]，要么是 csv 文件，要么是 eval 目录（将自动找 exp1_* 文件）
    methods: 与 files_or_roots 对应的方法名（或从 CSV 内 method 列读取）
    labels : 图例显示名（可与 methods 相同）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    summaries = []

    for i, src in enumerate(files_or_roots):
        method = methods[i]
        label = labels[i] if labels and i < len(labels) and labels[i] else method

        if src.is_dir():
            csv_path = _try_csv_candidates(src, concept, str(guidance))
            if csv_path is None:
                print(f"[WARN] no CSV found under {src} for concept={concept}, guidance={guidance}", file=sys.stderr)
                continue
        else:
            csv_path = src

        loaded = _load_one_csv(csv_path, method, concept, float(guidance))
        if loaded is None:
            print(f"[WARN] {csv_path} has no rows for method={method}, concept={concept}, guidance={guidance}", file=sys.stderr)
            continue
        df, summary = loaded

        # 画线 + 阴影
        x = df["recall"].values
        y = df["precision_mu"].values
        sd = df["precision_sd"].values if "precision_sd" in df.columns else None

        line, = ax.plot(x, y, lw=2, label=f"{label} (AUC={summary['auc_prd_mu']:.3f})")
        if sd is not None:
            ax.fill_between(x, y - sd, y + sd, alpha=0.2)

        summaries.append((label, summary, csv_path))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.4)

    g_str = f"{float(guidance):g}"
    ax.set_title(f"PRD – {concept} – guidance={g_str}" + (f" {title_suffix}" if title_suffix else ""))

    leg = ax.legend(loc="lower left", frameon=True)
    for lh in leg.legend_handles:
        lh.set_linewidth(2)

    out_png = save_dir / f"prd_{concept}_g{g_str}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[SAVE] {out_png}")

    # 同时输出一个小汇总表，便于对比数值
    summary_rows = []
    for label, s, p in summaries:
        summary_rows.append({
            "label": label,
            "auc_prd_mu": s["auc_prd_mu"],
            "rec_at_p0.60_mu": s["rec_at_p0.60_mu"],
            "rec_at_p0.70_mu": s["rec_at_p0.70_mu"],
            "rec_at_p0.76_mu": s["rec_at_p0.76_mu"],
            "source": str(p),
        })
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = save_dir / f"summary_{concept}_g{g_str}.csv"
        df_sum.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    # 方式A：给根目录 + methods + concept + guidance
    ap.add_argument("--outputs_root", type=Path, default=None,
                    help="形如 outputs/ 的根目录（将寻找 {method}_{concept}/eval/exp1_<g>_<concept>.csv）")
    ap.add_argument("--methods", type=str, default="",
                    help="逗号分隔的方法名（与 {method}_{concept} 前缀一致）")
    # 方式B：直接给 CSV 或 eval 目录（逗号分隔）
    ap.add_argument("--files", type=str, default="",
                    help="逗号分隔的 CSV 路径或 eval 目录路径；与 --labels 一一对应")
    ap.add_argument("--labels", type=str, default="",
                    help="逗号分隔的图例名称，留空则用方法名/文件名")

    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--guidance", type=str, required=True, help="如 3.0 或 5 或 7.5（作为字符串更稳）")
    ap.add_argument("--save_dir", type=Path, default=None, help="输出图/汇总表目录；默认 outputs_root/{concept}_exp1/plots")

    args = ap.parse_args()

    concept = args.concept
    g_str = args.guidance

    files_or_roots = []
    methods = []
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else []

    # 优先使用 --files；否则用 --outputs_root + --methods
    if args.files:
        for tok in args.files.split(","):
            tok = tok.strip()
            if not tok: continue
            p = Path(tok)
            files_or_roots.append(p)
            # method 名：优先 labels，否则用目录/文件名推断一个简短名
            if args.methods:
                # 若也传了 methods，则按 methods 顺序绑定
                pass
        # methods 若没给，就从文件/目录名生成一个短标签
        if not args.methods:
            for p in files_or_roots:
                methods.append(p.parent.parent.name.split("_")[0] if p.is_file() else p.parent.name.split("_")[0])
        else:
            methods = [m.strip() for m in args.methods.split(",") if m.strip()]
            if len(methods) != len(files_or_roots):
                print("[ERROR] 当使用 --files 时，若提供 --methods，二者数量必须一致。", file=sys.stderr)
                sys.exit(1)
        # labels 若不足，补成 methods
        if not labels or len(labels) != len(files_or_roots):
            labels = methods[:]
    else:
        # 用 outputs_root + methods + concept 自动找 CSV
        assert args.outputs_root is not None and args.methods, \
            "未提供 --files 时，需要 --outputs_root 与 --methods"
        outputs_root = args.outputs_root
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        if not methods:
            print("[ERROR] 空的 --methods", file=sys.stderr)
            sys.exit(1)
        for m in methods:
            eval_dir = outputs_root / f"{m}_{concept}" / "eval"
            files_or_roots.append(eval_dir)
        if not labels or len(labels) != len(files_or_roots):
            labels = methods[:]

    # save_dir
    if args.save_dir is None:
        if args.outputs_root is not None:
            save_dir = args.outputs_root / f"{concept}_exp1" / "plots"
        else:
            # 用第一个文件/目录附近
            base = files_or_roots[0] if files_or_roots else Path("./")
            save_dir = (base.parent if base.is_file() else base) / "plots"
    else:
        save_dir = args.save_dir

    plot_guidance(
        files_or_roots=files_or_roots,
        methods=methods,
        labels=labels,
        concept=concept,
        guidance=g_str,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main()