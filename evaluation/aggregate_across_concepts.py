#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate metrics across multiple concepts for each method and CFG (guidance).

输入:
  - 若干个 CSV, 路径形如:
      outputs/{method}_{concept}/eval/{method}_{concept}.csv
    每个 CSV 的列类似:
      method,concept,folder,prompt,seed,guidance,steps,num_images,
      vendi_pixel,vendi_inception,fid,fid_clean,kid_mean,kid_std,
      clip_score,one_minus_ms_ssim,brisque

输出:
  - 一个聚合后的 CSV (默认: aggregated_metrics.csv)
    每一行对应 (method, guidance) 一对，在所有 concept 和 seed 上做平均。
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=str, default="outputs",
                    help="根目录，里面包含 {method}_{concept}/eval/")
    ap.add_argument("--methods", type=str, required=True,
                    help="逗号分隔的方法名，例如 'dpp,pg,cads,oscar'")
    ap.add_argument("--concepts", type=str, required=True,
                    help="逗号分隔的 concept 名，例如 'apple,truck,bus,bicycle'")
    ap.add_argument("--guidances", type=str, default="3.0,5.0,7.5",
                    help="需要保留的 CFG/guidance 值，逗号分隔")
    ap.add_argument("--out_csv", type=str, default="aggregated_metrics.csv",
                    help="聚合结果输出路径")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    methods   = [m.strip() for m in args.methods.split(",") if m.strip()]
    concepts  = [c.strip() for c in args.concepts.split(",") if c.strip()]
    guidances = [float(g) for g in args.guidances.split(",") if g.strip()]

    all_dfs = []

    for method in methods:
        for concept in concepts:
            csv_path = outputs_root / f"{method}_{concept}" / "eval" / f"{method}_{concept}.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing CSV: {csv_path}, 跳过")
                continue
            print(f"[LOAD] {csv_path}")
            df = pd.read_csv(csv_path)

            # 确保 guidance 是 float
            if "guidance" in df.columns:
                df["guidance"] = df["guidance"].astype(float)

            # 只保留感兴趣的 guidance
            df = df[df["guidance"].isin(guidances)]
            if df.empty:
                print(f"[WARN] {csv_path} 在指定 guidance {guidances} 下没有数据，跳过")
                continue

            all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] 没有任何有效的 CSV 被加载到，检查路径/方法/类别是否正确。")
        return

    big_df = pd.concat(all_dfs, ignore_index=True)

    # 需要聚合的指标: 除去这些“元信息”列，剩下的都做平均
    meta_cols = {
        "method", "concept", "folder", "prompt",
        "seed", "guidance", "steps", "num_images"
    }
    metric_cols = [c for c in big_df.columns if c not in meta_cols]

    # 有些列可能是空字符串，先转成数值 (无法转换的变成 NaN)
    for c in metric_cols:
        big_df[c] = pd.to_numeric(big_df[c], errors="coerce")

    # 按 (method, guidance) 聚合，跨 concept+seed 求平均
    grouped = (
        big_df
        .groupby(["method", "guidance"])[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["method", "guidance"])
    )

    out_csv = Path(args.out_csv)
    grouped.to_csv(out_csv, index=False)
    print(f"[SAVE] Aggregated metrics -> {out_csv}")
    print(grouped)

if __name__ == "__main__":
    main()
