#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize per-(method, concept, guidance) eval CSVs into a single results file.

Input layout:
  outputs/{method}_{concept}/eval/{method}_{concept}.csv

Each CSV has columns like:
  method,concept,folder,prompt,seed,guidance,steps,num_images,
  vendi_pixel,vendi_inception,
  fid,fid_clean,kid_mean,kid_std,
  clip_score,one_minus_ms_ssim,brisque

This script:
  - loops over methods × concepts
  - for each existing CSV, groups by 'guidance'
  - computes mean and 95% CI for each metric within each guidance group
  - writes a single summary CSV:
        outputs/results/summary_metrics.csv
    with one row per (method, concept, guidance).
"""

import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd


# 你要汇总的所有指标列名
METRICS = [
    "vendi_pixel",
    "vendi_inception",
    "fid",
    "fid_clean",
    "kid_mean",
    "kid_std",
    "clip_score",
    "one_minus_ms_ssim",
    "brisque",
]


def parse_list(s_or_list) -> List[str]:
    """支持 'a,b,c' 或 ['a','b','c'] 两种形式."""
    if s_or_list is None:
        return []
    if isinstance(s_or_list, (list, tuple)):
        return [str(x) for x in s_or_list]
    s = str(s_or_list)
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def compute_mean_ci(series: pd.Series):
    """Return (mean, 95% CI) for a numeric series, ignoring NaNs."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mean = x.mean()
    if n == 1:
        # 只有一个样本，CI 记为 0
        return mean, 0.0
    std = x.std(ddof=1)  # sample std
    ci = 1.96 * std / np.sqrt(n)
    return float(mean), float(ci)


def summarize_one_csv(csv_path: str, method: str, concept: str) -> List[Dict]:
    """
    读取单个 method_concept 的 CSV，
    按 guidance 分组，每个 guidance 返回一行 summary dict。
    """
    df = pd.read_csv(csv_path)

    if "guidance" not in df.columns:
        raise ValueError(f"'guidance' column not found in {csv_path}")

    rows = []
    # 按 guidance 分组
    for gval, gdf in df.groupby("guidance"):
        row: Dict = {
            "method": method,
            "concept": concept,
            "guidance": gval,
            "n_rows": len(gdf),
        }
        for m in METRICS:
            if m not in gdf.columns:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_ci95"] = np.nan
                continue
            mean, ci = compute_mean_ci(gdf[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_ci95"] = ci
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize (method, concept, guidance) eval CSVs into outputs/results/summary_metrics.csv"
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        required=True,
        help='List of methods, e.g. "modified standardFM pg dpp" or "modified,standardFM,pg,dpp"',
    )
    parser.add_argument(
        "--concepts",
        nargs="*",
        required=True,
        help='List of concepts, e.g. "bowl dog car" or "bowl,dog,car"',
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Root outputs directory (default: ./outputs)",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary_metrics.csv",
        help="Summary CSV file name (default: summary_metrics.csv)",
    )
    args = parser.parse_args()

    methods = parse_list(args.methods)
    concepts = parse_list(args.concepts)
    outputs_root = os.path.abspath(args.outputs_root)

    print(f"[INFO] methods  : {methods}")
    print(f"[INFO] concepts : {concepts}")
    print(f"[INFO] outputs  : {outputs_root}")

    all_rows = []
    for method in methods:
        for concept in concepts:
            # e.g. outputs/modified_bowl/eval/modified_bowl.csv
            folder_name = f"{method}_{concept}"
            eval_dir = os.path.join(outputs_root, folder_name, "eval")
            csv_path = os.path.join(eval_dir, f"{method}_{concept}.csv")

            if not os.path.exists(csv_path):
                print(f"[WARN] CSV not found for (method={method}, concept={concept}): {csv_path}")
                continue

            print(f"[INFO] summarizing {csv_path}")
            rows = summarize_one_csv(csv_path, method=method, concept=concept)
            all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No CSVs found, nothing to summarize.")
        return

    df_out = pd.DataFrame(all_rows)

    results_dir = os.path.join(outputs_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, args.summary_name)
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] summary written to: {out_path}")


if __name__ == "__main__":
    main()
