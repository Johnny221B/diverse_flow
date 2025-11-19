#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate per-prompt metrics (mean/std over seeds) from RAW CSV files.

输入 CSV 至少包含：
  method, concept, folder, prompt, seed, guidance, steps, num_images,
  以及若干 metric 列（例如 vendi_pixel, vendi_inception, fid, clip_score,
  one_minus_ms_ssim, brisque, kid, ...）。

注意：
  - 本脚本会自动忽略所有列名以 `_mean` 或 `_std` 结尾的列，
    避免对已经聚合过的表再聚合一次。
  - 对每个 metric 只输出两列：<metric>_mean 和 <metric>_std。
  - 整列都是 NaN 的 metric 直接丢弃。

输出：
  - per_prompt_stats_all.csv
  - 每个 (method, concept)：<method>_<concept>_per_prompt_stats.csv
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd


def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts: List[str] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p for p in chunk.split() if p])
    return parts


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-prompt metrics (mean/std over seeds) from RAW CSV files."
    )
    ap.add_argument(
        "--csv_root",
        type=str,
        default=".",
        help="Folder containing CSV files.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to select CSV files, e.g. 'raw_*.csv'. "
             "注意不要让之前聚合出的 per_prompt_stats_*.csv 也被匹配进来。",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Methods to keep (comma/space separated). If omitted, keep all.",
    )
    ap.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Concepts to keep (comma/space separated). If omitted, keep all.",
    )
    args = ap.parse_args()

    csv_root = Path(args.csv_root)
    assert csv_root.exists(), f"csv_root not found: {csv_root}"

    methods = parse_list(args.methods)
    concepts = parse_list(args.concepts)

    # -------- 读所有 RAW CSV --------
    all_dfs = []
    for csv_path in csv_root.glob(args.pattern):
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            all_dfs.append(df)
            print(f"[INFO] Loaded {csv_path} with {len(df)} rows.")
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")

    if not all_dfs:
        print("[ERROR] No CSVs loaded. Check --csv_root and --pattern.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # -------- 按 method / concept 过滤 --------
    if methods:
        df_all = df_all[df_all["method"].isin(methods)]
    if concepts:
        df_all = df_all[df_all["concept"].isin(concepts)]

    if df_all.empty:
        print("[ERROR] No rows remaining after filtering by methods/concepts.")
        return

    # -------- 定义 meta 和 metric 列 --------
    meta_cols = [
        "method",
        "concept",
        "folder",
        "prompt",
        "seed",
        "guidance",
        "steps",
        "num_images",  # 也会被统计 mean/std
    ]

    # 候选 metric = 所有非 meta 列，且列名不以 _mean/_std 结尾
    candidate_metrics = [
        c for c in df_all.columns
        if c not in meta_cols and not c.endswith("_mean") and not c.endswith("_std")
    ]

    metric_cols: List[str] = []
    for col in candidate_metrics:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        # 整列全 NaN 的直接丢弃
        if df_all[col].notna().any():
            metric_cols.append(col)

    # num_images 也当作一个 metric（如果有）
    if "num_images" in df_all.columns:
        df_all["num_images"] = pd.to_numeric(df_all["num_images"], errors="coerce")
        if df_all["num_images"].notna().any():
            metric_cols.append("num_images")

    # 去重防万一
    metric_cols = sorted(set(metric_cols))

    if not metric_cols:
        print("[ERROR] No metric columns with data (all NaN or filtered out).")
        return

    print(f"[INFO] Metric columns (non-empty, raw): {metric_cols}")

    # -------- 分组聚合 --------
    group_keys = ["method", "concept", "prompt", "guidance", "steps"]
    gb = df_all.groupby(group_keys)

    size_series = gb.size().rename("n")

    # 每个 metric 只取 mean / std
    agg_dict: Dict[str, List[str]] = {m: ["mean", "std"] for m in metric_cols}
    agg_metrics = gb[metric_cols].agg(agg_dict)

    # 展平：vendi_pixel_mean, vendi_pixel_std, ...
    agg_metrics.columns = [
        f"{metric}_{stat}" for metric, stat in agg_metrics.columns.to_flat_index()
    ]

    agg_df = agg_metrics.reset_index()
    agg_df = agg_df.merge(size_series.reset_index(), on=group_keys, how="left")

    # -------- 写出 CSV --------
    all_out = csv_root / "per_prompt_stats_all.csv"
    agg_df.to_csv(all_out, index=False)
    print(f"[DONE] Wrote global per-prompt stats -> {all_out}")

    for (m, c), sub in agg_df.groupby(["method", "concept"]):
        out_path = csv_root / f"{m}_{c}_per_prompt_stats.csv"
        sub.to_csv(out_path, index=False)
        print(f"[DONE] ({m}, {c}) -> {out_path}")


if __name__ == "__main__":
    main()
