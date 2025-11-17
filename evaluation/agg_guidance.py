#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给定一个实验结果 csv，按 guidance（可选再细分到 prompt）聚合各指标的 mean / std。

csv 示例列：
method,concept,folder,prompt,seed,guidance,steps,num_images,
vendi_pixel,vendi_inception,fid,fid_clean,kid_mean,kid_std,
clip_score,one_minus_ms_ssim,brisque

用法示例：

1）只按 guidance 聚合：
python aggregate_by_guidance_and_prompt.py \
  --csv path/to/results.csv

2）只用若干指定 prompt，并按 (prompt, guidance) 聚合：
python aggregate_by_guidance_and_prompt.py \
  --csv path/to/results.csv \
  --prompts "a close-up photo of a truck" "a bus on the road" \
  --out path/to/out.csv
"""

import os
import re
import argparse
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="输入的 csv 文件路径")
    ap.add_argument("--out", type=str, default=None,
                    help="输出的聚合结果 csv 文件路径（默认：在原文件名后加 _agg）")
    ap.add_argument("--prompts", type=str, nargs="+", default=None,
                    help="只保留这些 prompt（原始文本形式，可以多个）")
    return ap.parse_args()


def ensure_guidance_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 df 里有 'guidance' 这一列：
    - 如果本身有 guidance 列：直接转成 float 用
    - 如果没有：从 folder 里解析 *_g3.0_s30* 这种格式
    """
    if "guidance" in df.columns:
        df["guidance"] = pd.to_numeric(df["guidance"], errors="coerce")
        return df

    if "folder" not in df.columns:
        raise ValueError("既没有 guidance 列，也没有 folder 列，没法解析 guidance。")

    pattern = re.compile(r"_g([0-9.]+)_s")  # 匹配 _g3.0_s30 里的 3.0

    def extract_g(x: str):
        m = pattern.search(str(x))
        if m:
            return float(m.group(1))
        return float("nan")

    df["guidance"] = df["folder"].apply(extract_g)
    return df


def ensure_prompt_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 df 里有 'prompt' 这一列：
    - 如果本身有 prompt 列：直接用
    - 如果没有：从 folder 前缀解析：
        a_close-up_photo_of_a_truck_seed1111_g3.0_s30
      -> 取 '_' + 'seed' 之间当 slug: a_close-up_photo_of_a_truck
      -> 下划线变空格: a close-up photo of a truck
    """
    if "prompt" in df.columns:
        return df

    if "folder" not in df.columns:
        raise ValueError("既没有 prompt 列，也没有 folder 列，没法解析 prompt。")

    pattern = re.compile(r"^(.*)_seed[0-9]+_g[0-9.]+_s[0-9]+")

    def extract_prompt_from_folder(x: str) -> str:
        x = str(x)
        m = pattern.match(x)
        if not m:
            return ""
        slug = m.group(1)
        # 下划线变空格
        return slug.replace("_", " ")

    df["prompt"] = df["folder"].apply(extract_prompt_from_folder)
    return df


def normalize_prompt(s: str) -> str:
    """简单归一化：去首尾空格，合并中间多空格，统一小写。"""
    s = str(s).strip().lower()
    # 把连续空白折叠成一个空格
    s = " ".join(s.split())
    return s


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"csv 文件不存在: {args.csv}")

    print(f"[info] 读取 csv: {args.csv}")
    df = pd.read_csv(args.csv)

    # 确保有 guidance / prompt
    df = ensure_guidance_column(df)
    df = ensure_prompt_column(df)

    # 选取要做 mean/std 的指标列（存在才用）
    candidate_metrics = [
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
    metrics = [c for c in candidate_metrics if c in df.columns]

    if not metrics:
        raise ValueError("在 csv 里没有找到任何候选指标列，检查列名是否匹配。")

    print("[info] 将对以下指标计算 mean/std：")
    for m in metrics:
        print("  -", m)

    # 丢掉无法解析 guidance 的行
    df = df.dropna(subset=["guidance"])

    # 如果指定了 prompts，就只保留这些 prompt 的行
    if args.prompts is not None and len(args.prompts) > 0:
        print("[info] 只保留以下 prompt 的数据：")
        for p in args.prompts:
            print("  -", p)

        # 归一化对齐
        target_norms = {normalize_prompt(p) for p in args.prompts}
        df["_prompt_norm"] = df["prompt"].astype(str).map(normalize_prompt)

        # 看看哪些 prompt 在数据里找到了
        found_norms = set(df["_prompt_norm"].unique()) & target_norms
        missing_norms = target_norms - found_norms
        if missing_norms:
            print("[warn] 以下 prompt 在数据中未找到（归一化后）：")
            for p in missing_norms:
                print("  -", p)

        # 过滤
        df = df[df["_prompt_norm"].isin(target_norms)]

        if df.empty:
            raise ValueError("过滤后没有任何数据行，请检查 --prompts 是否和 csv 中的 prompt 对得上。")

        # 按 prompt + guidance 分组
        group_keys = ["prompt", "guidance"]
    else:
        # 只按 guidance 分组
        group_keys = ["guidance"]

    print(f"[info] 按 {group_keys} 做聚合")
    grouped = (
        df.groupby(group_keys)[metrics]
          .agg(["mean", "std"])
          .sort_index()
    )

    grouped = grouped.round(6)

    # 生成默认输出路径
    if args.out is None:
        base, ext = os.path.splitext(args.csv)
        out_path = base + "_agg.csv"
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[info] 保存结果到: {out_path}")
    grouped.to_csv(out_path)

    print("\n[info] 聚合结果预览：")
    print(grouped.head())


if __name__ == "__main__":
    main()
