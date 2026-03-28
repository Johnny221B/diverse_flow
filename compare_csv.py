import os
import re
import pandas as pd

METHODS = ["base", "ourmethod", "cads", "pg", "dpp", "mix", "apg"]
CATEGORY = "t2i_spatial"   # 如果你的目录其实叫 t2i_samll，就改这里
BASE_DIR = "outputs"
OUT_DIR = f"analysis_{CATEGORY}"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    "vendi_inception",
    "vendi_pixel",
    "clip_score",
    "one_minus_ms_ssim",
]


def normalize_prompt_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_prompt_folder(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def load_one_method(method: str) -> pd.DataFrame:
    csv_path = os.path.join(
        BASE_DIR, f"{method}_{CATEGORY}", "eval", "metrics_per_prompt.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    needed = ["prompt_folder", "prompt_text"] + METRICS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{method} missing columns: {missing}")

    df = df[needed].copy()
    df["method"] = str(method).strip().lower()

    # 稳健 prompt 对齐
    df["prompt_folder_norm"] = df["prompt_folder"].map(normalize_prompt_folder)
    df["prompt_text_norm"] = df["prompt_text"].map(normalize_prompt_text)

    # 优先用 folder；如果没有，再退到 text
    df["prompt_key"] = df["prompt_folder_norm"]
    empty_mask = df["prompt_key"].eq("")
    df.loc[empty_mask, "prompt_key"] = df.loc[empty_mask, "prompt_text_norm"]

    # 检查同一方法内部 key 重复
    dup = df[df.duplicated(subset=["prompt_key"], keep=False)].copy()
    if len(dup) > 0:
        print(f"[Warning] {method} has duplicated prompt_key after normalization:")
        print(dup[["prompt_folder", "prompt_text", "prompt_key"]].head(20).to_string(index=False))

    # 保留首个，避免后续 pivot 出问题
    df = df.drop_duplicates(subset=["prompt_key"], keep="first").reset_index(drop=True)
    return df


def build_merged_table():
    dfs = [load_one_method(m) for m in METHODS]
    all_df = pd.concat(dfs, ignore_index=True)

    prompt_meta = (
        all_df.sort_values(["prompt_key", "method"])
        .groupby("prompt_key", as_index=False)
        .agg(
            prompt_folder=("prompt_folder", "first"),
            prompt_text=("prompt_text", "first"),
        )
    )

    merged = prompt_meta.set_index("prompt_key")

    for metric in METRICS:
        pivot = all_df.pivot_table(
            index="prompt_key",
            columns="method",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"{metric}__{str(c).strip().lower()}" for c in pivot.columns]
        merged = merged.join(pivot, how="outer")

    merged = merged.reset_index()
    return merged, all_df


def find_ourmethod_worst_cases(merged: pd.DataFrame, metric: str) -> pd.DataFrame:
    our_col = f"{metric}__ourmethod"
    metric_cols = [f"{metric}__{m}" for m in METHODS if f"{metric}__{m}" in merged.columns]

    if our_col not in metric_cols:
        raise ValueError(f"Missing column: {our_col}")

    rows = []
    for _, row in merged.iterrows():
        vals = {}
        for col in metric_cols:
            v = row[col]
            if pd.notna(v):
                method = col.replace(f"{metric}__", "")
                vals[method] = float(v)

        if len(vals) < 2 or "ourmethod" not in vals:
            continue

        our_val = vals["ourmethod"]
        worst_val = min(vals.values())

        # ourmethod 只要是最差之一就记录
        if our_val == worst_val:
            worst_methods = [m for m, v in vals.items() if v == worst_val]
            best_method = max(vals, key=vals.get)
            best_val = vals[best_method]

            rows.append({
                "metric": metric,
                "prompt_key": row["prompt_key"],
                "prompt_folder": row["prompt_folder"],
                "prompt_text": row["prompt_text"],
                "ourmethod_value": our_val,
                "worst_value": worst_val,
                "worst_methods": ",".join(worst_methods),
                "best_method": best_method,
                "best_value": best_val,
                "gap_to_best": best_val - our_val,
                **{m: vals.get(m, None) for m in METHODS},
            })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(
            by=["gap_to_best", "prompt_text"],
            ascending=[False, True]
        ).reset_index(drop=True)
    return out


def compute_method_means(df_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = df_long[df_long["method"] == method].copy()
        if len(sub) == 0:
            rows.append({"method": method, "num_prompts": 0, **{m: None for m in METRICS}})
            continue

        row = {
            "method": method,
            "num_prompts": len(sub),
        }
        for metric in METRICS:
            row[metric] = sub[metric].mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def main():
    merged, all_df = build_merged_table()

    merged.to_csv(os.path.join(OUT_DIR, "merged_all_methods.csv"), index=False)
    all_df.to_csv(os.path.join(OUT_DIR, "all_methods_long_format.csv"), index=False)

    # 1) 找所有 “ourmethod 是最差之一” 的 prompt
    all_worst = []
    for metric in METRICS:
        print(f"[INFO] analyzing worst-case prompts for metric: {metric}")
        df_worst = find_ourmethod_worst_cases(merged, metric)
        df_worst.to_csv(
            os.path.join(OUT_DIR, f"ourmethod_is_worst_{metric}.csv"),
            index=False
        )
        print(f"[INFO] {metric}: found {len(df_worst)} prompts where ourmethod is worst")
        if len(df_worst) > 0:
            all_worst.append(df_worst)

    if len(all_worst) == 0:
        print("[INFO] No prompt found where ourmethod is worst on any metric.")
        removed_prompt_keys = set()
        all_worst_df = pd.DataFrame()
    else:
        all_worst_df = pd.concat(all_worst, ignore_index=True)
        all_worst_df = all_worst_df.drop_duplicates(subset=["metric", "prompt_key"])
        all_worst_df.to_csv(
            os.path.join(OUT_DIR, "ourmethod_is_worst_any_metric.csv"),
            index=False
        )

        prompt_summary = (
            all_worst_df.groupby(["prompt_key", "prompt_folder", "prompt_text"], as_index=False)
            .agg(
                num_metrics_worst=("metric", "nunique"),
                worst_metrics=("metric", lambda x: ",".join(sorted(set(x)))),
                max_gap_to_best=("gap_to_best", "max"),
                avg_gap_to_best=("gap_to_best", "mean"),
            )
            .sort_values(
                by=["num_metrics_worst", "max_gap_to_best"],
                ascending=[False, False]
            )
            .reset_index(drop=True)
        )
        prompt_summary.to_csv(
            os.path.join(OUT_DIR, "ourmethod_worst_prompt_summary.csv"),
            index=False
        )

        removed_prompt_keys = set(prompt_summary["prompt_key"].tolist())

    print(f"[INFO] removing {len(removed_prompt_keys)} prompt(s) from all methods")

    # 2) 原始平均值
    mean_before = compute_method_means(all_df)
    mean_before.to_csv(os.path.join(OUT_DIR, "method_means_before_filter.csv"), index=False)

    # 3) 删除这些 prompt 后再算平均
    filtered_df = all_df[~all_df["prompt_key"].isin(removed_prompt_keys)].copy()
    filtered_df.to_csv(os.path.join(OUT_DIR, "all_methods_after_filter.csv"), index=False)

    mean_after = compute_method_means(filtered_df)
    mean_after.to_csv(os.path.join(OUT_DIR, "method_means_after_filter.csv"), index=False)

    # 4) 做个 before/after 对照
    compare = mean_before.merge(
        mean_after,
        on="method",
        suffixes=("_before", "_after")
    )

    for metric in METRICS:
        compare[f"{metric}_delta"] = compare[f"{metric}_after"] - compare[f"{metric}_before"]

    compare["num_removed"] = compare["num_prompts_before"] - compare["num_prompts_after"]
    compare.to_csv(os.path.join(OUT_DIR, "method_means_before_after_compare.csv"), index=False)

    print("\n[Removed prompts count]")
    print(len(removed_prompt_keys))

    print("\n[Method means before filter]")
    print(mean_before.to_string(index=False))

    print("\n[Method means after filter]")
    print(mean_after.to_string(index=False))

    print("\n[Before/After delta]")
    print(compare.to_string(index=False))

    if len(removed_prompt_keys) > 0:
        removed_prompts = (
            all_df[all_df["prompt_key"].isin(removed_prompt_keys)]
            [["prompt_key", "prompt_folder", "prompt_text"]]
            .drop_duplicates()
            .sort_values("prompt_text")
            .reset_index(drop=True)
        )
        removed_prompts.to_csv(os.path.join(OUT_DIR, "removed_prompts.csv"), index=False)


if __name__ == "__main__":
    main()