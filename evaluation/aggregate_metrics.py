"""
聚合所有方法的 per-prompt 指标，输出 mean ± std 汇总表。
用法：
    python evaluation/aggregate_metrics.py \
        --outputs-root ./outputs \
        --methods ourmethod dpp cads base mix pg apg \
        --concepts t2i_color t2i_complex
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


METRICS = ["vendi_inception", "vendi_pixel", "clip_score", "one_minus_ms_ssim"]


def _find_method_dir(outputs_root: Path, method: str, concept: str) -> Optional[Path]:
    for candidate in [
        outputs_root / f"{method}_{concept}",
        outputs_root / f"{method}_t2i_{concept}",
        outputs_root / f"baseline_{method}_{concept}",
    ]:
        if (candidate / "eval" / "metrics_per_prompt.csv").exists():
            return candidate
    return None


def load_per_prompt(outputs_root: Path, method: str, concept: str
                    ) -> Optional[List[Dict]]:
    d = _find_method_dir(outputs_root, method, concept)
    if d is None:
        return None
    csv_path = d / "eval" / "metrics_per_prompt.csv"
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_stats(rows: List[Dict], metric: str):
    vals = []
    for r in rows:
        v = r.get(metric, "")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            pass
    if not vals:
        return None, None
    a = np.array(vals)
    return float(a.mean()), float(a.std(ddof=1))


def load_global_kid(outputs_root: Path) -> Dict:
    """Returns {(method, concept): (kid_mean, kid_std)}"""
    kid_csv = outputs_root / "global_kid.csv"
    result = {}
    if not kid_csv.exists():
        return result
    with open(kid_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["method"], row["concept"])
            try:
                km = float(row["kid_mean"]) if row["kid_mean"] else None
                ks = float(row["kid_std"])  if row["kid_std"]  else None
            except (ValueError, TypeError):
                km, ks = None, None
            result[key] = (km, ks)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="./outputs")
    parser.add_argument("--methods",  nargs="+",
                        default=["ourmethod", "dpp", "cads", "base", "mix", "pg", "apg"])
    parser.add_argument("--concepts", nargs="+",
                        default=["t2i_color", "t2i_complex"])
    args = parser.parse_args()

    root = Path(args.outputs_root)
    kid_data = load_global_kid(root)

    all_rows = []  # for CSV output

    for concept in args.concepts:
        print(f"\n{'='*70}")
        print(f"  Concept: {concept}")
        print(f"{'='*70}")

        # header
        col_w = 14
        header = f"{'Method':<12}" + "".join(
            f"{m:>{col_w}}" for m in METRICS
        ) + f"{'KID(×1e-3)':>{col_w}}"
        print(header)
        print("-" * len(header))

        for method in args.methods:
            rows = load_per_prompt(root, method, concept)
            if rows is None:
                print(f"{method:<12}  [not found]")
                continue

            row_data = {"method": method, "concept": concept, "n_prompts": len(rows)}
            line = f"{method:<12}"

            for metric in METRICS:
                mean, std = compute_stats(rows, metric)
                if mean is not None:
                    line += f"  {mean:.3f}±{std:.3f}".rjust(col_w)
                    row_data[f"{metric}_mean"] = round(mean, 4)
                    row_data[f"{metric}_std"]  = round(std,  4)
                else:
                    line += f"{'N/A':>{col_w}}"
                    row_data[f"{metric}_mean"] = None
                    row_data[f"{metric}_std"]  = None

            # KID
            kid_key = (method, concept)
            # also try without "t2i_" prefix in concept for lookup
            km, ks = kid_data.get(kid_key, (None, None))
            if km is not None:
                line += f"  {km:.3f}±{ks:.3f}".rjust(col_w)
                row_data["kid_mean"] = round(km, 4)
                row_data["kid_std"]  = round(ks, 4)
            else:
                line += f"{'N/A':>{col_w}}"
                row_data["kid_mean"] = None
                row_data["kid_std"]  = None

            print(line)
            all_rows.append(row_data)

    # Save summary CSV
    out_csv = root / "metrics_summary.csv"
    fieldnames = (
        ["method", "concept", "n_prompts"]
        + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
        + ["kid_mean", "kid_std"]
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n>> Summary saved to {out_csv}")


if __name__ == "__main__":
    main()
