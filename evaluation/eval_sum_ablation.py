#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize metrics (mean, std) for ablation & robust studies.

Input layouts (already produced by your evaluators):

A) Ablation:
   outputs/ablation/<method>/eval/ablation_<method>.csv
   (fallback: any *.csv in that eval/ if the canonical name is missing)

B) Robust:
   outputs/robust_study/<group>/eval/<group>_<value>.csv
   where group ∈ {lambda, alpha, noise_gate}, value like '0.60' or '0.50-0.95'

Output:
   <outputs-root>/results/
     - ablation_summary.csv
     - robust_lambda_summary.csv
     - robust_alpha_summary.csv
     - robust_noise_gate_summary.csv

Each summary row aggregates across all rows in a source CSV (i.e., across seeds/runs).
"""

import argparse, os, re, csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import numpy as np

# Try pandas, but work without it too.
try:
    import pandas as pd
except Exception:
    pd = None

IMG_METRICS_DEFAULT = [
    "vendi_pixel",
    "vendi_inception",
    "fid",
    "fid_clean",
    "clip_score",
    "one_minus_ms_ssim",
    "brisque",
]

ROBUST_GROUPS_ORDER = ["lambda", "alpha", "noise_gate"]


# ------------------ IO helpers ------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV as list of dict rows; robust to missing pandas."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def _to_float_or_nan(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _available_metrics(rows: List[Dict[str, str]], wanted: List[str]) -> List[str]:
    if not rows:
        return []
    cols = set(rows[0].keys())
    return [m for m in wanted if m in cols]

def _mean_std_of_column(rows: List[Dict[str, str]], col: str) -> Tuple[float, float, int]:
    """Return (mean, std, count_used) using NaN-robust aggregation."""
    vals = np.array([_to_float_or_nan(r.get(col)) for r in rows], dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"), 0)
    # population std by default; use ddof=1 for sample std if len>1
    if vals.size >= 2:
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)), int(vals.size))
    else:
        return (float(vals.mean()), float(0.0), 1)

def _write_rows_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        # Write empty file with no columns? Better: write header of metrics_mean/std anyway
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")  # intentionally empty
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------ Ablation summarizer ------------------

def _find_ablation_methods(ablation_root: Path) -> List[str]:
    if not ablation_root.exists():
        return []
    methods = []
    for d in sorted(ablation_root.iterdir()):
        if d.is_dir() and (d / "eval").exists():
            methods.append(d.name)
    return methods

def _find_ablation_eval_csv(method_root: Path, method_name: str) -> Optional[Path]:
    eval_dir = method_root / "eval"
    if not eval_dir.exists():
        return None
    cand = eval_dir / f"ablation_{method_name}.csv"
    if cand.exists():
        return cand
    # fallback: first csv in eval/
    for p in sorted(eval_dir.glob("*.csv")):
        return p
    return None

def summarize_ablation(outputs_root: Path,
                       methods: Optional[List[str]],
                       metrics: List[str]) -> Path:
    ablation_root = outputs_root / "ablation"
    if methods is None or len(methods) == 0:
        methods = _find_ablation_methods(ablation_root)

    summary_rows: List[Dict[str, object]] = []
    for m in methods:
        mroot = ablation_root / m
        csv_path = _find_ablation_eval_csv(mroot, m)
        if csv_path is None or not csv_path.exists():
            print(f"[ablation] skip: no eval csv for method={m}")
            continue
        rows = _read_csv_rows(csv_path)
        if not rows:
            print(f"[ablation] empty csv: {csv_path}")
            continue
        use_metrics = _available_metrics(rows, metrics)
        out_row: Dict[str, object] = {"method": m}
        for col in use_metrics:
            mean, std, n = _mean_std_of_column(rows, col)
            out_row[f"{col}_mean"] = mean
            out_row[f"{col}_std"] = std
            out_row[f"{col}_n"] = n
        summary_rows.append(out_row)

    out_dir = outputs_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_summary.csv"
    if summary_rows:
        # order columns nicely: method first, then metric mean/std/n triplets in fixed order
        ordered_cols = ["method"]
        used_metrics_all = []
        for col in metrics:
            # include metric only if present in any row
            present = any(f"{col}_mean" in r for r in summary_rows)
            if present:
                used_metrics_all.append(col)
        for col in used_metrics_all:
            ordered_cols += [f"{col}_mean", f"{col}_std", f"{col}_n"]
        # reorder rows
        rows_reordered = []
        for r in summary_rows:
            rr = {k: r.get(k, "") for k in ordered_cols}
            rows_reordered.append(rr)
        _write_rows_csv(out_path, rows_reordered)
    else:
        _write_rows_csv(out_path, [])
    print(f"[ablation] summary -> {out_path}")
    return out_path


# ------------------ Robust summarizer ------------------

VALUE_FROM_FILENAME_RE = re.compile(r"^(?P<group>lambda|alpha|noise_gate)_(?P<val>.+)\.csv$")

def _find_robust_groups(robust_root: Path) -> List[str]:
    if not robust_root.exists():
        return []
    groups = [d.name for d in robust_root.iterdir() if d.is_dir() and (d / "eval").exists()]
    # keep preferred order
    ordered = [g for g in ROBUST_GROUPS_ORDER if g in groups]
    for g in sorted(groups):
        if g not in ordered:
            ordered.append(g)
    return ordered

def _collect_group_value_csvs(group_root: Path, group_name: str) -> List[Tuple[str, Path]]:
    """Return list of (value_str, csv_path). Value from filename; fallback to file content."""
    eval_dir = group_root / "eval"
    out: List[Tuple[str, Path]] = []
    if not eval_dir.exists():
        return out
    for p in sorted(eval_dir.glob("*.csv")):
        m = VALUE_FROM_FILENAME_RE.match(p.name)
        if m and m.group("group") == group_name:
            out.append((m.group("val"), p))
        else:
            # fallback: try read 'value' column first row
            try:
                rows = _read_csv_rows(p)
                v = rows[0].get("value", None) if rows else None
                if v is None:
                    continue
                out.append((str(v), p))
            except Exception:
                continue
    return out

def _robust_sort_key(group: str, value: str):
    # Sort numbers numerically; noise_gate "a-b" by (a,b)
    try:
        if group in ("lambda", "alpha"):
            return (float(value),)
        if group == "noise_gate":
            if "-" in value:
                a, b = value.split("-", 1)
                return (float(a), float(b))
            return (float(value),)
    except Exception:
        pass
    return (str(value),)

def summarize_robust(outputs_root: Path,
                     groups: Optional[List[str]],
                     metrics: List[str]) -> List[Path]:
    robust_root = outputs_root / "robust_study"
    if groups is None or len(groups) == 0:
        groups = _find_robust_groups(robust_root)

    out_paths: List[Path] = []
    for g in groups:
        groot = robust_root / g
        pairs = _collect_group_value_csvs(groot, g)
        if not pairs:
            print(f"[robust:{g}] no value csvs found under {groot/'eval'}")
            continue

        # build rows per value
        rows_out: List[Dict[str, object]] = []
        for value, csv_path in sorted(pairs, key=lambda kv: _robust_sort_key(g, kv[0])):
            rows = _read_csv_rows(csv_path)
            if not rows:
                continue
            use_metrics = _available_metrics(rows, metrics)
            out_row: Dict[str, object] = {"group": g, "value": value}
            for col in use_metrics:
                mean, std, n = _mean_std_of_column(rows, col)
                out_row[f"{col}_mean"] = mean
                out_row[f"{col}_std"] = std
                out_row[f"{col}_n"] = n
            rows_out.append(out_row)

        # write one CSV per group
        out_dir = outputs_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"robust_{g}_summary.csv"
        if rows_out:
            ordered_cols = ["group", "value"]
            used_metrics_all = []
            for col in metrics:
                present = any(f"{col}_mean" in r for r in rows_out)
                if present:
                    used_metrics_all.append(col)
            for col in used_metrics_all:
                ordered_cols += [f"{col}_mean", f"{col}_std", f"{col}_n"]
            rows_reordered = [{k: r.get(k, "") for k in ordered_cols} for r in rows_out]
            _write_rows_csv(out_path, rows_reordered)
        else:
            _write_rows_csv(out_path, [])
        print(f"[robust:{g}] summary -> {out_path}")
        out_paths.append(out_path)

    return out_paths


# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize mean/std for ablation & robust studies.")
    ap.add_argument("--outputs-root", type=str, default="./outputs",
                    help="Root path containing 'ablation' and/or 'robust_study'.")
    ap.add_argument("--study", choices=["ablation", "robust", "both"], default="both")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="(Optional) Methods for ablation; if omitted, auto-detect under outputs/ablation.")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="(Optional) Groups for robust; if omitted, auto-detect under outputs/robust_study.")
    ap.add_argument("--metrics", nargs="+", default=IMG_METRICS_DEFAULT,
                    help="Which metric columns to aggregate (must exist in source CSVs).")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root).resolve()

    if args.study in ("ablation", "both"):
        summarize_ablation(outputs_root, args.methods, args.metrics)

    if args.study in ("robust", "both"):
        summarize_robust(outputs_root, args.groups, args.metrics)


if __name__ == "__main__":
    main()