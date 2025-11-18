#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute ImageReward mean ± std per guidance for multiple (method, concept) pairs.

Layout per pair:
  outputs/<method>_<concept>/imgs/**                 # images (possibly inside subfolders named like: <prompt>_seed1111_g3.0_s30)
  outputs/<method>_<concept>/eval/imagereward_agg.csv  # this script writes here

How it works:
- Parse prompt/seed/guidance/step from parent folder or filename:
    "<prompt>_seed<seed>_g<guidance>_s<step>"
- Prompt is restored by replacing "_" -> " "
- Score each image with ImageReward (higher = better): model.score(prompt, path_or_PIL)
- Aggregate mean ± std over (method, concept, guidance)
- Write one CSV per pair, plus a master CSV at outputs_root.

Requires:
  pip install torch pillow pandas
  (and your ImageReward repo installed; import interface: `import ImageReward as RM`)
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import ImageReward as RM  # you confirmed this works in your env

PROMPT_RE = re.compile(
    r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[0-9.]+)_s(?P<step>\d+)$"
)

def parse_token(name: str) -> Dict[str, str]:
    m = PROMPT_RE.match(name)
    if not m:
        return {"prompt_token": name, "seed": "", "guidance": "", "step": ""}
    d = m.groupdict()
    return {
        "prompt_token": d["prompt"],
        "seed": d["seed"],
        "guidance": d["guidance"],
        "step": d["step"],
    }

def token_to_prompt(token: str) -> str:
    return token.replace("_", " ").strip()

def walk_images(root: Path, exts=(".png", ".jpg", ".jpeg", ".webp")) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)

def parse_method_concept(dirname: str) -> Tuple[str, str]:
    return dirname.split("_", 1) if "_" in dirname else (dirname, "")

def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p for p in chunk.split() if p])
    return parts

def auto_discover_pairs(outputs_root: Path) -> List[Tuple[str, str]]:
    pairs = []
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        m, c = parse_method_concept(d.name)
        if (d / "imgs").exists():
            pairs.append((m, c))
    return sorted(set(pairs))

def eval_one_pair_imagereward(
    outputs_root: Path,
    method: str,
    concept: str,
    model,  # RM.load(...) returned object
) -> Optional[pd.DataFrame]:
    mc_dir = outputs_root / f"{method}_{concept}"
    imgs_dir = mc_dir / "imgs"
    eval_dir = mc_dir / "eval"
    if not imgs_dir.exists():
        print(f"[WARN] Skip: {imgs_dir} not found.")
        return None
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for img_path in walk_images(imgs_dir):
            parent_token = img_path.parent.name
            stem_token = img_path.stem
            meta = parse_token(parent_token)
            if meta["seed"] == "" and meta["guidance"] == "" and meta["step"] == "":
                meta = parse_token(stem_token)

            prompt_token = meta["prompt_token"]
            if not prompt_token:
                # no prompt → skip (ImageReward requires prompt)
                continue
            prompt = token_to_prompt(prompt_token)

            # You can pass a file path; the RM implementation accepts path or PIL.Image
            try:
                score = float(model.score(prompt, str(img_path)))
            except Exception:
                # fallback: open via PIL in case path input is not accepted in your build
                try:
                    score = float(model.score(prompt, Image.open(img_path).convert("RGB")))
                except Exception as e:
                    print(f"[WARN] ImageReward failed on {img_path}: {e}")
                    continue

            rows.append({
                "method": method,
                "concept": concept,
                "guidance": str(meta["guidance"]),
                "score": score,
            })

    if not rows:
        print(f"[WARN] No valid (prompt, image) under {imgs_dir}.")
        agg = pd.DataFrame(columns=["method","concept","guidance","n","mean","std"])
    else:
        df = pd.DataFrame(rows)
        g = df.groupby(["method","concept","guidance"])["score"]
        agg = pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "std": g.std(ddof=1)
        }).reset_index().sort_values(["method","concept","guidance"])

    out_csv = eval_dir / "imagereward_agg.csv"
    agg.to_csv(out_csv, index=False)
    print(f"[DONE] {method}_{concept} -> {out_csv}")
    return agg

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute ImageReward mean±std per guidance for multiple (method, concept) pairs.")
    ap.add_argument("--outputs_root", type=str, default="/mnt/data6t/yyz/flow_grpo/flow_base/outputs", help="Root folder containing <method>_<concept> subfolders.")
    ap.add_argument("--methods", type=str, default=None, help="Methods (comma/space separated), e.g., 'ourmethod,baselineA baselineB'")
    ap.add_argument("--concepts", type=str, default=None, help="Concepts (comma/space separated), e.g., 'truck,cat'")
    ap.add_argument("--auto_discover", action="store_true", help="Auto scan outputs_root for all <method>_<concept> pairs.")
    ap.add_argument("--device", type=str, default="cuda", help="Device passed to ImageReward; use 'cpu' if no GPU.")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    assert outputs_root.exists(), f"outputs_root not found: {outputs_root}"

    # Resolve pairs
    if args.auto_discover:
        pairs = auto_discover_pairs(outputs_root)
        if not pairs:
            print(f"[WARN] auto_discover found no pairs under {outputs_root}.")
    else:
        methods = parse_list(args.methods)
        concepts = parse_list(args.concepts)
        if not methods or not concepts:
            raise ValueError("Please provide --methods and --concepts (or use --auto_discover).")
        pairs = [(m, c) for m in methods for c in concepts]

    # Load ImageReward once (your confirmed API)
    model = RM.load("ImageReward-v1.0", device=args.device)

    # Run all pairs
    all_aggs: List[pd.DataFrame] = []
    for (method, concept) in pairs:
        agg = eval_one_pair_imagereward(outputs_root, method, concept, model)
        if agg is not None and not agg.empty:
            all_aggs.append(agg)

    # Master CSV at root
    if all_aggs:
        master = pd.concat(all_aggs, ignore_index=True)
        master_csv = outputs_root / "imagereward_agg_all.csv"
        master.to_csv(master_csv, index=False)
        print(f"[DONE] Master aggregated CSV -> {master_csv}")
    else:
        print("[INFO] No aggregated results to write at master level.")

if __name__ == "__main__":
    main()