#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute CLIP-IQA (LAION Aesthetic Predictor) for MULTIPLE (method, concept) pairs.

Directory layout (per pair):
  outputs/<method>_<concept>/imgs/**  # images live here (possibly under subfolders named like "..._seed1111_g3.0_s30")
  outputs/<method>_<concept>/eval/   # we will write aggregated CSV here

What it does:
- Loads CLIP (OpenAI or TorchScript JIT) once, and a matching LAION linear head once.
- For each <method,concept>:
    * walks all images under imgs/**,
    * parses guidance from parent folder or filename (regex: "..._seed(?P<seed>\\d+)_g(?P<guidance>[0-9.]+)_s(?P<step>\\d+)"),
    * computes an aesthetic score (higher = better),
    * aggregates mean ± std per guidance,
    * writes ONLY: outputs/<method>_<concept>/eval/clip_iqa_agg.csv
- Additionally writes a master CSV combining all pairs: outputs/clip_iqa_agg_all.csv

Usage examples:
  python run_multi_clip_iqa.py --methods ourmethod,baseline --concepts truck,cat --clip_model vit_b_32
  python run_multi_clip_iqa.py --methods ourmethod --concepts truck,dog --jit_path ~/.cache/ViT-B-32.pt
  # auto-discover all pairs under outputs:
  python run_multi_clip_iqa.py --auto_discover
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from os.path import expanduser
from urllib.request import urlretrieve

# ----------------------------- LAION head (as provided) -----------------------------
def get_aesthetic_model(clip_model: str = "vit_l_14"):
    """load the aesthetic model (LAION linear head on CLIP features)"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError(f"Unsupported clip_model: {clip_model}")
    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m

# ----------------------------- CLIP feature extraction -----------------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def preprocess_clip(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize to size×size and normalize for CLIP; return [3,H,W] tensor."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(_CLIP_MEAN)[None, None, :]) / np.array(_CLIP_STD)[None, None, :]
    arr = arr.transpose(2, 0, 1)  # HWC->CHW
    return torch.from_numpy(arr)

def load_clip_openai(clip_model: str, device: torch.device):
    """Load OpenAI CLIP model; return (encode_image_fn, embed_dim)."""
    import clip  # openai-clip
    name_map = {"vit_b_32": "ViT-B/32", "vit_l_14": "ViT-L/14"}
    if clip_model not in name_map:
        raise ValueError(f"clip_model must be one of {list(name_map.keys())}")
    model, _ = clip.load(name_map[clip_model], device=device, jit=False)
    model.eval()

    def encode_image_fn(img_chw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = img_chw.unsqueeze(0).to(device)
            feat = model.encode_image(x).float()
            feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat.squeeze(0)

    with torch.no_grad():
        d = int(encode_image_fn(torch.zeros(3, 224, 224, device=device)).numel())
    return encode_image_fn, d

def load_clip_jit(jit_path: str, device: torch.device):
    """Load TorchScript JIT CLIP; return (encode_image_fn, embed_dim)."""
    model = torch.jit.load(os.path.expanduser(jit_path), map_location=device)
    model.eval()

    def encode_image_fn(img_chw: torch.Tensor) -> torch.Tensor:
        x = img_chw.unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "encode_image"):
                feat = model.encode_image(x)
            elif hasattr(model, "visual"):
                feat = model.visual(x)
            else:
                out = model(x, None)
                if isinstance(out, dict) and "image_features" in out:
                    feat = out["image_features"]
                elif isinstance(out, (tuple, list)) and len(out) > 0:
                    feat = out[0]
                else:
                    raise RuntimeError("JIT CLIP does not expose image features. Check your checkpoint.")
            feat = feat.float()
            feat = torch.nn.functional.normalize(feat, dim=-1)
            return feat.squeeze(0)

    with torch.no_grad():
        d = int(encode_image_fn(torch.zeros(3, 224, 224, device=device)).numel())
    return encode_image_fn, d

# ----------------------------- Folder parsing -----------------------------
PROMPT_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[0-9.]+)_s(?P<step>\d+)$")

def parse_token(name: str) -> Dict[str, str]:
    """Parse '..._seed1111_g3.0_s30' from parent folder or filename stem."""
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

def walk_images(root: Path, exts=(".png", ".jpg", ".jpeg", ".webp")) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)

def parse_method_concept(dirname: str) -> Tuple[str, str]:
    return dirname.split("_", 1) if "_" in dirname else (dirname, "")

# ----------------------------- Evaluate one pair -----------------------------
def eval_one_pair(
    outputs_root: Path,
    method: str,
    concept: str,
    encode_image,
    laion_head: nn.Module,
    device: torch.device,
) -> Optional[pd.DataFrame]:
    """Evaluate a single (method, concept). Writes agg CSV and returns the agg DataFrame."""
    mc_dir = outputs_root / f"{method}_{concept}"
    imgs_dir = mc_dir / "imgs"
    eval_dir = mc_dir / "eval"
    if not imgs_dir.exists():
        print(f"[WARN] Skip: {imgs_dir} not found.")
        return None
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in walk_images(imgs_dir):
        parent_token = img_path.parent.name
        stem_token = img_path.stem
        meta = parse_token(parent_token)
        if meta["seed"] == "" and meta["guidance"] == "" and meta["step"] == "":
            meta = parse_token(stem_token)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open: {img_path} (skip). Reason: {e}")
            continue

        x = preprocess_clip(img, size=224).to(device)
        with torch.no_grad():
            feat = encode_image(x)
            score = laion_head(feat.unsqueeze(0)).item()  # higher = better

        rows.append({
            "method": method,
            "concept": concept,
            "guidance": str(meta["guidance"]),
            "score": float(score),
        })

    if not rows:
        print(f"[WARN] No images under {imgs_dir}.")
        agg = pd.DataFrame(columns=["method","concept","guidance","n","mean","std"])
    else:
        df = pd.DataFrame(rows)
        g = df.groupby(["method","concept","guidance"])["score"]
        agg = pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "std": g.std(ddof=1)
        }).reset_index().sort_values(["method","concept","guidance"])

    out_csv = eval_dir / "clip_iqa_agg.csv"
    agg.to_csv(out_csv, index=False)
    print(f"[DONE] {method}_{concept} -> {out_csv}")
    return agg

# ----------------------------- Main: multiple methods & concepts -----------------------------
def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # support comma-separated or space-separated
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p for p in chunk.split() if p])
    return parts

def auto_discover_pairs(outputs_root: Path) -> List[Tuple[str,str]]:
    pairs = []
    for d in outputs_root.iterdir():
        if not d.is_dir():
            continue
        m, c = parse_method_concept(d.name)
        if (d / "imgs").exists():
            pairs.append((m, c))
    return sorted(set(pairs))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute CLIP-IQA (LAION head) for multiple (method, concept) pairs.")
    ap.add_argument("--outputs_root", type=str, default="/mnt/data6t/yyz/flow_grpo/flow_base/outputs", help="Root folder containing <method>_<concept> subfolders.")
    ap.add_argument("--methods", type=str, default=None, help="Methods (comma/space separated). Example: 'ourmethod,baselineA baselineB'")
    ap.add_argument("--concepts", type=str, default=None, help="Concepts (comma/space separated). Example: 'truck,cat'")
    ap.add_argument("--auto_discover", action="store_true", help="If set, scan outputs_root to discover all <method>_<concept> pairs.")
    ap.add_argument("--clip_model", type=str, default="vit_b_32", choices=["vit_b_32","vit_l_14"], help="OpenAI CLIP backbone to use (ignored if --jit_path set).")
    ap.add_argument("--jit_path", type=str, default="~/.cache/clip/ViT-B-32.pt", help="Path to TorchScript CLIP (e.g., ~/.cache/ViT-B-32.pt). If set, overrides --clip_model.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    assert outputs_root.exists(), f"outputs_root not found: {outputs_root}"

    # Resolve list of (method, concept) pairs
    pairs: List[Tuple[str,str]] = []
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

    # Device
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    # Load CLIP + LAION head ONCE
    if args.jit_path:
        encode_image, feat_dim = load_clip_jit(args.jit_path, device)
        if feat_dim == 512:
            clip_variant = "vit_b_32"
        elif feat_dim == 768:
            clip_variant = "vit_l_14"
        else:
            raise ValueError(f"Unsupported JIT feature dim {feat_dim}. Expect 512 (ViT-B/32) or 768 (ViT-L/14).")
    else:
        encode_image, _ = load_clip_openai(args.clip_model, device)
        clip_variant = args.clip_model
    laion_head = get_aesthetic_model(clip_variant).to(device).eval()

    # Run all pairs
    all_aggs = []
    for (method, concept) in pairs:
        agg = eval_one_pair(outputs_root, method, concept, encode_image, laion_head, device)
        if agg is not None and not agg.empty:
            all_aggs.append(agg)

    # Write master CSV at outputs_root level
    if all_aggs:
        master = pd.concat(all_aggs, ignore_index=True)
        master_csv = outputs_root / "clip_iqa_agg_all.csv"
        master.to_csv(master_csv, index=False)
        print(f"[DONE] Master aggregated CSV -> {master_csv}")
    else:
        print("[INFO] No aggregated data to write at master level.")

if __name__ == "__main__":
    main()