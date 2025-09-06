# -*- coding: utf-8 -*-
"""
DIMCIM-style prompt generator for SD3.5 + Particle Guidance (PG).

Reads a single DIMCIM dense_prompts JSON, e.g.:
{
  "concept": "bus",
  "coarse_dense_prompts": [
    {
      "coco_seed_caption": "...",
      "coarse_prompt": "A bus ...",
      "dense_prompts": [
        {"attribute_type": "color", "attribute": "red", "dense_prompt": "A red bus ..."},
        ...
      ]
    },
    ...
  ]
}

Output layout (fixed by spec):
<OUTDIR>/
  pg_{concept}_CIMDIM/
    ├── eval/
    ├── coarse_imgs/
    │     └── <coarse_prompt_with_underscores>/
    │           00.png ... 19.png
    └── dense_imgs/
          └── <dense_prompt_with_underscores>/
                00.png ... 19.png

Defaults per request:
- images per prompt (group size): 20
- cfg_scale: 7.5
- steps: 50
- seed: 1234
- HxW: 512x512
- fp16: on by default
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

# Repo root for imports
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Your PG backend
from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything


# ---------------------------
# Utilities
# ---------------------------
def prompt_dirname(prompt: str) -> str:
    """
    Keep original case; remove punctuation; collapse spaces->underscores.
    Examples:
      "A bus stopping at a bus stop in a city." -> "A_bus_stopping_at_a_bus_stop_in_a_city"
      "a double-decker bus" -> "a_doubledecker_bus"
    """
    s = prompt.strip()
    # Remove punctuation except word chars and spaces/hyphens
    s = re.sub(r"[^\w\s-]", "", s)
    # Collapse hyphens within words (optional): keep for simplicity
    s = s.replace("-", "")
    # Collapse whitespace -> underscores
    s = re.sub(r"\s+", "_", s)
    # Trim underscores
    return s.strip("_")


def unique_preserve(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_dimcim_dense_json(path: Path) -> Tuple[str, List[str], List[str]]:
    """
    Return (concept, coarse_prompts, dense_prompts) with duplicates removed, order preserved.
    """
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    concept = str(data.get("concept", "unknown")).strip() or "unknown"

    coarse_list, dense_list = [], []
    entries = data.get("coarse_dense_prompts", [])
    if not isinstance(entries, list):
        raise ValueError("Invalid DIMCIM JSON: 'coarse_dense_prompts' must be a list.")

    for e in entries:
        if not isinstance(e, dict):
            continue
        c = e.get("coarse_prompt", None)
        if isinstance(c, str) and c.strip():
            coarse_list.append(c.strip())
        dlist = e.get("dense_prompts", [])
        if isinstance(dlist, list):
            for d in dlist:
                if isinstance(d, dict):
                    dp = d.get("dense_prompt", None)
                    if isinstance(dp, str) and dp.strip():
                        dense_list.append(dp.strip())

    coarse_u = unique_preserve(coarse_list)
    dense_u = unique_preserve(dense_list)

    if not coarse_u and not dense_u:
        raise ValueError("No valid prompts found in DIMCIM file.")
    return concept, coarse_u, dense_u


def _outputs_root(outdir: Path, method: str, concept: str) -> Tuple[Path, Path, Path, Path]:
    """
    Create:
      <outdir>/pg_{concept}_CIMDIM/{eval, coarse_imgs, dense_imgs}
    """
    base = outdir / f"{method}_{concept}_CIMDIM"
    eval_dir = base / "eval"
    coarse_dir = base / "coarse_imgs"
    dense_dir = base / "dense_imgs"
    for d in (eval_dir, coarse_dir, dense_dir):
        d.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, coarse_dir, dense_dir


def _save_images(imgs: List[Image.Image], out_dir: Path, wh: Tuple[int, int]):
    W, H = int(wh[0]), int(wh[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")


# ---------------------------
# Args
# ---------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images for DIMCIM prompts (SD3.5 + PG).")

    # DIMCIM spec
    p.add_argument("--dimcim-json", type=str, required=True,
                   help="Path to DIMCIM dense_prompts JSON, e.g., bus_dense_prompts.json")
    p.add_argument("--outdir", type=str, required=True,
                   help="Output root directory (will create pg_{concept}_CIMDIM under this)")

    # Fixed naming knobs
    p.add_argument("--method", type=str, default="pg",
                   help="Method name for folder prefix (default: pg)")

    # Generation knobs (defaults as requested)
    p.add_argument("--G", type=int, default=20, help="Images per prompt (group size)")
    p.add_argument("--steps", type=int, default=50, help="#flow steps (NFE)")
    p.add_argument("--cfg", type=float, default=7.5, help="CFG scale (fixed)")
    p.add_argument("--seed", type=int, default=1234, help="Base seed per prompt")
    p.add_argument("--negative", type=str, default=None)

    # Resolution
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Model/device
    p.add_argument("--model", type=str, required=True,
                   help="Path to SD3.5 (Diffusers layout)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fp16", action="store_true", help="Use float16 for model & sampling")
    p.set_defaults(fp16=True)  # default ON as requested
    p.add_argument("--attention-slicing", action="store_true")

    # Toggles
    p.add_argument("--skip-coarse", action="store_true", help="Skip generating coarse prompts")
    p.add_argument("--skip-dense", action="store_true", help="Skip generating dense prompts")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_args()

    dimcim_path = Path(args.dimcim_json)
    if not dimcim_path.exists():
        raise FileNotFoundError(f"DIMCIM JSON not found: {dimcim_path}")

    # Parse DIMCIM prompts
    concept, coarse_prompts, dense_prompts = parse_dimcim_dense_json(dimcim_path)

    # Prepare outputs
    out_root = Path(args.outdir)
    base_dir, eval_dir, coarse_dir, dense_dir = _outputs_root(out_root, args.method, concept)
    print(f"[PG] base:  {base_dir}")
    print(f"[PG] eval:  {eval_dir}")
    print(f"[PG] coarse:{coarse_dir}")
    print(f"[PG] dense: {dense_dir}")

    # Sampler config (single CFG per request)
    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    pg_cfg = PGConfig(
        group_size=args.G,
        mode="ode",              # keep PG default unless you want to expose a flag
        alpha_scale=30.0,
        sigma_a=0.5,
        bandwidth=None,
    )

    fs_cfg = FlowSamplerConfig(
        model_path=args.model,
        device=str(device),
        dtype=dtype,
        steps=int(args.steps),
        cfg_scale=float(args.cfg),
        height=int(args.height),
        width=int(args.width),
        enable_attention_slicing=args.attention_slicing,
    )
    sampler = FlowSampler(fs_cfg, pg_cfg)

    def do_generate(ptxt: str, dest_parent: Path):
        seed_everything(int(args.seed))
        subdir = prompt_dirname(ptxt)
        out_dir = dest_parent / subdir
        print(f"[PG] sampling: concept='{concept}' | images={args.G} | cfg={args.cfg} | steps={args.steps}")
        print(f"     prompt: {ptxt}")
        print(f"     -> {out_dir}")
        images = sampler.generate(
            prompt=ptxt,
            negative_prompt=args.negative,
            num_images=args.G,
            seed=int(args.seed),
        )
        _save_images(images, out_dir, (args.width, args.height))

    # Generate COARSE
    if not args.skip_coarse:
        for cp in coarse_prompts:
            do_generate(cp, coarse_dir)

    # Generate DENSE
    if not args.skip_dense:
        for dp in dense_prompts:
            do_generate(dp, dense_dir)

    print("[PG] Done.")


if __name__ == "__main__":
    main()
