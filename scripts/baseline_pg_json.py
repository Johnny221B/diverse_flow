# -*- coding: utf-8 -*-
"""
Grid evaluation for Particle Guidance (PG) on SD3.5 (Flow Matching).

- Inputs:
  * Either a DIM/CIM spec JSON (concept + grouped prompts),
    or a single --prompt string (back-compat).
  * Multiple guidance scales (--guidances) and multiple seeds (--seeds).

- Outputs:
  outputs/{method}_{concept}/
    ├── eval/             # reserved for metrics
    └── imgs/
        └── {prompt_slug}_seed{SEED}_g{GUIDANCE}_s{STEPS}/
            00.png, 01.png, ...

Example:
  python -u scripts/pg_eval_grid.py \
    --model "./models/stable-diffusion-3.5-medium" \
    --spec ./specs/truck.json \
    --negative "low quality, blurry" \
    --G 16 --steps 10 \
    --guidances 3.0 7.5 12.0 \
    --seeds 1111 2222 3333 4444 \
    --height 512 --width 512 \
    --device cuda:0 --fp16 \
    --method pg

NOTE:
- This script reuses FlowSampler/PGConfig from Particle_Guidance.
- It (re)builds FlowSampler per guidance for clarity & correctness.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

# Repo root for imports
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# PG backend (your code)
from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything, slugify


# ---------------------------
# Utilities
# ---------------------------
def _flatten_prompts_from_spec(spec: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Parse DIM/CIM-style spec:
    {
        "concept": "truck",
        "simple_attribute_prompts": {
            "color": {"red": "a red truck", ...},
            "accessories": {...},
            ...
        }
    }
    Returns: (concept, [prompt1, prompt2, ...])
    """
    concept = spec.get("concept", "concept")
    out: List[str] = []
    sap = spec.get("simple_attribute_prompts", {})
    for _, mp in sap.items():
        if isinstance(mp, dict):
            for _, txt in mp.items():
                if isinstance(txt, str) and txt.strip():
                    out.append(txt.strip())
    # de-dup while keeping order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if not uniq:
        raise ValueError("[PG] No valid prompts parsed from spec.simple_attribute_prompts")
    return concept, uniq


def _outputs_root(method: str, concept: str) -> Tuple[Path, Path, Path]:
    """
    Create:
      outputs/{method}_{concept}/eval
      outputs/{method}_{concept}/imgs
    """
    base = Path(REPO_DIR) / "outputs" / f"{method}_{slugify(concept)}"
    eval_dir = base / "eval"
    imgs_dir = base / "imgs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, imgs_dir


def _prompt_run_dir(imgs_root: Path, prompt: str, seed: int, guidance: float, steps: int) -> Path:
    pslug = slugify(prompt)  # spaces -> _, lowercase, etc.
    # keep guidance literal; you can change formatting here if needed
    return imgs_root / f"{pslug}_seed{seed}_g{guidance}_s{steps}"


def _save_images(imgs: List[Image.Image], out_dir: Path, wh: Tuple[int, int]):
    W, H = int(wh[0]), int(wh[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        # enforce uniform size
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")


# ---------------------------
# Args
# ---------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (grid evaluation)")

    # Input spec OR single prompt
    p.add_argument("--spec", type=str, default=None, help="Path to DIM/CIM JSON spec")
    p.add_argument("--prompt", type=str, default=None, help="Fallback single prompt if --spec not provided")
    p.add_argument("--negative", type=str, default=None)

    # Grid
    p.add_argument("--G", type=int, default=4, help="Group size (images per prompt)")
    p.add_argument("--steps", type=int, default=30, help="#flow steps (NFE)")
    p.add_argument("--guidances", type=float, nargs="+", default=None, help="List of cfg scales; e.g., 3.0 7.5 12.0")
    p.add_argument("--cfg", type=float, default=5.0, help="Single cfg if --guidances is omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444])

    # Resolution
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Model/device
    p.add_argument("--model", type=str, required=True, help="Path to SD3.5 (Diffusers layout)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attention-slicing", action="store_true")

    # PG hyper-params
    p.add_argument("--pg-mode", type=str, choices=["ode", "sde"], default="ode")
    p.add_argument("--pg-alpha-scale", type=float, default=30.0,
                   help="alpha_t = alpha_scale * sigma(t)^2")
    p.add_argument("--pg-sigma-a", type=float, default=0.5,
                   help="sigma(t)=a*sqrt(t/(1-t))")
    p.add_argument("--pg-bandwidth", type=float, default=None,
                   help="Fixed RBF bandwidth; None -> median trick")

    # For directory naming
    p.add_argument("--method", type=str, default="pg", help="Method name for outputs/{method}_{concept}")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_args()

    # Parse prompts
    concept = "single"
    prompts: List[str] = []
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        concept, prompts = _flatten_prompts_from_spec(spec)
    elif args.prompt:
        concept, prompts = "single", [args.prompt]
    else:
        raise ValueError("Provide either --spec (DIM/CIM JSON) or --prompt")

    guidances = args.guidances if args.guidances is not None else [args.cfg]
    guidances = [float(g) for g in guidances]

    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    # Prepare output directories
    base_dir, eval_dir, imgs_root = _outputs_root(args.method, concept)
    print(f"[PG] outputs base: {base_dir}")
    print(f"[PG] eval dir:     {eval_dir}")
    print(f"[PG] imgs root:    {imgs_root}")

    # PG static config (per run we only vary cfg & seed)
    pg_cfg = PGConfig(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )

    # Run grid: prompt × guidance × seed
    for ptxt in prompts:
        for g in guidances:
            # Build a FlowSampler per-guidance to guarantee cfg applied
            fs_cfg = FlowSamplerConfig(
                model_path=args.model,
                device=str(device),
                dtype=dtype,
                steps=args.steps,
                cfg_scale=float(g),
                height=args.height,
                width=args.width,
                enable_attention_slicing=args.attention_slicing,
            )
            sampler = FlowSampler(fs_cfg, pg_cfg)

            for sd in args.seeds:
                # Reproducibility per (prompt, guidance, seed)
                seed_everything(int(sd))

                out_dir = _prompt_run_dir(
                    imgs_root=imgs_root,
                    prompt=ptxt,
                    seed=int(sd),
                    guidance=float(g),
                    steps=int(args.steps),
                )
                print(f"[PG] sampling: prompt='{ptxt}' | seed={sd} | cfg={g} | steps={args.steps} -> {out_dir}")

                images = sampler.generate(
                    prompt=ptxt,
                    negative_prompt=args.negative,
                    num_images=args.G,
                    seed=int(sd),
                )
                _save_images(images, out_dir, (args.width, args.height))

    print("[PG] Done.")


if __name__ == "__main__":
    main()
