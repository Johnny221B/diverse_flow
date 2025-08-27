# Repository layout (all files below). Copy into your project as-is.

# ===============================
# FILE: scripts/main_particle_flow.py
# ===============================
# -*- coding: utf-8 -*-
"""
Main entry: prompt -> images via Stable Diffusion 3.5 (Flow Matching) + Particle Guidance (fixed potential).

Usage (example):
    python -u scripts/main_particle_flow.py \
        --model "./models/stable-diffusion-3.5-medium" \
        --prompt "a cozy cabin in a snowy forest" \
        --negative "low quality, blurry" \
        --G 4 --steps 30 --cfg 5.0 \
        --height 1024 --width 1024 \
        --out ./outputs/cabin \
        --device cuda:0 --seed 42 --fp16
"""
from __future__ import annotations
import argparse
import os
import sys
import torch

# Allow running from repo root
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything, save_images_grid, slugify


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (Flow Matching)")
    p.add_argument("--model", type=str, required=True, help="Path to SD3.5 directory (Diffusers layout)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative", type=str, default=None)
    p.add_argument("--G", type=int, default=4, help="Group size (images per prompt)")
    p.add_argument("--steps", type=int, default=30, help="#flow steps")
    p.add_argument("--cfg", type=float, default=5.0, help="Classifier-Free Guidance scale")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--out", type=str, default="./outputs/pg_run", help="Output directory")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)

    # PG hyper-params
    p.add_argument("--pg-mode", type=str, choices=["ode", "sde"], default="ode")
    p.add_argument("--pg-alpha-scale", type=float, default=30.0, help="alpha_t = alpha_scale * sigma(t)^2")
    p.add_argument("--pg-sigma-a", type=float, default=0.5, help="sigma(t)=a*sqrt(t/(1-t))")
    p.add_argument("--pg-bandwidth", type=float, default=None, help="fixed bandwidth; None -> median trick")

    # Performance toggles
    p.add_argument("--fp16", action="store_true", help="Use float16 where possible")
    p.add_argument("--attention-slicing", action="store_true")

    return p.parse_args()


def main():
    args = build_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    seed_everything(args.seed)

    # Configure Particle Guidance
    pg_cfg = PGConfig(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )

    # Configure Flow Sampler (wraps Diffusers SD3 pipeline)
    fs_cfg = FlowSamplerConfig(
        model_path=args.model,
        device=str(device),
        dtype=dtype,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=args.height,
        width=args.width,
        enable_attention_slicing=args.attention_slicing,
    )

    sampler = FlowSampler(fs_cfg, pg_cfg)

    # Ensure output dir exists
    base_out = args.out  # honor user-specified path exactly
    os.makedirs(base_out, exist_ok=True)

    images = sampler.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_images=args.G,
        seed=args.seed,
    )
    
    slug = slugify(args.prompt)
    out_root = args.out or "outputs"
    run_dir = os.path.join(out_root,f"PG_{slug}")
    imgs_dir = os.path.join(run_dir, "imgs")
    evals_dir = os.path.join(run_dir, "eval")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(evals_dir, exist_ok=True)
    

    # Save each image and a grid
    saved = []
    for i, img in enumerate(images):
        path = os.path.join(imgs_dir, f"{i:02d}.png")
        img.save(path)
        saved.append(path)
        
    grid_path = os.path.join(evals_dir, "grid.png")
    save_images_grid(images, grid_path)

    print("[PG] Saved:")
    for pth in saved:
        print("  ", pth)
    print("[PG] Grid:", grid_path)


if __name__ == "__main__":
    main()
