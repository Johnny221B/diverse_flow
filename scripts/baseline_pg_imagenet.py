# -*- coding: utf-8 -*-
"""
Grid evaluation for Particle Guidance (PG) on SD3.5 (Flow Matching).

- Mode A: Grid (原始逻辑)
  * Inputs:
    - JSON: { "dog": [...], "truck": [...], ... } (concept -> [prompts...])
    - 或单个 --prompt
  * Outputs:
    outputs/{method}_{concept}/
      ├── eval/             # reserved for metrics
      └── imgs/
          └── {prompt_slug}_seed{SEED}_g{GUIDANCE}_s{STEPS}/
              00.png, 01.png, ...

- Mode B: ImageNet-400 单图评估
  * Inputs:
    - --imagenet-json imagenet_400_prompts.json
      {
        "0": "a photo of a tench",
        "1": "a photo of a goldfish",
        ...
        "399": "a photo of an abaya"
      }
  * Outputs:
    outputs/imagenet_400/{method}/
      cls_000.png
      cls_001.png
      ...
      cls_399.png
"""

from __future__ import annotations
import argparse
import gc
import json
import os
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

# PG backend (your code)
from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything, slugify


# ---------------------------
# Utilities
# ---------------------------
def _parse_concepts_spec(spec: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    if not isinstance(spec, dict):
        raise ValueError("[PG] Spec must be a JSON object: {concept: [prompts...]}")
    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
        if cleaned:
            concept_to_prompts[concept] = cleaned
    if not concept_to_prompts:
        raise ValueError("[PG] No valid {concept: [prompts...]} found in spec.")
    return concept_to_prompts


def _outputs_root(method: str, concept: str) -> Tuple[Path, Path, Path]:
    base = Path(REPO_DIR) / "outputs" / f"{method}_{slugify(concept)}"
    eval_dir = base / "eval"
    imgs_dir = base / "imgs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, imgs_dir


def _prompt_run_dir(imgs_root: Path, prompt: str, seed: int, guidance: float, steps: int) -> Path:
    pslug = slugify(prompt)
    return imgs_root / f"{pslug}_seed{seed}_g{guidance}_s{steps}"


def _save_images(imgs: List[Image.Image], out_dir: Path, wh: Tuple[int, int]):
    W, H = int(wh[0]), int(wh[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")


def _cuda_clear():
    """Best-effort to release cached CUDA memory between iterations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# ---------------------------
# Args
# ---------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (grid / ImageNet-400 evaluation)")

    # Input spec OR single prompt OR ImageNet-400 json
    p.add_argument("--spec", type=str, default=None,
                   help="Path to JSON: {concept: [prompts...]} (overrides --prompt; ignored if --imagenet-json is given)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt if --spec/--imagenet-json not provided")
    p.add_argument("--imagenet-json", type=str, default=None,
                   help="ImageNet-400 prompt JSON: {class_id: prompt}")
    p.add_argument("--negative", type=str, default=None)

    # Grid (Mode A)
    p.add_argument("--G", type=int, default=1, help="Group size (images per prompt) in grid mode")
    p.add_argument("--steps", type=int, default=30, help="#flow steps (NFE)")
    p.add_argument("--guidances", type=float, nargs="+", default=3.0,
                   help="List of cfg scales; e.g., 3.0 7.5 12.0 (grid mode)")
    p.add_argument("--cfg", type=float, default=5.0,
                   help="Single cfg if --guidances is omitted; also used in ImageNet-400 mode")

    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444],
                   help="Grid mode: seeds list. ImageNet mode: seeds[0] as base if --seed is None.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for ImageNet-400 mode; if None, use seeds[0].")

    # Resolution
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # Model/device
    p.add_argument("--model", type=str, required=True,
                   help="Path to SD3.5 (Diffusers layout)")
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
    p.add_argument("--method", type=str, default="pg",
                   help="Method name for outputs/{method}_{concept} or outputs/imagenet_400/{method}")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    # ============================================================
    # Mode B: ImageNet-400 单图模式
    # ============================================================
    if args.imagenet_json is not None:
        imagenet_path = Path(args.imagenet_json)
        if not imagenet_path.exists():
            raise FileNotFoundError(f"[PG] imagenet-json not found: {imagenet_path}")

        with imagenet_path.open("r", encoding="utf-8") as fp:
            # 保持 class_id 顺序
            cls_to_prompt: Dict[str, str] = json.load(fp, object_pairs_hook=OrderedDict)

        base_seed = args.seed if args.seed is not None else (args.seeds[0] if args.seeds else 0)
        guidance = float(args.cfg)

        print(f"[PG][ImageNet-400] base_seed={base_seed}, cfg={guidance}, steps={args.steps}")
        print(f"[PG][ImageNet-400] #classes = {len(cls_to_prompt)}")

        # ImageNet-400 模式：我们按“单图/单粒子”配置 PG
        pg_cfg = PGConfig(
            group_size=1,                 # 只生成 1 张图 / 1 粒子
            mode=args.pg_mode,
            alpha_scale=args.pg_alpha_scale,
            sigma_a=args.pg_sigma_a,
            bandwidth=args.pg_bandwidth,
        )

        fs_cfg = FlowSamplerConfig(
            model_path=args.model,
            device=str(device),
            dtype=dtype,
            steps=args.steps,
            cfg_scale=guidance,
            height=args.height,
            width=args.width,
            enable_attention_slicing=args.attention_slicing,
        )
        sampler = FlowSampler(fs_cfg, pg_cfg)

        out_root = Path(REPO_DIR) / "outputs" / "imagenet_400" / args.method
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[PG][ImageNet-400] output dir: {out_root}")

        for cls_id_str, prompt in sorted(cls_to_prompt.items(), key=lambda kv: int(kv[0])):
            cls_id = int(cls_id_str)
            cur_seed = int(base_seed) + cls_id

            print(f"[PG][ImageNet-400] cls={cls_id:03d} seed={cur_seed} prompt='{prompt}'")
            seed_everything(cur_seed)

            with torch.inference_mode():
                images = sampler.generate(
                    prompt=prompt,
                    negative_prompt=args.negative,
                    num_images=1,     # 每个 prompt 一张图
                    seed=cur_seed,
                )

            img = images[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.size != (args.width, args.height):
                img = img.resize((int(args.width), int(args.height)), resample=Image.BICUBIC)

            filename = f"cls_{cls_id:03d}.png"
            img.save(out_root / filename)

            del images, img
            _cuda_clear()

        # 释放 sampler
        del sampler
        _cuda_clear()

        print("[PG][ImageNet-400] Done.")
        return

    # ============================================================
    # Mode A: 原始 grid 模式（多 concept）
    # ============================================================
    # Parse source of prompts (preserve order in file)
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide either --spec (JSON file), --prompt, or --imagenet-json")

    # Helpful echo
    total_prompts = sum(len(v) for v in concept_to_prompts.values())
    print(f"[PG] concepts: {len(concept_to_prompts)} | total prompts: {total_prompts}")
    for k, v in concept_to_prompts.items():
        print(f"[PG]   - {k}: {len(v)} prompts")

    guidances = args.guidances if args.guidances is not None else [args.cfg]
    guidances = [float(g) for g in guidances]

    # Static PG config for grid mode
    pg_cfg = PGConfig(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )

    # Iterate by concept to write into outputs/{method}_{concept}/...
    for concept, prompts in concept_to_prompts.items():
        base_dir, eval_dir, imgs_root = _outputs_root(args.method, concept)
        print(f"[PG] outputs base: {base_dir}")
        print(f"[PG] eval dir:     {eval_dir}")
        print(f"[PG] imgs root:    {imgs_root}")

        for g in guidances:
            # Build one sampler per (concept, guidance) and reuse for all prompts & seeds
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

            try:
                for ptxt in prompts:
                    for sd in args.seeds:
                        seed_everything(int(sd))
                        out_dir = _prompt_run_dir(
                            imgs_root=imgs_root,
                            prompt=ptxt,
                            seed=int(sd),
                            guidance=float(g),
                            steps=int(args.steps),
                        )
                        print(f"[PG] sampling: concept='{concept}' | prompt='{ptxt}' | seed={sd} | cfg={g} | steps={args.steps} -> {out_dir}")

                        with torch.inference_mode():
                            images = sampler.generate(
                                prompt=ptxt,
                                negative_prompt=args.negative,
                                num_images=args.G,
                                seed=int(sd),
                            )
                        _save_images(images, out_dir, (args.width, args.height))

                        # Explicitly release references & cached memory between seeds
                        del images
                        _cuda_clear()

            finally:
                # Release model/sampler for this guidance before moving to next
                del sampler
                _cuda_clear()

    print("[PG] Done.")


if __name__ == "__main__":
    main()
