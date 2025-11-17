# -*- coding: utf-8 -*-
"""
Grid evaluation for Particle Guidance (PG) on SD3.5 (Flow Matching).

- Inputs:
  * New JSON format: { "dog": [...], "truck": [...], ... }
    (top-level keys are concepts; each value is a list of prompts)
  * Or a single --prompt string (back-compat).

- Outputs:
  outputs/{method}_{concept}/
    ├── eval/
    │     └── {method}_{concept}_cost.csv     # 新增：开销统计
    └── imgs/
        └── {prompt_slug}_seed{SEED}_g{GUIDANCE}_s{STEPS}/
            00.png, 01.png, ...
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

# 可选：torch.profiler 用于 FLOPs 统计
try:
    import torch.profiler as torch_profiler
except Exception:
    torch_profiler = None

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
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (grid evaluation)")

    # Input spec OR single prompt
    p.add_argument("--spec", type=str, default=None,
                   help="Path to JSON: {concept: [prompts...]} (overrides --prompt)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt if --spec not provided")
    p.add_argument("--negative", type=str, default=None)

    # Grid
    p.add_argument("--G", type=int, default=32, help="Group size (images per prompt)")
    p.add_argument("--steps", type=int, default=30, help="#flow steps (NFE)")
    p.add_argument("--guidances", type=float, nargs="+", default=None,
                   help="List of cfg scales; e.g., 3.0 7.5 12.0")
    p.add_argument("--cfg", type=float, default=5.0,
                   help="Single cfg if --guidances is omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444, 5555, 6666])

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
                   help="Method name for outputs/{method}_{concept}")

    # Cost profiling
    p.add_argument(
        "--profile-flops",
        action="store_true",
        help="Use torch.profiler to estimate FLOPs per run (may add overhead)."
    )

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = build_args()

    # Parse source of prompts (preserve order in file)
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide either --spec (JSON file) or --prompt")

    # Helpful echo
    total_prompts = sum(len(v) for v in concept_to_prompts.values())
    print(f"[PG] concepts: {len(concept_to_prompts)} | total prompts: {total_prompts}")
    for k, v in concept_to_prompts.items():
        print(f"[PG]   - {k}: {len(v)} prompts")

    # Robust处理 guidances：支持 list / 单个 float / None
    if isinstance(args.guidances, (list, tuple)):
        guidances = [float(g) for g in args.guidances]
    elif args.guidances is None:
        guidances = [float(args.cfg)]
    else:
        guidances = [float(args.guidances)]

    dtype = torch.float16 if args.fp16 else torch.float32
    dtype_str = "fp16" if args.fp16 else "fp32"
    device = torch.device(args.device)

    # Static PG config
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

        # 准备 cost CSV：outputs/{method}_{concept}/eval/{method}_{concept}_cost.csv
        cost_csv_path = eval_dir / f"{args.method}_{slugify(concept)}_cost.csv"
        csv_new = not cost_csv_path.exists()
        cost_f = open(cost_csv_path, "a", encoding="utf-8", newline="")
        cost_writer = csv.writer(cost_f)
        if csv_new:
            cost_writer.writerow([
                "method",
                "concept",
                "prompt",
                "seed",
                "guidance",
                "steps",
                "num_images",
                "height",
                "width",
                "dtype",
                "device_transformer",  # PG: 统一记成 args.device
                "device_vae",
                "device_clip",
                "wall_time_total",
                "wall_time_per_image",
                "flops_total",
                "flops_per_image",
                "gpu_mem_peak_mb",
            ])

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

                        flops_total = -1.0
                        gpu_mem_peak_mb = -1.0

                        # reset peak memory on this device
                        if torch.cuda.is_available() and device.type == "cuda":
                            try:
                                torch.cuda.reset_peak_memory_stats(device)
                            except Exception as e:
                                print(f"[PG][WARN] could not reset peak memory stats on {device}: {e}")

                        t0 = time.perf_counter()

                        def _run_generate():
                            with torch.inference_mode():
                                return sampler.generate(
                                    prompt=ptxt,
                                    negative_prompt=args.negative,
                                    num_images=args.G,
                                    seed=int(sd),
                                )

                        if args.profile_flops and torch_profiler is not None:
                            activities = [torch_profiler.ProfilerActivity.CPU]
                            if torch.cuda.is_available():
                                activities.append(torch_profiler.ProfilerActivity.CUDA)
                            try:
                                with torch_profiler.profile(
                                    activities=activities,
                                    record_shapes=False,
                                    profile_memory=False,
                                    with_flops=True,
                                ) as prof:
                                    images = _run_generate()
                                try:
                                    flops_total = float(sum(
                                        e.flops for e in prof.key_averages()
                                        if hasattr(e, "flops") and e.flops is not None
                                    ))
                                except Exception as e:
                                    print(f"[PG][WARN] failed to aggregate FLOPs: {e}")
                                    flops_total = -1.0
                            except Exception as e:
                                print(f"[PG][WARN] FLOPs profiling failed, fallback without FLOPs. Error: {e}")
                                images = _run_generate()
                                flops_total = -1.0
                        else:
                            if args.profile_flops and torch_profiler is None:
                                print("[PG][WARN] torch.profiler not available; FLOPs will be -1.")
                            images = _run_generate()

                        _save_images(images, out_dir, (args.width, args.height))

                        t1 = time.perf_counter()
                        wall_time_total = float(t1 - t0)

                        # 读取峰值显存
                        if torch.cuda.is_available() and device.type == "cuda":
                            try:
                                peak_bytes = torch.cuda.max_memory_allocated(device)
                                gpu_mem_peak_mb = float(peak_bytes) / (1024.0 ** 2)
                            except Exception as e:
                                print(f"[PG][WARN] failed to get peak memory on {device}: {e}")
                                gpu_mem_peak_mb = -1.0
                        else:
                            gpu_mem_peak_mb = -1.0

                        num_images = int(len(images)) if images is not None else 0
                        wall_time_per_image = wall_time_total / num_images if num_images > 0 else -1.0
                        flops_per_image = flops_total / num_images if (num_images > 0 and flops_total > 0) else -1.0

                        # 写 cost 行（为了和 CADS / DPP 对齐，device_transformer 用 args.device，其它空）
                        cost_writer.writerow([
                            args.method,
                            concept,
                            ptxt,
                            int(sd),
                            float(g),
                            int(args.steps),
                            num_images,
                            int(args.height),
                            int(args.width),
                            dtype_str,
                            args.device,
                            "",
                            "",
                            f"{wall_time_total:.6f}",
                            f"{wall_time_per_image:.6f}",
                            f"{flops_total:.3f}",
                            f"{flops_per_image:.3f}",
                            f"{gpu_mem_peak_mb:.3f}",
                        ])
                        cost_f.flush()

                        # Explicitly release references & cached memory between seeds
                        del images
                        _cuda_clear()

            finally:
                # Release model/sampler for this guidance before moving to next
                del sampler
                _cuda_clear()
                cost_f.flush()

        cost_f.close()

    print("[PG] Done.")


if __name__ == "__main__":
    main()
