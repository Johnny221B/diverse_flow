#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline: PG (Particle Guidance) for SD3.5 - T2I CompBench Version
适配批量生成与目录管理，输出命名包含 t2i。
"""

from __future__ import annotations
import argparse
import os
import sys
import torch
import json
import time
from collections import OrderedDict

# Allow running from repo root
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything, save_images_grid, slugify
from oscar.utils import resolve_model_dir

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (T2I Batch)")
    # --- 批量测试参数 ---
    p.add_argument("--spec", type=str, default=None, help="Path to JSON spec file")
    p.add_argument("--method", type=str, default="pg_t2i", help="Method name suffix")
    
    p.add_argument("--model", type=str, required=True, help="Path to SD3.5 directory")
    p.add_argument("--prompt", type=str, default=None, help="Fallback single prompt")
    p.add_argument("--negative", type=str, default=None)
    p.add_argument("--G", type=int, default=16, help="Images per prompt")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1111)

    # PG 核心参数
    p.add_argument("--pg-mode", type=str, choices=["ode", "sde"], default="ode")
    p.add_argument("--pg-alpha-scale", type=float, default=30.0)
    p.add_argument("--pg-sigma-a", type=float, default=0.5)
    p.add_argument("--pg-bandwidth", type=float, default=None)

    # 性能优化
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attention-slicing", action="store_true")

    return p.parse_args()

def _log(s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def main():
    args = build_args()
    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    # 1. 解析 Spec
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as f:
            concept_to_prompts = json.load(f, object_pairs_hook=OrderedDict)
    elif args.prompt:
        concept_to_prompts = {"single": [args.prompt]}
    else:
        raise ValueError("Must provide --spec or --prompt")

    # 2. 初始化 Sampler (只加载一次模型)
    _log("Initializing PG Sampler...")
    pg_cfg = PGConfig(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )
    resolved_model_path = resolve_model_dir(args.model)
    _log(f"Resolved PG model path: {resolved_model_path}")
    fs_cfg = FlowSamplerConfig(
        model_path=str(resolved_model_path),
        device=str(device),
        dtype=dtype,
        steps=args.steps,
        cfg_scale=args.cfg,
        height=args.height,
        width=args.width,
        enable_attention_slicing=args.attention_slicing,
    )
    sampler = FlowSampler(fs_cfg, pg_cfg)

    # 3. 循环处理
    outputs_root = os.path.join(REPO_DIR, "outputs")

    for concept, prompts in concept_to_prompts.items():
        # 建立目录: outputs/pg_t2i_color/
        base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
        imgs_root_dir = os.path.join(base_out_dir, "imgs")
        eval_dir = os.path.join(base_out_dir, "eval")
        
        os.makedirs(imgs_root_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        _log(f"\n>>> Concept: {concept} | Output: {base_out_dir}")

        for prompt_text in prompts:
            pslug = slugify(prompt_text)
            run_dir = os.path.join(imgs_root_dir, pslug)

            # SKIP 逻辑
            if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{prompt_text[:40]}...' already generated.")
                continue

            os.makedirs(run_dir, exist_ok=True)
            _log(f"  [RUN] Prompt: '{prompt_text}'")

            # 固定随机种子
            seed_everything(args.seed)

            # 生成图片
            # 注意: PG Sampler 内部会根据 num_images (G) 自动并行处理
            images = sampler.generate(
                prompt=prompt_text,
                negative_prompt=args.negative,
                num_images=args.G,
                seed=args.seed,
            )

            # 保存图片 (000.png 格式)
            for i, img in enumerate(images):
                save_path = os.path.join(run_dir, f"{i:03d}.png")
                img.save(save_path)

            # 每组生成完释放显存
            del images
            torch.cuda.empty_cache()

    _log("All PG-t2i tasks completed.")

if __name__ == "__main__":
    main()