#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline: Vanilla Stable Diffusion 3.5 - T2I CompBench Version
最基础的基线：没有任何修饰，直接调用原版 SD3.5 模型生成图片。
"""

import os
import re
import sys
import time
import json
import argparse
import traceback
from typing import Any, Dict, Optional
from collections import OrderedDict

import torch
from diffusers import StableDiffusion3Pipeline

# 将项目根目录添加到路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from oscar.utils import resolve_model_dir

def _slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip())
    s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
    s = re.sub(r'_{2,}', '_', s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def parse_args():
    ap = argparse.ArgumentParser(description='Base Model Baseline (T2I Batch)')
    ap.add_argument('--spec', type=str, required=True, help='Path to JSON spec (e.g. mini_color.json)')
    ap.add_argument('--method', type=str, default='baseline_base', help='Method name prefix')
    ap.add_argument('--model-dir', type=str, required=True)
    
    ap.add_argument('--G', type=int, default=16, help='Number of images per prompt')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1111)

    ap.add_argument('--device', type=str, default='cuda:0', help='Primary device for offloading')
    ap.add_argument('--fp16', action='store_true')
    
    return ap.parse_args()

def _log(s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32

    # 1. 解析 Spec
    with open(args.spec, "r", encoding="utf-8") as f:
        concept_to_prompts = json.load(f, object_pairs_hook=OrderedDict)

    # 2. 加载模型
    model_path = resolve_model_dir(args.model_dir)
    
    _log(f"Loading SD3.5 Base Model from {model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(model_path), # 使用解析后的绝对路径
        torch_dtype=dtype, 
        local_files_only=True
    )
    
    # 显存优化组合拳：针对 G=16 必须开启
    # 使用官方推荐的 offload 模式，它会自动管理多卡显存分布
    pipe.enable_model_cpu_offload(device=device) 
    
    if hasattr(pipe, "vae"):
        # 修复之前提到的 AttributeError，直接访问模块
        pipe.vae.enable_tiling()
        
    pipe.set_progress_bar_config(disable=True)

    # --------------------- 循环处理 ---------------------
    outputs_root = os.path.join(REPO_ROOT, 'outputs')

    for concept, prompts in concept_to_prompts.items():
        # 建立 Concept 目录： outputs/baseline_base_color/
        base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
        imgs_root_dir = os.path.join(base_out_dir, "imgs")
        eval_dir = os.path.join(base_out_dir, "eval")
        
        os.makedirs(imgs_root_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        
        _log(f"\n>>> [BASE] Starting Concept: {concept} | Saving to: {base_out_dir}")

        for prompt_text in prompts:
            prompt_slug = _slugify(prompt_text)
            run_dir = os.path.join(imgs_root_dir, prompt_slug)

            # 断点续跑支持
            if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{prompt_text[:40]}...' already generated.")
                continue
            
            os.makedirs(run_dir, exist_ok=True)
            _log(f"  [RUN] Prompt: '{prompt_text}'")

            generator = torch.Generator(device=device).manual_seed(args.seed)

            # 调用原版 Pipeline
            images = pipe(
                prompt=prompt_text,
                height=512,
                width=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=args.G,
                generator=generator,
                output_type="pil"
            ).images

            # 保存图片 (000.png 格式适配 T2I-CompBench 评估)
            for i, img in enumerate(images):
                img.save(os.path.join(run_dir, f"{i:03d}.png"))

            # 及时回收显存
            del images
            torch.cuda.empty_cache()

    _log("Base Model (Vanilla ODE) Generation Tasks Completed.")

if __name__ == "__main__":
    main()