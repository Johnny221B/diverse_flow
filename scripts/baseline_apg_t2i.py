# -*- coding: utf-8 -*-
import json
import argparse
import os
import sys
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
from diffusers import StableDiffusion3Pipeline

# 将项目根目录添加到路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from oscar.utils import (
    slugify,
    resolve_model_dir,
    parse_concepts_spec,
)

def parse_args():
    ap = argparse.ArgumentParser(description='Baseline: APG for SD3.5 (T2I CompBench Optimized)')
    ap.add_argument('--spec', type=str, required=True, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--method', type=str, default='apg_t2i')
    ap.add_argument('--model-dir', type=str, required=True)
    
    ap.add_argument('--G', type=int, default=16, help='Total samples per prompt')
    ap.add_argument('--micro-batch', type=int, default=4, help='Avoid OOM')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1111)
    
    # APG 潜在参数
    ap.add_argument('--eta-apg', type=float, default=0.15)
    
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--fp16', action='store_true')
    
    return ap.parse_args()

def _log(s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32

    # 1. 加载模型
    model_path = resolve_model_dir(args.model_dir)
    _log(f"Loading SD3.5 for APG Baseline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True
    )

    # 显存优化
    pipe.enable_model_cpu_offload(device=device) 
    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    # 2. 解析 Spec
    with open(args.spec, "r", encoding="utf-8") as f:
        spec_data = json.load(f, object_pairs_hook=OrderedDict)
    concept_to_prompts = spec_data

    # 3. 循环生成
    outputs_root = os.path.join(REPO_ROOT, 'outputs')

    for concept, prompts in concept_to_prompts.items():
        base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
        imgs_root_dir = os.path.join(base_out_dir, "imgs")
        os.makedirs(imgs_root_dir, exist_ok=True)
        os.makedirs(os.path.join(base_out_dir, "eval"), exist_ok=True)
        
        _log(f"\n>>> [APG-t2i] Concept: {concept}")

        for prompt_text in prompts:
            pslug = slugify(prompt_text)
            run_dir = os.path.join(imgs_root_dir, pslug)

            if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{prompt_text[:40]}...' already exists.")
                continue
            
            os.makedirs(run_dir, exist_ok=True)
            _log(f"  [RUN] Prompt: '{prompt_text}'")

            generated_count = 0
            while generated_count < args.G:
                current_bs = min(args.micro_batch, args.G - generated_count)
                # 保证每个 micro-batch 种子不同但可复现
                sub_generator = torch.Generator(device=device).manual_seed(args.seed + generated_count)
                
                with torch.inference_mode():
                    images = pipe(
                        prompt=prompt_text,
                        height=512, width=512,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        num_images_per_prompt=current_bs,
                        generator=sub_generator,
                        output_type="pil"
                    ).images
                
                for idx, img in enumerate(images):
                    img.save(os.path.join(run_dir, f"{generated_count + idx:03d}.png"))
                
                generated_count += current_bs
                del images
                gc.collect()
                torch.cuda.empty_cache()

    _log("APG Baseline T2I Tasks Completed.")

if __name__ == "__main__":
    main()