#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline: Vanilla Stable Diffusion 3.5 - T2I CompBench Version
最基础的基线：没有任何修饰，直接调用原版 SD3.5 模型生成图片。
支持双卡部署：transformer+text encoders 在 cuda:1，VAE 在 cuda:0。
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

    ap.add_argument('--device_transformer', type=str, default='cuda:1')
    ap.add_argument('--device_vae',         type=str, default='cuda:0')
    ap.add_argument('--device_text1',       type=str, default='cuda:1')
    ap.add_argument('--device_text2',       type=str, default='cuda:1')
    ap.add_argument('--device_text3',       type=str, default='cuda:1')
    ap.add_argument('--fp16', action='store_true')

    return ap.parse_args()

def _log(s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def main():
    args = parse_args()
    dtype = torch.float16 if args.fp16 else torch.float32

    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)

    # 1. 解析 Spec
    with open(args.spec, "r", encoding="utf-8") as f:
        concept_to_prompts = json.load(f, object_pairs_hook=OrderedDict)

    # 2. 加载模型
    model_path = resolve_model_dir(args.model_dir)

    _log(f"Loading SD3.5 Base Model from {model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True
    )

    # 2-GPU component placement (transformer+TEs on cuda:1, VAE on cuda:0)
    pipe.transformer.to(dev_tr)
    pipe.vae.to(dev_vae)
    if pipe.text_encoder:   pipe.text_encoder.to(torch.device(args.device_text1))
    if pipe.text_encoder_2: pipe.text_encoder_2.to(torch.device(args.device_text2))
    if pipe.text_encoder_3: pipe.text_encoder_3.to(torch.device(args.device_text3))

    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()

    pipe.set_progress_bar_config(disable=True)

    # --------------------- 循环处理 ---------------------
    outputs_root = os.path.join(REPO_ROOT, 'outputs')

    for concept, prompts in concept_to_prompts.items():
        base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
        imgs_root_dir = os.path.join(base_out_dir, "imgs")
        eval_dir = os.path.join(base_out_dir, "eval")

        os.makedirs(imgs_root_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        _log(f"\n>>> [BASE] Starting Concept: {concept} | Saving to: {base_out_dir}")

        for prompt_text in prompts:
            prompt_slug = _slugify(prompt_text)
            run_dir = os.path.join(imgs_root_dir, prompt_slug)

            if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{prompt_text[:40]}...' already generated.")
                continue

            os.makedirs(run_dir, exist_ok=True)
            _log(f"  [RUN] Prompt: '{prompt_text}'")

            generator = torch.Generator(device=dev_tr).manual_seed(args.seed)

            # Generate latents on transformer device, decode manually on VAE device
            latents = pipe(
                prompt=prompt_text,
                height=512,
                width=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=args.G,
                generator=generator,
                output_type="latent",
            ).images

            latents = latents.to(device=dev_vae, dtype=pipe.vae.dtype)
            with torch.no_grad():
                decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                images = pipe.image_processor.postprocess(decoded, output_type="pil")

            for i, img in enumerate(images):
                img.save(os.path.join(run_dir, f"{i:03d}.png"))

            del images, latents, decoded
            torch.cuda.empty_cache()

    _log("Base Model (Vanilla ODE) Generation Tasks Completed.")

if __name__ == "__main__":
    main()
