# -*- coding: utf-8 -*-
import json
import argparse
import os
import sys
import gc
import time
import re
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
)

def parse_args():
    ap = argparse.ArgumentParser(description='Baseline: Mix-Flow (SDE+ODE) T2I CompBench Version')
    ap.add_argument('--spec', type=str, required=True, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--method', type=str, default='mix_flow_t2i')
    ap.add_argument('--model-dir', type=str, required=True)

    ap.add_argument('--G', type=int, default=16, help='Total samples per prompt')
    ap.add_argument('--micro-batch', type=int, default=4, help='Avoid OOM by splitting G')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1111)

    # Mix-Flow 特有参数
    ap.add_argument('--t-gate', type=float, default=0.7, help='Time gate to switch SDE to ODE')
    ap.add_argument('--eta-sde', type=float, default=1.0, help='Noise strength')

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
    _log(f"Loading SD3.5 for Mix-Flow Baseline from {model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True
    )

    # 2-GPU component placement
    pipe.transformer.to(dev_tr)
    pipe.vae.to(dev_vae)
    if pipe.text_encoder:   pipe.text_encoder.to(torch.device(args.device_text1))
    if pipe.text_encoder_2: pipe.text_encoder_2.to(torch.device(args.device_text2))
    if pipe.text_encoder_3: pipe.text_encoder_3.to(torch.device(args.device_text3))

    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    # 3. 混合采样回调函数 — latents stay on dev_tr throughout; randn_like is device-agnostic
    def mix_callback(ppl, i, t, kw):
        latents = kw.get("latents")
        if latents is None: return kw

        num_steps = getattr(ppl, "num_timesteps", None) or args.steps
        t_norm = 1.0 - (i / max(1, num_steps))
        dt = 1.0 / max(1, num_steps)

        if t_norm > args.t_gate:
            noise = torch.randn_like(latents)
            noise_std = args.eta_sde * (dt ** 0.5)
            kw["latents"] = latents + noise_std * noise
        return kw

    # 4. 循环生成
    outputs_root = os.path.join(REPO_ROOT, 'outputs')

    for concept, prompts in concept_to_prompts.items():
        base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
        imgs_root_dir = os.path.join(base_out_dir, "imgs")
        os.makedirs(imgs_root_dir, exist_ok=True)
        os.makedirs(os.path.join(base_out_dir, "eval"), exist_ok=True)

        _log(f"\n>>> [Mix-Flow] Concept: {concept}")

        for ptxt in prompts:
            pslug = slugify(ptxt)
            run_dir = os.path.join(imgs_root_dir, pslug)

            if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{ptxt[:40]}...' exists.")
                continue

            os.makedirs(run_dir, exist_ok=True)
            _log(f"  [RUN] Prompt: '{ptxt}'")

            generated_count = 0
            while generated_count < args.G:
                current_bs = min(args.micro_batch, args.G - generated_count)
                sub_generator = torch.Generator(device=dev_tr).manual_seed(args.seed + generated_count)

                with torch.inference_mode():
                    latents = pipe(
                        prompt=ptxt,
                        height=512,
                        width=512,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        num_images_per_prompt=current_bs,
                        generator=sub_generator,
                        callback_on_step_end=mix_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                        output_type="latent",
                    ).images

                latents = latents.to(device=dev_vae, dtype=pipe.vae.dtype)
                with torch.no_grad():
                    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    images = pipe.image_processor.postprocess(decoded, output_type="pil")

                for idx, img in enumerate(images):
                    img.save(os.path.join(run_dir, f"{generated_count + idx:03d}.png"))

                generated_count += current_bs
                del images, latents, decoded
                gc.collect()
                torch.cuda.empty_cache()

    _log("Mix-Flow Baseline Tasks Completed.")

if __name__ == "__main__":
    main()
