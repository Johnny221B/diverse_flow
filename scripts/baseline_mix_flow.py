# -*- coding: utf-8 -*-
import json
import argparse
import os
import sys
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
from diffusers import StableDiffusion3Pipeline

# 将项目根目录添加到路径
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oscar.utils import (
    log as _log,
    resolve_model_dir as _resolve_model_dir,
    parse_concepts_spec as _parse_concepts_spec,
    build_root_out as _build_root_out,
    prompt_run_dir as _prompt_run_dir,
)

def parse_args():
    ap = argparse.ArgumentParser(description='Baseline: Early-SDE + Late-ODE (OOM Optimized)')
    ap.add_argument('--spec', type=str, default=None)
    ap.add_argument('--prompt', type=str, default=None)
    ap.add_argument('--model-dir', type=str, required=True)
    ap.add_argument('--method', type=str, default='mix_sde_ode')
    
    ap.add_argument('--G', type=int, default=64, help='Total samples per prompt')
    ap.add_argument('--micro-batch', type=int, default=4, help='Samples per forward pass to avoid OOM')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidances', type=float, nargs='+', default=[3.0, 5.0, 7.5])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])
    
    ap.add_argument('--t-gate', type=float, default=0.7)
    ap.add_argument('--eta-sde', type=float, default=1.0)
    
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    
    return ap.parse_args()

def get_dtype(name):
    if name == 'fp16': return torch.float16
    if name == 'bf16': return torch.bfloat16
    return torch.float32

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    # 1. 加载模型并应用显存优化
    model_path = _resolve_model_dir(Path(args.model_dir))
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True
    )
    
    # --- 修改后的关键显存优化 ---
    
    # A. 核心优化：将不参与当前计算的模块（如巨大的 T5 编码器）移出显存
    pipe.enable_model_cpu_offload(device=device) 
    
    # B. 修复 AttributeError：直接访问 vae 模块开启 tiling
    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
        # 如果显存依然非常紧张，可以取消下面这一行的注释：
        # pipe.vae.enable_slicing() 
        
    # C. 减少不必要的开销
    pipe.set_progress_bar_config(disable=True) 
    
    # --------------------------
    # 2. 解析 Prompt
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as f:
            spec = json.load(f, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec)
    else:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])

    # 3. 混合采样回调
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
    for concept, prompts in concept_to_prompts.items():
        _, imgs_root, _ = _build_root_out(REPO_ROOT, args.method, concept)
        
        for g in args.guidances:
            for ptxt in prompts:
                for sd in args.seeds:
                    out_dir = _prompt_run_dir(imgs_root, ptxt, sd, g, args.steps)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    _log(f"Running Concept: {concept} | g={g} | seed={sd}", True)
                    
                    generated_count = 0
                    while generated_count < args.G:
                        # 计算本次 Batch 大小，防止最后一次溢出
                        current_bs = min(args.micro_batch, args.G - generated_count)
                        
                        # 为每个 micro-batch 使用不同的子种子以保证多样性
                        sub_generator = torch.Generator(device=device).manual_seed(sd + generated_count)
                        
                        with torch.inference_mode():
                            images = pipe(
                                prompt=ptxt,
                                height=512,
                                width=512,
                                num_inference_steps=args.steps,
                                guidance_scale=g,
                                num_images_per_prompt=current_bs,
                                generator=sub_generator,
                                callback_on_step_end=mix_callback,
                                callback_on_step_end_tensor_inputs=["latents"],
                                output_type="pil"
                            ).images
                        
                        for idx, img in enumerate(images):
                            img.save(os.path.join(out_dir, f"{generated_count + idx:02d}.png"))
                        
                        generated_count += current_bs
                        
                        # 每跑完一个 micro-batch 彻底清理显存
                        del images
                        gc.collect()
                        torch.cuda.empty_cache()

    _log("Task Completed.", True)

if __name__ == "__main__":
    main()