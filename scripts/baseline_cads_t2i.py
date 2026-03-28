import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline

# 确保项目根目录在路径中
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 导入 CADS
try:
    from cads import CADS
except ImportError:
    from cads.cads import CADSConditionAnnealer as CADS

from oscar.utils import resolve_model_dir

def slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _log(s):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def parse_args():
    ap = argparse.ArgumentParser(description='CADS Multi-GPU Batch Generator')
    # 路径与方法
    ap.add_argument('--spec', type=str, required=True, help='Path to color.json')
    ap.add_argument('--method', type=str, default='cads')
    ap.add_argument('--category', type=str, default='color')
    ap.add_argument('--model-dir', type=str, default='models/stable-diffusion-3.5-medium')
    
    # 采样参数
    ap.add_argument('--G', type=int, default=16, help='Images per prompt')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1111)

    # CADS 参数
    ap.add_argument("--tau1", type=float, default=0.6)
    ap.add_argument("--tau2", type=float, default=0.9)
    ap.add_argument("--cads-s", type=float, default=0.10)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--no-rescale", action="store_true")

    # 严格按照你原来的分配
    ap.add_argument("--device_transformer", type=str, default="cuda:1")
    ap.add_argument("--device_vae",         type=str, default="cuda:0")
    ap.add_argument("--device_text1",       type=str, default="cuda:1")
    ap.add_argument("--device_text2",       type=str, default="cuda:1")
    ap.add_argument("--device_text3",       type=str, default="cuda:1")
    
    return ap.parse_args()

def main():
    args = parse_args()
    dev_tr = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dtype = torch.bfloat16

    # 1. 加载模型
    model_path = resolve_model_dir(args.model_dir)
    _log(f"Loading SD3.5 components...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    
    # 2. 组件分发
    pipe.transformer.to(dev_tr)
    pipe.vae.to(dev_vae)
    if pipe.text_encoder:   pipe.text_encoder.to(torch.device(args.device_text1))
    if pipe.text_encoder_2: pipe.text_encoder_2.to(torch.device(args.device_text2))
    if pipe.text_encoder_3: pipe.text_encoder_3.to(torch.device(args.device_text3))
    pipe.set_progress_bar_config(disable=True)

    # 3. 初始化 CADS 
    cads_annealer = CADS(
        num_inference_steps=args.steps,
        tau1=args.tau1, tau2=args.tau2,
        s=args.cads_s, psi=args.psi,
        rescale=(not args.no_rescale),
        seed=args.seed,
    )

    # 4. 解析 JSON Spec
    with open(args.spec, "r", encoding="utf-8") as f:
        concept_to_prompts = json.load(f, object_pairs_hook=OrderedDict)

    outputs_root = REPO_ROOT / 'outputs'
    base_out_dir = outputs_root / f"{args.method}_t2i_{args.category}"

    # 5. 循环生成
    for concept, prompts in concept_to_prompts.items():
        _log(f"\n>>> Concept: {concept}")
        
        for prompt_text in prompts:
            pslug = slugify(prompt_text)
            run_dir = base_out_dir / "imgs" / pslug

            if run_dir.exists() and len(os.listdir(run_dir)) >= args.G:
                _log(f"  [SKIP] '{prompt_text[:40]}...' already exists.")
                continue
            
            run_dir.mkdir(parents=True, exist_ok=True)
            _log(f"  [RUN] Prompt: '{prompt_text}'")

            generator = torch.Generator(device=dev_tr).manual_seed(args.seed)

            # --- 关键：先生成 Latents ---
            # 这样所有的计算都在 Transformer 所在的 dev_tr 上完成
            latents = pipe(
                prompt=prompt_text,
                height=512, width=512,
                num_images_per_prompt=args.G,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                callback_on_step_end=cads_annealer,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="latent", # 不在 pipe 内部解码
            ).images

            # --- 关键：手动搬运并解码 ---
            # 将生成的 latents 从 cuda:1 搬运到 VAE 所在的 cuda:0
            latents = latents.to(device=dev_vae, dtype=pipe.vae.dtype)
            
            with torch.no_grad():
                # 按照 SD3.5 标准缩放因子进行解码
                decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                # 使用 pipeline 内部的图像处理器转换为 PIL 格式
                images = pipe.image_processor.postprocess(decoded, output_type="pil")

            # 6. 保存图片
            for i, img in enumerate(images):
                img.save(run_dir / f"{i:03d}.png")

            torch.cuda.empty_cache()

    _log(f"CADS-t2i color tasks completed. Outputs in {base_out_dir}")

if __name__ == "__main__":
    main()