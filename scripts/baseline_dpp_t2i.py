# =============================
# File: flow_base/scripts/baseline_dpp.py
# 适配 T2I-CompBench 的批量生成版本
# =============================

import os, sys, argparse, random, re, json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from typing import Dict, Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPP.vision_feat import VisionCfg, build_vision_feature
from DPP.dpp_coupler_mgpu import DPPCouplerMGPU, MGPUConfig

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _resolve_model_dir(p: str) -> Path:
    base = Path(p)
    if base.is_file():
        raise RuntimeError(f"[DPP] 期望的是 Diffusers 目录而不是单文件: {base}")
    if (base / "model_index.json").exists():
        return base
    for cand in base.rglob("model_index.json"):
        return cand.parent
    raise FileNotFoundError(
        f"[DPP] 未找到 diffusers 的 model_index.json，给定路径: {base}. "
        "请指向包含 model_index.json 的目录（ModelScope 包请指到含 diffusers 权重的子目录）。"
    )

def _slugify(text: str, maxlen: int = 80) -> str:
    s = text.strip().lower(); s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s); s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def parse_args():
    p = argparse.ArgumentParser()
    # ---- 适配 T2I-CompBench 的输入参数 ----
    p.add_argument("--spec", type=str, default=None, help="Path to JSON spec")
    p.add_argument("--prompt", type=str, default=None, help="Fallback single prompt")
    p.add_argument("--method", type=str, default="baseline_dpp")
    p.add_argument("--G", type=int, default=16, help="Number of images per prompt (Replaces --K)")
    # -------------------------------------
    
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)

    # 多卡放置
    p.add_argument("--device_transformer", type=str, default="cuda:1")
    p.add_argument("--device_vae", type=str, default="cuda:0")
    p.add_argument("--device_clip", type=str, default="cuda:0")

    # 本地 SD3.5
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "stable-diffusion-3.5-medium")),
        help="本地 SD3.5 Diffusers 目录",
    )

    # 精度 & 低显存
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--enable_vae_tiling", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")

    # JIT CLIP（本地）
    p.add_argument("--openai_clip_jit_path", type=str, required=True, help="本地 OpenAI CLIP JIT .pt (如 ~/.cache/clip/ViT-B-32.pt)")

    # DPP 超参 & 显存控制
    p.add_argument("--gamma_max", type=float, default=0.12)
    p.add_argument("--kernel_spread", type=float, default=3.0)
    p.add_argument("--gamma_sched", type=str, default="sqrt", choices=["sqrt", "sin2", "poly"])
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--decode_size", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=2)

    return p.parse_args()

def build_pipe_cpu(model_dir: str):
    from diffusers import StableDiffusion3Pipeline
    sd_dir = _resolve_model_dir(model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_dir, torch_dtype=torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=False)
    return pipe.to("cpu")  # 先在 CPU，随后手动 .to() 各子模块

def main():
    args = parse_args(); set_seed(args.seed)
    
    # 1) 解析 JSON 或单 Prompt
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as f:
            concept_to_prompts = json.load(f, object_pairs_hook=OrderedDict)
    elif args.prompt:
        concept_to_prompts = {"single": [args.prompt]}
    else:
        raise ValueError("Must provide either --spec or --prompt")

    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dev_clip= torch.device(args.device_clip)
    dtype   = torch.float16 if args.fp16 else torch.float32

    # 2) CPU 加载，再手动上卡
    print("[DPP] Loading Pipeline to CPU...")
    pipe = build_pipe_cpu(args.model_dir)
    if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
    if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

    if args.enable_vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    if args.enable_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception as e: print(f"[DPP] enable_xformers failed: {e}")

    # 3) 本地 JIT CLIP （放到 dev_clip）
    print(f"[DPP] Loading CLIP to {dev_clip}...")
    vcfg = VisionCfg(backend="openai_clip_jit", jit_path=args.openai_clip_jit_path, device=str(dev_clip))
    feat = build_vision_feature(vcfg)

    mcfg = MGPUConfig(
        decode_size=args.decode_size,
        kernel_spread=args.kernel_spread,
        gamma_max=args.gamma_max,
        gamma_sched=args.gamma_sched,
        clip_grad_norm=args.clip_grad_norm,
        chunk_size=args.chunk_size,
        use_quality_term=False,
    )

    outputs_root = REPO_ROOT / "outputs"

    # 4) 循环处理 Concept 和 Prompts
    for concept, prompts in concept_to_prompts.items():
        base_out_dir = outputs_root / f"{args.method}_{concept}"
        imgs_root_dir = base_out_dir / "imgs"
        eval_dir = base_out_dir / "eval"
        
        imgs_root_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [DPP] Starting Concept: {concept} ===")

        for prompt_text in prompts:
            prompt_slug = _slugify(prompt_text)
            run_dir = imgs_root_dir / prompt_slug
            
            # 断点续跑支持
            if run_dir.exists() and len(list(run_dir.glob("*.png"))) >= args.G:
                print(f"  [SKIP] '{prompt_text[:40]}...' already generated.")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [RUN] Prompt: '{prompt_text}' | G: {args.G} | Steps: {args.steps}")

            # [重要] 每次换 prompt 重新实例化 Coupler，防止 DPP 内部缓存串扰
            coupler = DPPCouplerMGPU(pipe, feat, dev_tr=dev_tr, dev_vae=dev_vae, dev_clip=dev_clip, cfg=mcfg)

            def dpp_callback(ppl, i, t, kw: Dict[str, Any]):
                num_steps = getattr(ppl.scheduler, "num_inference_steps", None) or args.steps
                t_norm = (i + 1) / float(num_steps)  # 0~1
                lat = kw.get("latents")
                if lat is None: return kw
                
                # DPP 更新
                lat_new = coupler(step_index=i, t_norm=float(t_norm), latents=lat)

                # 最后一小步：搬给 VAE
                if (i + 1) == num_steps:
                    vae = ppl.vae
                    vae_dtype = next(vae.parameters()).dtype
                    lat_new = lat_new.to(device=vae.device, dtype=vae_dtype, non_blocking=True)

                kw["latents"] = lat_new
                return kw

            prompts_list = [prompt_text] * args.G

            # 跑管线
            result = pipe(
                prompt=prompts_list,
                height=512, width=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                callback_on_step_end=dpp_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="pil",
            )

            # 保存图片
            images = result.images if hasattr(result, "images") else result
            for i, img in enumerate(images):
                if isinstance(img, Image.Image):
                    img.save(run_dir / f"{i:03d}.png")
                else:
                    Image.fromarray(img).save(run_dir / f"{i:03d}.png")
            
            # 清理显存
            del result, images, coupler
            torch.cuda.empty_cache()

    print("\n[DPP] All Tasks Completed Successfully.")

if __name__ == "__main__":
    main()