#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPP 批量版（多 seeds × 多 steps；固定 guidance & prompt）

输出结构（固定）：
    outputs/dpp_NFE/{eval,imgs}/
    └─ imgs/
       └─ seed{seed}_steps{steps}/
          └─ img_***.png

相较原版 baseline_dpp.py：
- 新增 CLI：--steps-list、--seeds-list，用于批量组合；保留 --K/--guidance 等；
- 模型、特征、耦合器仅初始化一次；
- 每个 (seed, steps) 组合单独采样与落盘；
- 在 eval/ 下维护 summary.csv 与 summary.jsonl（可用于后续 eval 脚本）。
"""

import os, sys, argparse, random, re, json, csv, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPP.vision_feat import VisionCfg, build_vision_feature
from DPP.dpp_coupler_mgpu import DPPCouplerMGPU, MGPUConfig

# ---------------------- utils ----------------------

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

def _ensure_dirs(outputs_root: Path, seed: int, steps: int) -> Tuple[Path, Path, Path]:
    root = outputs_root / "dpp_NFE"
    eval_dir = root / "eval"
    imgs_dir = root / "imgs"
    sub = imgs_dir / f"seed{seed}_steps{steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    return root, eval_dir, sub


def _append_log(eval_dir: Path, record: Dict[str, Any]):
    csv_path = eval_dir / "summary.csv"
    jsonl_path = eval_dir / "summary.jsonl"
    fieldnames = [
        "timestamp", "method", "steps", "seed", "guidance", "prompt",
        "K", "height", "width", "out_dir", "images"
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in record.items()}
        w.writerow(row)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------- args ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    # 固定项
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    # 批量项
    p.add_argument("--steps-list", type=str, required=True, help="逗号分隔，例如 '10,20,30'")
    p.add_argument("--seeds-list", type=str, required=True, help="逗号分隔，例如 '0,42,1234'")

    # 设备
    p.add_argument("--device_transformer", type=str, default="cuda:1")
    p.add_argument("--device_vae", type=str, default="cuda:0")
    p.add_argument("--device_clip", type=str, default="cuda:0")

    # 本地 SD3.5
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "stable-diffusion-3.5-medium")),
        help="本地 SD3.5 Diffusers 目录（ModelScope 解压目录亦可；会递归寻找 model_index.json）",
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

    # 输出根目录
    p.add_argument("--outputs", type=str, default=None, help="默认 <repo_root>/outputs")

    return p.parse_args()


# ---------------------- build & main ----------------------

def build_pipe_cpu(model_dir: str, dtype: torch.dtype):
    from diffusers import StableDiffusion3Pipeline
    sd_dir = _resolve_model_dir(model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_dir, torch_dtype=dtype if dtype==torch.float32 else torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=False)
    return pipe.to("cpu")  # 先在 CPU，随后手动 .to() 各子模块


def main():
    args = parse_args()
    outputs_root = Path(args.outputs).resolve() if args.outputs else (REPO_ROOT / "outputs")
    outputs_root.mkdir(parents=True, exist_ok=True)

    # 解析批量列表
    steps_list: List[int] = [int(s.strip()) for s in args.steps_list.split(',') if s.strip()]
    seeds_list: List[int] = [int(s.strip()) for s in args.seeds_list.split(',') if s.strip()]

    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dev_clip= torch.device(args.device_clip)
    dtype   = torch.float16 if args.fp16 else torch.float32

    # 1) CPU 加载，再手动上卡（一次）
    pipe = build_pipe_cpu(args.model_dir, dtype=torch.float32)
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

    # 2) 本地 JIT CLIP （放到 dev_clip）
    vcfg = VisionCfg(backend="openai_clip_jit", jit_path=args.openai_clip_jit_path, device=str(dev_clip))
    feat = build_vision_feature(vcfg)

    # 3) 多卡 DPP 耦合器（一次）
    mcfg = MGPUConfig(
        decode_size=args.decode_size,
        kernel_spread=args.kernel_spread,
        gamma_max=args.gamma_max,
        gamma_sched=args.gamma_sched,
        clip_grad_norm=args.clip_grad_norm,
        chunk_size=args.chunk_size,
        use_quality_term=False,
    )
    coupler = DPPCouplerMGPU(pipe, feat, dev_tr=dev_tr, dev_vae=dev_vae, dev_clip=dev_clip, cfg=mcfg)

    # 回调（闭包捕获 coupler & steps）
    def make_callback(num_steps: int):
        def dpp_callback(ppl, i, t, kw: Dict[str, Any]):
            t_norm = (i + 1) / float(num_steps)
            lat = kw.get("latents")
            if lat is None:
                return kw
            lat_new = coupler(step_index=i, t_norm=float(t_norm), latents=lat)
            if (i + 1) == num_steps:
                vae = ppl.vae
                vae_dtype = next(vae.parameters()).dtype
                lat_new = lat_new.to(device=vae.device, dtype=vae_dtype, non_blocking=True)
            kw["latents"] = lat_new
            return kw
        return dpp_callback

    method_name = "dpp"

    # 4) 主循环：多 steps × 多 seeds
    for steps in steps_list:
        cb = make_callback(num_steps=int(steps))
        for seed in seeds_list:
            set_seed(seed)
            root, eval_dir, sub = _ensure_dirs(outputs_root, seed=seed, steps=int(steps))
            print(f"[DPP] Output dir: {sub}")

            prompts = [args.prompt] * args.K
            print(f"[DPP] Sampling K={args.K}, steps={steps}, guidance={args.guidance}, seed={seed} ...")

            result = pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=int(steps),
                guidance_scale=args.guidance,
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="pil",
            )

            images = result.images if hasattr(result, "images") else result
            img_paths = []
            for i, img in enumerate(images):
                p = sub / f"img_{i:03d}.png"
                if isinstance(img, Image.Image):
                    img.save(p)
                else:
                    Image.fromarray(img).save(p)
                img_paths.append(str(p))
            print(f"[DPP] Saved {len(images)} images to: {sub}")

            _append_log(
                eval_dir,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method_name,
                    "steps": int(steps),
                    "seed": int(seed),
                    "guidance": float(args.guidance),
                    "prompt": args.prompt,
                    "K": int(args.K),
                    "height": int(args.height),
                    "width": int(args.width),
                    "out_dir": str(sub),
                    "images": img_paths,
                },
            )

    print("[DPP] All runs finished.")


if __name__ == "__main__":
    main()