#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline: CADS on SD-3.5 (Flow-Matching) — NFE batch with robust multi-device support
- 支持多 steps × 多 seeds，固定 prompt / guidance；方法名固定为 cads。
- 目录结构：outputs/cads_NFE/{eval,imgs}/imgs/seed{seed}_steps{steps}/img_***.png
- 关键修复：避免 VAE/latent 跨卡报错，改为 pipeline 输出 latent，脚本内手动在 VAE 卡 decode。

用法示例：
python scripts/baseline_cads_NFE.py \
  --model models/stable-diffusion-3.5-medium \
  --prompt "a cozy cabin in the snowy forest" \
  --negative "low quality, blurry" \
  --G 8 --guidance 7.5 --height 1024 --width 1024 --dtype fp16 \
  --steps-list 20,40 --seeds-list 0,42,1234 \
  --device-transformer cuda:1 --device-vae cuda:0 --device-clip cuda:0 \
  --device-text1 cuda:1 --device-text2 cuda:1 --device-text3 cuda:1 \
  --enable-vae-tiling --enable-xformers
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline

# repo root
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# CADS import（兼容大小写）
try:
    from cads import CADS
except Exception:
    try:
        from CADS import CADS
    except Exception:
        from cads.cads import CADSConditionAnnealer as CADS


def _log(*a):
    print(*a, flush=True)


def select_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


# ---------------- Robust loader ----------------

def _resolve_model_dir(root: Path) -> Path:
    if (root / "model_index.json").exists():
        return root
    if root.exists() and root.is_dir():
        snaps = root / "snapshots"
        if snaps.is_dir():
            for d1 in snaps.iterdir():
                if (d1 / "model_index.json").exists():
                    return d1
                if d1.is_dir():
                    for d2 in d1.iterdir():
                        if (d2 / "model_index.json").exists():
                            return d2
        for d1 in root.iterdir():
            if (d1 / "model_index.json").exists():
                return d1
            if d1.is_dir():
                for d2 in d1.iterdir():
                    if (d2 / "model_index.json").exists():
                        return d2
    raise FileNotFoundError(f"Could not find model_index.json under '{root}'.")


def load_sd35(model_path: str, torch_dtype):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path not found: {p}")
    if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt"}:
        _log(f"Loading single-file checkpoint: {p}")
        pipe = DiffusionPipeline.from_single_file(p.as_posix(), torch_dtype=torch_dtype)
        if not isinstance(pipe, StableDiffusion3Pipeline):
            pipe = StableDiffusion3Pipeline(**pipe.components)
        return pipe
    idx_dir = _resolve_model_dir(p)
    _log(f"Using diffusers folder: {idx_dir}")
    return StableDiffusion3Pipeline.from_pretrained(idx_dir.as_posix(), torch_dtype=torch_dtype, local_files_only=True)


# ---------------- IO helpers ----------------

def ensure_dirs(outputs_root: Path, seed: int, steps: int):
    root = outputs_root / "cads_NFE"
    eval_dir = root / "eval"
    imgs_dir = root / "imgs"
    sub = imgs_dir / f"seed{seed}_steps{steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    return root, eval_dir, sub


def append_log(eval_dir: Path, record: Dict[str, Any]):
    csv_path = eval_dir / "summary.csv"
    jsonl_path = eval_dir / "summary.jsonl"
    fields = ["timestamp","method","steps","seed","guidance","prompt","G","height","width","out_dir","images"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v) for k,v in record.items()}
        w.writerow(row)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False)+"\n")


# ---------------- parser ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CADS batch on SD-3.5 (Flow Matching) — NFE")
    # I/O
    p.add_argument("--model", type=str, default="models/stable-diffusion-3.5-medium")
    p.add_argument("--outputs", type=str, default=None, help="输出根目录（默认 <repo_root>/outputs）")

    # prompts
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative", type=str, default="low quality, blurry")

    # sampling fixed
    p.add_argument("--G", type=int, default=32)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16","fp32","bf16"])

    # batch
    p.add_argument("--steps-list", type=str, required=True, help="逗号分隔，例如 '20,40'")
    p.add_argument("--seeds-list", type=str, required=True, help="逗号分隔，例如 '0,42,1234'")

    # CADS hyperparams
    g = p.add_argument_group("CADS")
    g.add_argument("--tau1", type=float, default=0.6)
    g.add_argument("--tau2", type=float, default=0.9)
    g.add_argument("--cads-s", type=float, default=0.10, dest="cads_s")
    g.add_argument("--psi", type=float, default=1.0)
    g.add_argument("--no-rescale", action="store_true")
    g.add_argument("--dynamic-cfg", action="store_true")

    # devices
    d = p.add_argument_group("Devices")
    d.add_argument("--device-transformer", type=str, default="cuda:1")
    d.add_argument("--device-vae", type=str, default="cuda:0")
    d.add_argument("--device-clip", type=str, default="cuda:0", help="Alias for text encoders")
    d.add_argument("--device-text1", type=str, default="cuda:1")
    d.add_argument("--device-text2", type=str, default="cuda:1")
    d.add_argument("--device-text3", type=str, default="cuda:1")

    # perf
    p.add_argument("--enable-vae-tiling", action="store_true")
    p.add_argument("--enable-xformers", action="store_true")

    # debug
    p.add_argument("--debug", action="store_true")
    return p


# ---------------- main ----------------

def main():
    args = build_parser().parse_args()
    torch_dtype = select_dtype(args.dtype)

    outputs_root = Path(args.outputs).resolve() if args.outputs else (_REPO_ROOT / "outputs")
    outputs_root.mkdir(parents=True, exist_ok=True)

    steps_list: List[int] = [int(s.strip()) for s in args.steps_list.split(',') if s.strip()]
    seeds_list: List[int] = [int(s.strip()) for s in args.seeds_list.split(',') if s.strip()]

    # 1) 加载 pipeline（CPU），随后模块上到各自卡
    pipe = load_sd35(args.model, torch_dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=False)
    pipe = pipe.to("cpu")

    dev_tr  = torch.device(args.device_transformer) if args.device_transformer else None
    dev_vae = torch.device(args.device_vae) if args.device_vae else None
    dev_t1  = torch.device(args.device_text1) if args.device_text1 else None
    dev_t2  = torch.device(args.device_text2) if args.device_text2 else None
    dev_t3  = torch.device(args.device_text3) if args.device_text3 else None

    # 模块搬家
    if hasattr(pipe, "transformer") and dev_tr is not None:
        pipe.transformer.to(dev_tr, dtype=torch_dtype)
    if hasattr(pipe, "text_encoder") and dev_t1 is not None:
        pipe.text_encoder.to(dev_t1, dtype=torch_dtype)
    if hasattr(pipe, "text_encoder_2") and dev_t2 is not None:
        pipe.text_encoder_2.to(dev_t2, dtype=torch_dtype)
    if hasattr(pipe, "text_encoder_3") and dev_t3 is not None:
        pipe.text_encoder_3.to(dev_t3, dtype=torch_dtype)
    if hasattr(pipe, "vae") and dev_vae is not None:
        pipe.vae.to(dev_vae, dtype=torch_dtype)

    # 低显存 VAE
    if args.enable_vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    # xFormers（可用则开）
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            _log(f"[CADS] enable_xformers failed: {e}")

    # 打印设备
    if args.debug:
        def dev_of(m):
            try:
                return next(m.parameters()).device
            except Exception:
                return None
        _log("[devices] transformer=", dev_of(getattr(pipe, "transformer", None)))
        _log("[devices] vae=", dev_of(getattr(pipe, "vae", None)))
        _log("[devices] text1=", dev_of(getattr(pipe, "text_encoder", None)))
        _log("[devices] text2=", dev_of(getattr(pipe, "text_encoder_2", None)))
        _log("[devices] text3=", dev_of(getattr(pipe, "text_encoder_3", None)))

    method_name = "cads"

    # 主循环：steps × seeds
    for steps in steps_list:
        # 为该 steps 构造 CADS（其内部需要 num_inference_steps）
        def make_cads(seed: int):
            return CADS(
                num_inference_steps=int(steps),
                tau1=args.tau1, tau2=args.tau2,
                s=args.cads_s, psi=args.psi,
                rescale=(not args.no_rescale), dynamic_cfg=args.dynamic_cfg,
                seed=int(seed),
            )

        for seed in seeds_list:
            # 输出目录
            root, eval_dir, sub = ensure_dirs(outputs_root, seed=int(seed), steps=int(steps))
            _log(f"[CADS] Output dir: {sub}")

            # 生成器的 device（由 pipeline 内部执行设备决定）
            try:
                exec_dev = str(pipe._execution_device)
            except Exception:
                # 回退：使用 transformer 所在设备
                exec_dev = args.device_transformer or ("cuda" if torch.cuda.is_available() else "cpu")
            generator = torch.Generator(device=exec_dev).manual_seed(int(seed))

            # 回调：直接用 CADS（不在回调内搬卡）
            cads_cb = make_cads(int(seed))

            # 关键：让 pipeline 输出 latent，由脚本手动 decode，避免跨卡解码报错
            result = pipe(
                prompt=args.prompt,
                negative_prompt=(args.negative if args.negative else None),
                height=args.height, width=args.width,
                num_images_per_prompt=args.G,
                num_inference_steps=int(steps),
                guidance_scale=float(args.guidance),
                generator=generator,
                callback_on_step_end=cads_cb,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="latent",
                return_dict=False,
            )

            # 兼容 tuple / dict：取 latent 张量
            latents_out = result[0] if isinstance(result, (list, tuple)) else getattr(result, "images", None)
            if not isinstance(latents_out, torch.Tensor):
                latents_out = getattr(result, "latents", None)
            if not isinstance(latents_out, torch.Tensor):
                raise RuntimeError("[CADS] pipeline 未返回 latent 张量，请检查 Diffusers 版本。")

            # 手动在 VAE 卡 decode
            vae = pipe.vae
            vae_dtype = next(vae.parameters()).dtype
            sf = getattr(vae.config, "scaling_factor", 1.0)
            z = latents_out.to(device=vae.device, dtype=vae_dtype, non_blocking=True)
            with torch.inference_mode():
                imgs = vae.decode(z / sf, return_dict=False)[0]  # [-1,1]
                imgs = (imgs.float().clamp(-1,1) + 1.0) / 2.0     # [0,1]
                imgs = imgs.cpu()

            # 保存
            img_paths = []
            for i in range(imgs.size(0)):
                pil = Image.fromarray((imgs[i].permute(1,2,0).numpy() * 255.0).round().clip(0,255).astype("uint8"))
                p = sub / f"img_{i:03d}.png"
                pil.save(p)
                img_paths.append(str(p))
            _log(f"[CADS] Saved {len(img_paths)} images to: {sub}")

            append_log(
                eval_dir,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method_name,
                    "steps": int(steps),
                    "seed": int(seed),
                    "guidance": float(args.guidance),
                    "prompt": args.prompt,
                    "G": int(args.G),
                    "height": int(args.height),
                    "width": int(args.width),
                    "out_dir": str(sub),
                    "images": img_paths,
                },
            )

    _log("[CADS] All runs finished.")


if __name__ == "__main__":
    main()