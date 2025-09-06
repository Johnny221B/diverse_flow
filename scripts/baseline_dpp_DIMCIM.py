#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPP baseline × DIMCIM generator (SD3.5, Flow-Matching)

- 输入：--dimcim-json 指向 DIMCIM 的 dense_prompts JSON（如 bus_dense_prompts.json）
- 输出：<OUTDIR>/{method}_{concept}_CIMDIM/{coarse_imgs,dense_imgs,eval}
- 规则：每个 prompt 生成 20 张（组内 DPP 联动），目录名为“去标点 + 空格→下划线”（保留大小写）

固定默认（可改参数，但已按你需求设好）：
  K=20, guidance=7.5, steps=50, seed=1234, H=W=512, fp16

用法：
  python -u scripts/dpp_dimcim_generate.py \
    --model-dir "./models/stable-diffusion-3.5-medium" \
    --openai_clip_jit_path ~/.cache/clip/ViT-B-32.pt \
    --dimcim-json /mnt/data/bus_dense_prompts.json \
    --outdir /path/to/outputs \
    --method dpp --concept bus \
    --device_transformer cuda:0 --device_vae cuda:1 --device_clip cuda:0 \
    --fp16
"""

import os, sys, argparse, random, re, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# DPP deps（与你原仓库一致）
from DPP.vision_feat import VisionCfg, build_vision_feature
from DPP.dpp_coupler_mgpu import DPPCouplerMGPU, MGPUConfig

# ------------------------
# Utils
# ------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prompt_dirname(prompt: str) -> str:
    """去标点；连字符去掉；空格→下划线；保留大小写。"""
    s = prompt.strip()
    s = re.sub(r"[^\w\s-]", "", s)   # 移除标点
    s = s.replace("-", "")           # 合并连字符
    s = re.sub(r"\s+", "_", s)       # 空格->_
    return s.strip("_")

def ensure_outdirs(outdir: Path, method: str, concept: str) -> Tuple[Path, Path, Path, Path]:
    """
    <outdir>/{method}_{concept}_CIMDIM/{eval, coarse_imgs, dense_imgs}
    """
    base = outdir / f"{method}_{concept}_CIMDIM"
    eval_dir  = base / "eval"
    coarse_d  = base / "coarse_imgs"
    dense_d   = base / "dense_imgs"
    for d in (eval_dir, coarse_d, dense_d):
        d.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, coarse_d, dense_d

def _generators_for_K(device: torch.device | str, base_seed: int, K: int) -> List[torch.Generator]:
    return [torch.Generator(device=str(device)).manual_seed(int(base_seed) + i) for i in range(K)]

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
        "请指向包含 model_index.json 的目录（或 HF snapshot 子目录）。"
    )

def parse_dimcim_dense_json(path: Path) -> Tuple[str, List[str], List[str]]:
    """
    读取 DIMCIM dense_prompts JSON，返回 (concept, coarse_prompts, dense_prompts)。
    预期结构：
    {
      "concept": "bus",
      "coarse_dense_prompts": [
        { "coarse_prompt": "...",
          "dense_prompts": [
            {"attribute_type": "...", "attribute": "...", "dense_prompt": "..."},
            ...
          ]
        }, ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    concept = (data.get("concept") or "unknown").strip() or "unknown"

    coarse, dense = [], []
    rows = data.get("coarse_dense_prompts", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid DIMCIM JSON: 'coarse_dense_prompts' must be a list.")
    for r in rows:
        if not isinstance(r, dict): continue
        cp = r.get("coarse_prompt")
        if isinstance(cp, str) and cp.strip():
            coarse.append(cp.strip())
        dlist = r.get("dense_prompts", [])
        if isinstance(dlist, list):
            for d in dlist:
                if isinstance(d, dict):
                    dp = d.get("dense_prompt")
                    if isinstance(dp, str) and dp.strip():
                        dense.append(dp.strip())

    # 去重保序
    def uniq(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return concept, uniq(coarse), uniq(dense)

# ------------------------
# Build pipeline & DPP coupler
# ------------------------
def build_pipe_cpu(model_dir: str):
    from diffusers import StableDiffusion3Pipeline
    sd_dir = _resolve_model_dir(model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_dir, torch_dtype=torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=False)
    return pipe.to("cpu")

def build_all(args):
    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dev_clip= torch.device(args.device_clip)
    dtype   = torch.float16 if args.fp16 else torch.float32

    pipe = build_pipe_cpu(args.model_dir)
    # 模块上卡
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

    # JIT CLIP（DPP 特征）
    vcfg = VisionCfg(backend="openai_clip_jit", jit_path=args.openai_clip_jit_path, device=str(dev_clip))
    feat = build_vision_feature(vcfg)

    # DPP 耦合器
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
    return pipe, coupler, dev_tr

# ------------------------
# Sampling (one prompt ⇒ K images via DPP)
# ------------------------
def run_one(pipe, coupler, device_tr: torch.device, prompt: str, K: int, steps: int,
            guidance: float, seed: int, target_wh: Tuple[int,int], out_dir: Path, negative: str | None):
    W, H = int(target_wh[0]), int(target_wh[1])

    # DPP 回调：每步更新 K 路 latents；最后一步把 latent 移到 VAE 设备/精度
    def dpp_callback(ppl, i, t, kw: Dict[str, Any]):
        num_steps = getattr(ppl.scheduler, "num_inference_steps", None) or steps
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

    # K 个独立 generator（base_seed+i）
    gens = _generators_for_K(device_tr, seed, K)

    prompts = [prompt] * K
    kwargs = dict(
        prompt=prompts,
        negative_prompt=([negative]*K if negative else None),
        num_inference_steps=steps,
        guidance_scale=guidance,
        callback_on_step_end=dpp_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        generator=gens,
        output_type="pil",
    )

    # 直接尝试固定尺寸；若分支不支持 height/width，则回退
    try:
        result = pipe(height=H, width=W, **kwargs)
    except TypeError:
        result = pipe(**kwargs)

    imgs = result.images if hasattr(result, "images") else result
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")

# ------------------------
# Args
# ------------------------
def build_args():
    p = argparse.ArgumentParser(description="DPP × DIMCIM image generator (SD3.5)")
    # I/O
    p.add_argument("--dimcim-json", required=True, help="DIMCIM dense_prompts JSON 路径")
    p.add_argument("--outdir", required=True, help="输出根目录（将创建 {method}_{concept}_CIMDIM）")
    p.add_argument("--method", default="dpp", help="方法名前缀（目录用）")
    p.add_argument("--concept", default=None, help="可覆盖 JSON 中的 concept（用于目录名）")
    p.add_argument("--negative", type=str, default=None)
    # 生成（固定默认按你要求）
    p.add_argument("--K", type=int, default=20, help="每个 prompt 生成的张数（DPP 组大小）")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--fp16", action="store_true")
    p.set_defaults(fp16=True)  # 默认开 fp16
    # 模型与设备
    p.add_argument("--model-dir", type=str, required=True, help="本地 SD3.5 diffusers 目录")
    p.add_argument("--device_transformer", type=str, default="cuda:0")
    p.add_argument("--device_vae", type=str, default="cuda:1")
    p.add_argument("--device_clip", type=str, default="cuda:0")
    # DPP / 特征
    p.add_argument("--openai_clip_jit_path", type=str, required=True, help="OpenAI CLIP JIT .pt 路径（ViT-B/32）")
    p.add_argument("--gamma_max", type=float, default=0.12)
    p.add_argument("--kernel_spread", type=float, default=3.0)
    p.add_argument("--gamma_sched", type=str, default="sqrt", choices=["sqrt", "sin2", "poly"])
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--decode_size", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=2)
    p.add_argument("--enable_vae_tiling", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")
    return p.parse_args()

# ------------------------
# Main
# ------------------------
def main():
    args = build_args()

    # 解析 DIMCIM prompts
    dimcim_path = Path(args.dimcim_json)
    concept_json, coarse_prompts, dense_prompts = parse_dimcim_dense_json(dimcim_path)
    concept = (args.concept or concept_json).strip()
    print(f"[DPP] concept = {concept}  (#coarse={len(coarse_prompts)}, #dense={len(dense_prompts)})")

    # 输出目录
    outroot = Path(args.outdir)
    base_dir, eval_dir, coarse_dir, dense_dir = ensure_outdirs(outroot, args.method, concept)
    print(f"[DPP] base:   {base_dir}\n[DPP] eval:   {eval_dir}\n[DPP] coarse: {coarse_dir}\n[DPP] dense:  {dense_dir}")

    # 构建 pipeline & DPP 耦合器（一次，复用）
    pipe, coupler, dev_tr = build_all(args)
    print(f"[DPP] ready. dtype={'fp16' if args.fp16 else 'fp32'}, K={args.K}, steps={args.steps}, W×H={args.width}×{args.height}, guidance={args.guidance}")

    # 生成函数（单个 prompt → K 张，DPP 联动）
    def gen_prompt(prompt: str, parent_dir: Path):
        sub = parent_dir / prompt_dirname(prompt)
        print(f"[DPP] prompt: {prompt}\n      -> {sub}")
        run_one(pipe, coupler, dev_tr,
                prompt=prompt, K=int(args.K), steps=int(args.steps),
                guidance=float(args.guidance), seed=int(args.seed),
                target_wh=(args.width, args.height),
                out_dir=sub, negative=args.negative)

    # 先跑 COARSE，再跑 DENSE
    for cp in coarse_prompts:
        gen_prompt(cp, coarse_dir)
    for dp in dense_prompts:
        gen_prompt(dp, dense_dir)

    print("[DPP] Done.")

if __name__ == "__main__":
    main()
