# -*- coding: utf-8 -*-
# =============================
# File: flow_base/scripts/dpp_eval_grid.py
# 扩展版：支持 DIM/CIM 的多 prompt × 多 guidance × 多 seed 网格评测
# 保存到 outputs/{method}_{concept}/imgs/{prompt_slug}_seed{seed}_g{guidance}_s{steps}/
# =============================

import os, sys, argparse, random, re, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPP.vision_feat import VisionCfg, build_vision_feature
from DPP.dpp_coupler_mgpu import DPPCouplerMGPU, MGPUConfig


# ------------------------
# Utils
# ------------------------
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

_slug_pat_space = re.compile(r"\s+")
def _slugify(text: str, maxlen: int = 120) -> str:
    s = text.strip().lower()
    s = _slug_pat_space.sub("_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _build_root_out(method: str, concept: str) -> Tuple[Path, Path, Path]:
    base = REPO_ROOT / "outputs" / f"{method}_{_slugify(concept)}"
    imgs = base / "imgs"
    eval_dir = base / "eval"
    imgs.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return base, imgs, eval_dir

def _flatten_prompts_from_spec(spec: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    """
    按照新格式解析（仅按你给的结构）：
      {
        "dog": ["a photo of a dog", "a dog", ...],
        "truck": ["a photo of a truck", "a truck", ...],
        ...
      }
    返回：OrderedDict{ concept -> [prompts...] }（按文件中出现顺序）
    """
    if not isinstance(spec, dict):
        raise ValueError("[DPP] 需要顶层为对象：{concept: [prompts...]}")

    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec.items():
        if not isinstance(concept, str):
            continue
        if not isinstance(plist, (list, tuple)):
            continue
        # 清洗并去重（保序）
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s); cleaned.append(s)
        if cleaned:
            concept_to_prompts[concept] = cleaned

    if not concept_to_prompts:
        raise ValueError("[DPP] 没有解析到有效的 prompts；请确保格式为 {concept: [\"a dog\", ...]}")

    return concept_to_prompts

def _generators_for_K(device: torch.device, base_seed: int, K: int) -> List[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(int(base_seed) + i) for i in range(K)]


# ------------------------
# Argparse
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DPP grid sampler for DIM/CIM evaluation")
    # 输入：二选一
    p.add_argument("--spec", type=str, default=None,
                   help="JSON 文件路径；格式为 {concept: [prompts...] }。提供则覆盖 --prompt。")
    p.add_argument("--prompt", type=str, default=None,
                   help="单条 prompt（若未提供 --spec 时可用）")

    # 网格
    p.add_argument("--K", type=int, default=4, help="每个 prompt 生成 K 张（组内联动做 DPP）")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidances", type=float, nargs="+", default=None,
                   help="multi guidance")
    p.add_argument("--guidance", type=float, default=7.0, help="single choice")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111])

    # 统一分辨率
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)

    # 模型与多卡
    p.add_argument("--device_transformer", type=str, default="cuda:1")
    p.add_argument("--device_vae", type=str, default="cuda:0")
    p.add_argument("--device_clip", type=str, default="cuda:0")
    p.add_argument("--model-dir", type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "stable-diffusion-3.5-medium")),
                   help="local SD3.5 Diffusers model")

    # 精度 & 低显存
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--enable_vae_tiling", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")

    # CLIP JIT
    p.add_argument("--openai_clip_jit_path", type=str, required=True,
                   help="local OpenAI CLIP JIT .pt (如 ~/.cache/clip/ViT-B-32.pt)")

    # DPP 超参
    p.add_argument("--gamma_max", type=float, default=0.12)
    p.add_argument("--kernel_spread", type=float, default=3.0)
    p.add_argument("--gamma_sched", type=str, default="sqrt", choices=["sqrt", "sin2", "poly"])
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--decode_size", type=int, default=256)
    p.add_argument("--chunk_size", type=int, default=2)

    # 方法名（用于落盘目录）
    p.add_argument("--method", type=str, default="dpp")

    return p.parse_args()


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

    # JIT CLIP
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
# Sampling (one prompt × one guidance × one seed)
# ------------------------
def run_one(pipe, coupler, device_tr: torch.device, prompt: str, K: int, steps: int,
            guidance: float, seed: int, target_wh: Tuple[int,int], out_dir: Path):
    """对单个 prompt / guidance / seed 进行一次 K 组联动采样，并保存到 out_dir"""
    W, H = int(target_wh[0]), int(target_wh[1])

    # DPP 更新回调：最后一步把 latent 搬到 VAE 卡/精度
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

    # K 个独立 generator（同一 base seed 的偏移）
    gens = _generators_for_K(device_tr, seed, K)

    prompts = [prompt] * K
    kwargs = dict(
        prompt=prompts,
        num_inference_steps=steps,
        guidance_scale=guidance,
        callback_on_step_end=dpp_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        generator=gens,
        output_type="pil",
    )

    # (尽量) 直接让 pipeline 生成指定尺寸；某些分支若不支持 height/width，会在 except 中回退到保存前 resize
    try:
        result = pipe(height=H, width=W, **kwargs)
    except TypeError:
        result = pipe(**kwargs)

    imgs = result.images if hasattr(result, "images") else result
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        # 保底 resize 到统一尺寸
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()

    # 解析成 concept -> prompts 的映射（仅按新格式；若不用 --spec，可用 --prompt）
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp)
        concept_to_prompts = _flatten_prompts_from_spec(spec)  # OrderedDict
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("请提供 --spec（JSON 文件路径）或 --prompt（单条）之一。")

    # guidance 列表
    guidances = args.guidances if args.guidances is not None else [args.guidance]
    guidances = [float(g) for g in guidances]

    # 构建一次 pipeline / DPP，循环复用
    pipe, coupler, dev_tr = build_all(args)
    dtype = torch.float16 if args.fp16 else torch.float32
    print(f"[DPP] ready. dtype={dtype}, K={args.K}, steps={args.steps}, W×H={args.width}×{args.height}")

    # 按 concept 分开在 outputs/{method}_{concept} 下落盘
    for concept, prompts in concept_to_prompts.items():
        _, imgs_root, eval_dir = _build_root_out(args.method, concept)
        print(f"[DPP] outputs base: {imgs_root.parent}")
        print(f"[DPP] eval dir:     {eval_dir}")

        for ptext in prompts:
            p_slug = _slugify(ptext)  # prompt 的文件夹名基底
            for g in guidances:
                for s in args.seeds:
                    subdir = imgs_root / f"{p_slug}_seed{s}_g{g}_s{args.steps}"
                    print(f"[DPP] sampling: concept='{concept}' | prompt='{ptext}' | seed={s} | guidance={g} | steps={args.steps} -> {subdir}")
                    run_one(pipe, coupler, dev_tr,
                            prompt=ptext, K=args.K, steps=args.steps,
                            guidance=float(g), seed=int(s),
                            target_wh=(args.width, args.height),
                            out_dir=subdir)

    print("[DPP] Done.")


if __name__ == "__main__":
    main()