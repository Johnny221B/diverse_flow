# -*- coding: utf-8 -*-
# =============================
# File: flow_base/scripts/dpp_eval_grid.py
# 扩展版：支持 DIM/CIM 的多 prompt × 多 guidance × 多 seed 网格评测
# 以及 ImageNet-400 单图评测
#
# Grid 模式输出:
#   outputs/{method}_{concept}/imgs/{prompt_slug}_seed{seed}_g{guidance}_s{steps}/
#   outputs/{method}_{concept}/eval/{method}_{concept}_cost.csv   # 新增：cost 统计
#
# ImageNet-400 模式输出:
#   outputs/imagenet_400/{method}/cls_000.png ... cls_399.png
# =============================

import os, sys, argparse, random, re, json, time, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image

# 可选：torch.profiler 用于 FLOPs 统计
try:
    import torch.profiler as torch_profiler
except Exception:
    torch_profiler = None

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
    按照新格式解析：
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
    p = argparse.ArgumentParser(description="DPP grid sampler for DIM/CIM evaluation + ImageNet-400")

    # 输入：三选一（优先级：imagenet-json > spec > prompt）
    p.add_argument("--imagenet-json", type=str, default=None,
                   help="ImageNet-400 prompt JSON: {class_id: prompt}")
    p.add_argument("--spec", type=str, default=None,
                   help="JSON 文件路径；格式为 {concept: [prompts...] }。提供则覆盖 --prompt。")
    p.add_argument("--prompt", type=str, default=None,
                   help="单条 prompt（若未提供 --spec / --imagenet-json 时可用）")

    # 网格 (Grid 模式使用)
    p.add_argument("--K", type=int, default=4, help="每个 prompt 生成 K 张（组内联动做 DPP）")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidances", type=float, nargs="+", default=None,
                   help="multi guidance (grid mode)")
    p.add_argument("--guidance", type=float, default=3.0, help="single choice; 也用于 ImageNet 模式")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 3333, 5555],
                   help="grid 模式的 seeds 列表；ImageNet 模式若未显式指定 --seed，则用 seeds[0] 作为 base_seed")
    p.add_argument("--seed", type=int, default=42,
                   help="ImageNet-400 模式 base_seed（若不指定则用 seeds[0]）")

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

    # cost profiling
    p.add_argument(
        "--profile-flops",
        action="store_true",
        help="使用 torch.profiler 估计单次 run 的 FLOPs（会有一定额外开销）"
    )

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
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[DPP] enable_xformers failed: {e}")

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
# Sampling (one prompt × one guidance × one seed) for Grid 模式
# ------------------------
def run_one(pipe, coupler, device_tr: torch.device, prompt: str, K: int, steps: int,
            guidance: float, seed: int, target_wh: Tuple[int,int], out_dir: Path,
            profile_flops: bool = False,
            mem_devices: List[torch.device] | None = None) -> Tuple[int, float, float, float]:
    """
    对单个 prompt / guidance / seed 进行一次 K 组联动采样，并保存到 out_dir
    返回: (num_images, wall_time_total, flops_total, gpu_mem_peak_mb)
    """
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

    def _call_pipe():
        try:
            return pipe(height=H, width=W, **kwargs)
        except TypeError:
            return pipe(**kwargs)

    flops_total = -1.0
    gpu_mem_peak_mb = -1.0

    # 多卡：重置所有相关 device 的 peak stats
    if mem_devices and torch.cuda.is_available():
        for d in mem_devices:
            try:
                torch.cuda.reset_peak_memory_stats(d)
            except Exception as e:
                print(f"[DPP][WARN] could not reset peak memory stats on {d}: {e}")

    t0 = time.perf_counter()

    if profile_flops and torch_profiler is not None:
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        try:
            with torch_profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                result = _call_pipe()
            # 累积所有事件的 FLOPs
            try:
                flops_total = float(sum(
                    e.flops for e in prof.key_averages()
                    if hasattr(e, "flops") and e.flops is not None
                ))
            except Exception as e:
                print(f"[DPP][WARN] failed to aggregate FLOPs: {e}")
                flops_total = -1.0
        except Exception as e:
            print(f"[DPP][WARN] FLOPs profiling failed, fallback without FLOPs. Error: {e}")
            result = _call_pipe()
            flops_total = -1.0
    else:
        if profile_flops and torch_profiler is None:
            print("[DPP][WARN] torch.profiler not available; FLOPs will be -1.")
        result = _call_pipe()

    imgs = result.images if hasattr(result, "images") else result
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        # 保底 resize 到统一尺寸
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")

    t1 = time.perf_counter()
    wall_time_total = float(t1 - t0)
    num_images = int(len(imgs)) if imgs is not None else 0

    # 读取所有相关 GPU 的峰值显存，取 max（MB）
    if mem_devices and torch.cuda.is_available():
        peaks_mb = []
        for d in mem_devices:
            try:
                peak_bytes = torch.cuda.max_memory_allocated(d)
                peaks_mb.append(float(peak_bytes) / (1024.0 ** 2))
            except Exception as e:
                print(f"[DPP][WARN] failed to get peak memory on {d}: {e}")
        if peaks_mb:
            gpu_mem_peak_mb = max(peaks_mb)
        else:
            gpu_mem_peak_mb = -1.0
    else:
        gpu_mem_peak_mb = -1.0

    return num_images, wall_time_total, flops_total, gpu_mem_peak_mb


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32

    # =============== 模式 B：ImageNet-400 单图评估 ===============
    if args.imagenet_json is not None:
        # 读取 ImageNet-400 的 class_id -> prompt
        imagenet_path = Path(args.imagenet_json)
        if not imagenet_path.exists():
            raise FileNotFoundError(f"[DPP] imagenet-json not found: {imagenet_path}")

        with imagenet_path.open("r", encoding="utf-8") as fp:
            cls_to_prompt: Dict[str, str] = json.load(fp, object_pairs_hook=OrderedDict)

        base_seed = args.seed if args.seed is not None else (args.seeds[0] if args.seeds else 0)
        guidance = float(args.guidance)

        print(f"[DPP][ImageNet-400] base_seed={base_seed}, guidance={guidance}, steps={args.steps}")
        print(f"[DPP][ImageNet-400] #classes = {len(cls_to_prompt)}")

        # 构建一次 pipeline / DPP，整个 ImageNet-400 循环复用
        pipe, coupler, dev_tr = build_all(args)
        print(f"[DPP][ImageNet-400] ready. dtype={dtype}, W×H={args.width}×{args.height}")

        out_root = REPO_ROOT / "outputs" / "imagenet_400" / args.method
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[DPP][ImageNet-400] output dir: {out_root}")

        W, H = int(args.width), int(args.height)

        # 逐类生成：每个 class_id 只生成一张图 cls_xxx.png
        for cls_id_str, prompt in sorted(cls_to_prompt.items(), key=lambda kv: int(kv[0])):
            cls_id = int(cls_id_str)
            cur_seed = int(base_seed) + cls_id

            print(f"[DPP][ImageNet-400] cls={cls_id:03d} seed={cur_seed} prompt='{prompt}'")
            set_seed(cur_seed)

            # 针对单图的 DPP 回调（K=1）
            def dpp_callback(ppl, i, t, kw: Dict[str, Any], steps=args.steps):
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

            # 只生成一张图：K = 1
            gens = _generators_for_K(dev_tr, cur_seed, 1)
            prompts = [prompt]
            kwargs = dict(
                prompt=prompts,
                num_inference_steps=int(args.steps),
                guidance_scale=guidance,
                callback_on_step_end=dpp_callback,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=gens,
                output_type="pil",
            )

            try:
                result = pipe(height=H, width=W, **kwargs)
            except TypeError:
                result = pipe(**kwargs)

            imgs = result.images if hasattr(result, "images") else result
            img = imgs[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.size != (W, H):
                img = img.resize((W, H), resample=Image.BICUBIC)

            save_path = out_root / f"cls_{cls_id:03d}.png"
            img.save(save_path)

            # 释放引用（必要时可加 cuda 清理）
            del imgs, img

        print("[DPP][ImageNet-400] Done.")
        # 如有需要，也可以类似 grid 模式添加 imagenet_400 的 cost 统计 CSV
        return

    # =============== 模式 A：原始 Grid 多 prompt × guidance × seed ===============

    # 解析成 concept -> prompts 的映射（仅按新格式；若不用 --spec，可用 --prompt）
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec = json.load(fp)
        concept_to_prompts = _flatten_prompts_from_spec(spec)  # OrderedDict
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("请提供 --imagenet-json 或 --spec（JSON 文件路径）或 --prompt（单条）之一。")

    # guidance 列表
    guidances = args.guidances if args.guidances is not None else [args.guidance]
    guidances = [float(g) for g in guidances]

    # 构建一次 pipeline / DPP，循环复用
    pipe, coupler, dev_tr = build_all(args)
    print(f"[DPP] ready. dtype={dtype}, K={args.K}, steps={args.steps}, W×H={args.width}×{args.height}")

    dtype_str = "fp16" if args.fp16 else "fp32"

    # 构建一次 mem_devices （多卡显存统计用）
    mem_devices: List[torch.device] = []
    if torch.cuda.is_available():
        dev_strs = set([args.device_transformer, args.device_vae, args.device_clip])
        for ds in dev_strs:
            if ds is None:
                continue
            try:
                d = torch.device(ds)
            except Exception:
                continue
            if d.type != "cuda":
                continue
            if all(d != x for x in mem_devices):
                mem_devices.append(d)

    # 按 concept 分开在 outputs/{method}_{concept} 下落盘
    for concept, prompts in concept_to_prompts.items():
        base_dir, imgs_root, eval_dir = _build_root_out(args.method, concept)
        print(f"[DPP] outputs base: {base_dir}")
        print(f"[DPP] eval dir:     {eval_dir}")

        # 每个 concept 一个 cost 文件: {method}_{concept}_cost.csv
        cost_csv_path = eval_dir / f"{args.method}_{_slugify(concept)}_cost.csv"
        csv_new = not cost_csv_path.exists()
        cost_f = open(cost_csv_path, "a", encoding="utf-8", newline="")
        cost_writer = csv.writer(cost_f)
        if csv_new:
            cost_writer.writerow([
                "method",
                "concept",
                "prompt",
                "seed",
                "guidance",
                "steps",
                "num_images",
                "height",
                "width",
                "dtype",
                "device_transformer",
                "device_vae",
                "device_clip",
                "wall_time_total",
                "wall_time_per_image",
                "flops_total",
                "flops_per_image",
                "gpu_mem_peak_mb",  # 新增：多卡取 max 峰值显存
            ])

        for ptext in prompts:
            p_slug = _slugify(ptext)  # prompt 的文件夹名基底
            for g in guidances:
                for s in args.seeds:
                    subdir = imgs_root / f"{p_slug}_seed{s}_g{g}_s{args.steps}"
                    print(f"[DPP] sampling: concept='{concept}' | prompt='{ptext}' | seed={s} | guidance={g} | steps={args.steps} -> {subdir}")

                    num_images, wall_time_total, flops_total, gpu_mem_peak_mb = run_one(
                        pipe, coupler, dev_tr,
                        prompt=ptext, K=args.K, steps=args.steps,
                        guidance=float(g), seed=int(s),
                        target_wh=(args.width, args.height),
                        out_dir=subdir,
                        profile_flops=args.profile_flops,
                        mem_devices=mem_devices,
                    )

                    wall_time_per_image = wall_time_total / num_images if num_images > 0 else -1.0
                    flops_per_image = flops_total / num_images if (num_images > 0 and flops_total > 0) else -1.0

                    cost_writer.writerow([
                        args.method,
                        concept,
                        ptext,
                        int(s),
                        float(g),
                        int(args.steps),
                        num_images,
                        int(args.width),
                        int(args.height),
                        dtype_str,
                        args.device_transformer or "",
                        args.device_vae or "",
                        args.device_clip or "",
                        f"{wall_time_total:.6f}",
                        f"{wall_time_per_image:.6f}",
                        f"{flops_total:.3f}",
                        f"{flops_per_image:.3f}",
                        f"{gpu_mem_peak_mb:.3f}",
                    ])
                    cost_f.flush()

        cost_f.close()

    print("[DPP] Done.")


if __name__ == "__main__":
    main()
