# =============================
# File: flow_base/scripts/baseline_dpp.py
# =============================

# CUDA_VISIBLE_DEVICES=5,6,7 python scripts/baseline_dpp.py   --prompt "a photo of boxer"   --K 16 --steps 10 --guidance 3.0 --fp16   --device_transformer cuda:1 --device_vae cuda:2 --device_clip cuda:0   --model-dir models/stable-diffusion-3.5-medium   --openai_clip_jit_path ~/.cache/clip/ViT-B-32.pt   --decode_size 224 --chunk_size 2

import os, sys, argparse, random, re
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

def _ensure_out(prompt: str) -> Path:
    out_dir = REPO_ROOT / "outputs" / f"dpp_{_slugify(prompt)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)

    # 多卡放置
    p.add_argument("--device_transformer", type=str, default="cuda:0")
    p.add_argument("--device_vae", type=str, default="cuda:1")
    p.add_argument("--device_clip", type=str, default="cuda:2")

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

    return p.parse_args()

def build_pipe_cpu(model_dir: str):
    from diffusers import StableDiffusion3Pipeline
    sd_dir = _resolve_model_dir(model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(sd_dir, torch_dtype=torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=False)
    return pipe.to("cpu")  # 先在 CPU，随后手动 .to() 各子模块

def main():
    args = parse_args(); set_seed(args.seed)
    out_dir = _ensure_out(args.prompt); print(f"[DPP] Output dir: {out_dir}")

    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dev_clip= torch.device(args.device_clip)
    dtype   = torch.float16 if args.fp16 else torch.float32

    # 1) CPU 加载，再手动上卡
    pipe = build_pipe_cpu(args.model_dir)
    # 模块搬家
    if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
    if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

    # 低显存 VAE
    if args.enable_vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    # xFormers（可用则开）
    if args.enable_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception as e: print(f"[DPP] enable_xformers failed: {e}")

    # 2) 本地 JIT CLIP （放到 dev_clip）
    vcfg = VisionCfg(backend="openai_clip_jit", jit_path=args.openai_clip_jit_path, device=str(dev_clip))
    feat = build_vision_feature(vcfg)

    # 3) 多卡 DPP 耦合器
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

    # 4) 回调：每步对 latents 做 DPP 步
    def dpp_callback(ppl, i, t, kw: Dict[str, Any]):
        num_steps = getattr(ppl.scheduler, "num_inference_steps", None) or args.steps
        t_norm = (i + 1) / float(num_steps)  # 0~1
        lat = kw.get("latents")
        if lat is None:
            return kw
        # DPP 更新（仍在 transformer 卡）
        lat_new = coupler(step_index=i, t_norm=float(t_norm), latents=lat)

        # ✅ 最后一小步：把 latents 搬到 VAE 的设备&dtype，交给管线做最终 decode
        if (i + 1) == num_steps:
            vae = ppl.vae
            vae_dtype = next(vae.parameters()).dtype
            lat_new = lat_new.to(device=vae.device, dtype=vae_dtype, non_blocking=True)

        kw["latents"] = lat_new
        return kw

    prompts = [args.prompt] * args.K
    print(f"[DPP] Sampling K={args.K}, steps={args.steps}, guidance={args.guidance} ...")

    result = pipe(
        prompt=prompts,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        callback_on_step_end=dpp_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    images = result.images if hasattr(result, "images") else result
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            img.save(out_dir / f"{i:02d}.png")
        else:
            Image.fromarray(img).save(out_dir / f"{i:02d}.png")
        # (out_dir / f"{i:02d}.png").write_bytes(b"")  # touch
        # if isinstance(img, Image.Image): img.save(out_dir / f"{i:02d}.png")
        # else: Image.fromarray(img).save(out_dir / f"{i:02d}.png")
    print(f"[DPP] Saved {len(images)} images to: {out_dir}")
    
    # # 兼容返回类型（Diffusers: 有的把 latent 放在 .images，有的叫 .latents）
    # latents_out = getattr(result, "images", None)
    # if not isinstance(latents_out, torch.Tensor):
    #     latents_out = getattr(result, "latents", None)
    # if not isinstance(latents_out, torch.Tensor):
    #     raise RuntimeError("[DPP] pipeline 未返回 latent 张量，请检查 Diffusers 版本。")

    # ✅ 把 latent 挪到 VAE 设备 & VAE 的 dtype 再解码
    # vae = pipe.vae
    # vae_dtype = next(vae.parameters()).dtype
    # z = latents_out.to(device=vae.device, dtype=vae_dtype, non_blocking=True)
    # # 注意：SD3.5 在其 pipeline 内部就是直接 vae.decode(z)
    # imgs = vae.decode(z, return_dict=False)[0]     # [-1, 1]
    # imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0         # [0,1]
    # imgs = imgs.float().cpu()                      # 存图用 CPU

    # # 保存
    # for i in range(imgs.size(0)):
    #     pil = Image.fromarray((imgs[i].permute(1, 2, 0).numpy() * 255).round().clip(0,255).astype("uint8"))
    #     pil.save(out_dir / f"{i:02d}.png")
    # print(f"[DPP] Saved {imgs.size(0)} images to: {out_dir}")

if __name__ == "__main__":
    main()
