#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse SDXL-Turbo (JSON-driven prompts, no-embeds, UNet=Transformer device, METHOD-CONSISTENT)
[with A+B noise + safe init-orthogonal injection + Heun-style endpoint volume]

特点：
- 从 JSON 文件中读取多个 concept 和 prompt：
    {
        "bowl": ["a photo of a bowl"],
        "truck": ["a close-up photo of a truck", ...]
    }
- 支持多 concept、多 prompt、多 seed、多 guidance。
- 对每个 (concept, prompt, seed, guidance) 运行一次 pipeline：
    回调：VAE 解码 -> CLIP 体积损失（在 Heun 预测 latent 上）-> VJP 回 latent -> 写回
- 输出：
    outputs/<method>_<concept>/{imgs,eval}/
        imgs/<prompt_slug>_seed<SEED>_g<GUIDANCE>_s<STEPS>/img_000.png ...

例如：
    method = sd35_inception
    concept = bowl
    prompt = "a photo of a bowl" -> prompt_slug = a_photo_of_a_bowl
    seed = 1111, guidance = 5.0, steps = 30

路径：
    outputs/sd35_inception_bowl/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_000.png
"""

import os
import re
import sys
import time
import json
import argparse
import traceback
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- utils --------------------

def _gb(x): return f"{x/1024**3:.2f} GB"


def print_mem_all(tag: str, devices: list):
    lines = [f"[{tag}]"]
    for d in devices:
        if d.type == 'cuda':
            free, total = torch.cuda.mem_get_info(d)
            used = total - free
            lines.append(f"  {d}: used={_gb(used)}, free={_gb(free)}")
        else:
            lines.append(f"  {d}: CPU")
    print("\n".join(lines), flush=True)


def _first_device_of_module(m: nn.Module):
    if not isinstance(m, nn.Module):
        return None
    for p in m.parameters(recurse=False):
        return p.device
    for b in m.buffers(recurse=False):
        return b.device
    for sm in m.children():
        for p in m.parameters(recurse=False):
            return p.device
        for b in m.buffers(recurse=False):
            return b.device
    return None


def inspect_pipe_devices(pipe):
    # 同时兼容 SD3.5 / SDXL-Turbo：
    names = [
        "transformer",   # SD3.5
        "unet",          # SDXL / SDXL-Turbo
        "text_encoder", "text_encoder_2", "text_encoder_3",
        "vae",
        "tokenizer", "tokenizer_2", "tokenizer_3", "scheduler",
    ]
    report = {}
    for name in names:
        if not hasattr(pipe, name):
            continue
        obj = getattr(pipe, name)
        if obj is None:
            report[name] = "None"
            continue
        if isinstance(obj, nn.Module):
            dev = _first_device_of_module(obj)
            report[name] = str(dev) if dev is not None else "module(no params)"
        else:
            report[name] = "non-module"
    print("[pipe-devices]", report, flush=True)


def assert_on(m, want):
    if not isinstance(m, nn.Module):
        return
    for p in m.parameters():
        if str(p.device) != str(want):
            raise RuntimeError(f"Param on {p.device}, expected {want}")
    for b in m.buffers():
        if str(b.device) != str(want):
            raise RuntimeError(f"Buffer on {b.device}, expected {want}")


def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S")
    if debug:
        print(f"[{ts}] {s}", flush=True)


def _slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip())
    s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
    s = re.sub(r'_{2,}', '_', s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s


def _resolve_model_dir(path: str) -> str:
    """
    优先认为是本地路径（含 model_index.json），否则抛错给上层走远程加载。
    """
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')):
        return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files:
            return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Diverse SDXL-Turbo (JSON-driven, UNet=Transformer device, METHOD-CONSISTENT)'
    )

    # JSON spec
    ap.add_argument(
        '--spec',
        type=str,
        required=True,
        help='JSON 文件路径，例如 /data2/toby/OSCAR/specs/prompt.json，'
             '结构如 {"bowl": ["a photo of a bowl", ...]}'
    )
    ap.add_argument(
        '--concepts',
        type=str,
        default=None,
        help='要使用的 concept，逗号分隔，如 "bowl,truck"；不填则使用 JSON 里的全部键'
    )

    # 生成参数（prompt 将从 JSON 读取）
    ap.add_argument('--G', type=int, default=32)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30, help="建议 SDXL-Turbo 用 1~4 步")

    # guidance：单个 + 多个
    ap.add_argument('--guidance', type=float, default=5.0,
                    help="SDXL-Turbo 官方建议 guidance_scale=0.0，可自行实验调整")
    ap.add_argument(
        '--guidances',
        type=str,
        default=None,
        help='多个 guidance, 逗号分隔，如 "0.0,1.0,2.0"，不填则用 --guidance'
    )

    # seed：单个 + 多个
    ap.add_argument('--seed', type=int, default=1111)
    ap.add_argument(
        '--seeds',
        type=str,
        default=None,
        help='多个 seed, 逗号分隔，如 "1111,2222"，不填则用 --seed'
    )

    ap.add_argument('--negative', type=str, default='')

    # 模型路径 / repo id
    ap.add_argument(
        '--model-dir',
        type=str,
        default='stabilityai/sdxl-turbo',
        help='可以是本地路径，或者 HuggingFace repo id（如 stabilityai/sdxl-turbo）'
    )
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # 输出与方法名
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='sdxl_turbo_tuned')

    # 多样性目标（方法相关）
    ap.add_argument('--gamma0', type=float, default=0.06)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.15)

    # 正交/门控
    ap.add_argument('--partial-ortho', type=float, default=0.96)
    ap.add_argument('--t-gate', type=str, default='0.70,0.99')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # 噪声时间形态：early=前期强，late=后期强
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])

    # 噪声控制（A+B）
    ap.add_argument('--eta-sde', type=float, default=0.5)
    ap.add_argument('--rho0', type=float, default=0.25)
    ap.add_argument('--rho1', type=float, default=0.05)
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)

    # 安全的“初噪正交化”注入（默认 blend 混入）
    ap.add_argument('--init-ortho', type=float, default=0.2,
                    help='第0步向latent混入正交噪声的强度（0关闭，建议0.1~0.3）')
    ap.add_argument('--init-ortho-mode', type=str, default='blend',
                    choices=['blend', 'off', 'replace'])

    # 设备
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae', type=str, default='cuda:0')
    ap.add_argument('--device-clip', type=str, default='cuda:0')
    ap.add_argument('--device-text1', type=str, default='cuda:1')
    ap.add_argument('--device-text2', type=str, default='cuda:1')
    ap.add_argument('--device-text3', type=str, default='cuda:1')

    # 省显存 + 调试
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()

    # ===== sys.path 注入（让脚本能 import diverse_flow） =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # TE 默认与 UNet / Transformer 同卡
    if args.device_text1 is None:
        args.device_text1 = args.device_transformer
    if args.device_text2 is None:
        args.device_text2 = args.device_transformer
    if args.device_text3 is None:
        args.device_text3 = args.device_transformer

    # ===== 解析 seeds / guidances / spec / concepts =====
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [args.seed]

    if args.guidances is not None:
        guidances = [float(g) for g in args.guidances.split(",") if g.strip() != ""]
    else:
        guidances = [args.guidance]

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    if args.concepts is not None:
        concept_list = [c.strip() for c in args.concepts.split(",") if c.strip() != ""]
    else:
        concept_list = list(spec.keys())

    # 过滤不存在的 concept
    concept_list = [c for c in concept_list if c in spec]
    if len(concept_list) == 0:
        raise ValueError("No valid concept found in spec for --concepts")

    try:
        import torch.backends.cudnn as cudnn
        from torch.utils.checkpoint import checkpoint
        from diffusers import DiffusionPipeline
        from torchvision.utils import save_image

        from diverse_flow.config import DiversityConfig
        from diverse_flow.clip_wrapper import CLIPWrapper
        from diverse_flow.volume_objective import VolumeObjective
        from diverse_flow.utils import project_partial_orth, batched_norm as _bn
        from diverse_flow.utils import sched_factor as time_sched_factor

        # 设备 + dtype
        dev_tr = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_clip = torch.device(args.device_clip)
        dtype = torch.float16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: transformer/unet={dev_tr}, vae={dev_vae}, clip={dev_clip}", args.debug)
        _log(f"Model dir / repo: {args.model_dir}", args.debug)
        _log(f"CLIP JIT: {args.clip_jit}", args.debug)
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

        # ===== 1) 加载 SDXL-Turbo：支持本地路径 / 远程 repo =====
        local_only = True
        model_id_or_path = args.model_dir
        try:
            model_id_or_path = _resolve_model_dir(args.model_dir)
        except FileNotFoundError:
            local_only = False

        _log("Loading SDXL-Turbo (DiffusionPipeline) ...", args.debug)
        pipe = DiffusionPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")
        print("scheduler:", pipe.scheduler.__class__.__name__)

        # ===== 2) 模块上卡 =====
        _log("Moving modules to target devices ...", args.debug)
        if hasattr(pipe, "unet"):
            pipe.unet.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(dev_tr, dtype=dtype)  # 兼容老 SD3.5
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"):
            pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):
            pipe.vae.to(dev_vae, dtype=dtype)

        if args.enable_vae_tiling and hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

        if args.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                _log(f"enable_xformers failed: {e}", args.debug)

        inspect_pipe_devices(pipe)
        if hasattr(pipe, "unet"):
            assert_on(pipe.unet, dev_tr)
        if hasattr(pipe, "transformer"):
            assert_on(pipe.transformer, dev_tr)
        if hasattr(pipe, "text_encoder"):
            assert_on(pipe.text_encoder, dev_tr)
        if hasattr(pipe, "text_encoder_2"):
            assert_on(pipe.text_encoder_2, dev_tr)
        if hasattr(pipe, "text_encoder_3"):
            assert_on(pipe.text_encoder_3, dev_tr)
        if hasattr(pipe, "vae"):
            assert_on(pipe.vae, dev_vae)

        # ===== 3) CLIP & Volume objective =====
        _log("Loading CLIP ...", args.debug)
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type == 'cuda' else torch.device("cpu"),
        )
        _log("CLIP ready.", args.debug)

        t0, t1 = args.t_gate.split(',')
        cfg = DiversityConfig(
            num_steps=args.steps,
            tau=args.tau,
            eps_logdet=args.eps_logdet,
            gamma0=args.gamma0,
            gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho,
            t_gate=(float(t0), float(t1)),
            sched_shape=args.sched_shape,
            clip_image_size=224,
            leverage_alpha=0.0,  # 先关闭 leverage，避免 cholesky 数值问题
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # ===== 4) 状态 & 工具函数 =====
        state: Dict[str, Optional[torch.Tensor]] = {
            "prev_latents_vae_cpu": None,
            "prev_ctrl_vae_cpu": None,
            "prev_dt_unit": None,
            "prev_prev_latents_vae_cpu": None,
            "last_logdet": None,
            "gamma_auto_done": False,
        }

        def reset_state():
            state["prev_latents_vae_cpu"] = None
            state["prev_ctrl_vae_cpu"] = None
            state["prev_dt_unit"] = None
            state["prev_prev_latents_vae_cpu"] = None
            state["last_logdet"] = None
            state["gamma_auto_done"] = False
            if "ctrl_cache" in state:
                del state["ctrl_cache"]

        def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
            """
            通用 VAE 解码，兼容 SDXL/SDXL-Turbo：
            - 支持 latents_mean / latents_std / force_upcast
            - 输出 [0,1]
            """
            vae = pipe.vae
            orig_dtype = vae.dtype

            needs_upcasting = (
                vae.dtype == torch.float16
                and getattr(vae.config, "force_upcast", False)
            )
            if needs_upcasting:
                vae.to(dtype=torch.float32)

            target_dtype = next(iter(vae.post_quant_conv.parameters())).dtype
            latents = z.to(device=vae.device, dtype=target_dtype)

            has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
            has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
            sf = getattr(vae.config, "scaling_factor", 1.0)

            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / sf + latents_mean
            else:
                latents = latents / sf

            image = vae.decode(latents, return_dict=False)[0]  # [-1,1]

            if needs_upcasting:
                vae.to(dtype=orig_dtype)

            return (image.float().clamp(-1, 1) + 1.0) / 2.0

        def _beta_sched(t_norm: float, eps: float = 1e-2) -> float:
            t = float(t_norm)
            if args.noise_timing == 'early':
                base = min(1.0, t / (1.0 - t + eps))
            else:
                base = min(1.0, (1.0 - t) / (t + eps))
            return float(base)

        def _brownian_std_from_scheduler(ppl, i):
            sch = ppl.scheduler
            try:
                if hasattr(sch, "sigmas"):
                    s = sch.sigmas.float()
                    cur = s[i].item()
                    nxt = s[i + 1].item() if i + 1 < len(s) else s[i].item()
                    var = max(nxt**2 - cur**2, 0.0)
                    return float(var**0.5)
                elif hasattr(sch, "alphas_cumprod"):
                    ac = sch.alphas_cumprod.float()
                    cur = ac[i].item()
                    nxt = ac[i + 1].item() if i + 1 < len(ac) else ac[i].item()
                    sig2_cur = max(1.0 - cur, 0.0)
                    sig2_nxt = max(1.0 - nxt, 0.0)
                    var = max(sig2_nxt - sig2_cur, 0.0)
                    return float(var**0.5)
            except Exception:
                pass
            ts = sch.timesteps
            t_cur = float(ts[i].item())
            t_next = float(ts[i + 1].item()) if i + 1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
            return float(dt_unit**0.5)

        def _make_orthonormal_noise_like(x: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            M = x[0].numel()
            y = torch.randn(B, M, device=x.device, dtype=torch.float32)
            Q, _ = torch.linalg.qr(y.t(), mode='reduced')  # [M,B]
            Qb = Q.t().to(dtype=x.dtype).contiguous().view_as(x)
            return Qb

        def _inject_init_ortho(latents: torch.Tensor, strength: float, mode: str = 'blend') -> torch.Tensor:
            if strength <= 0 or mode == 'off':
                return latents

            with torch.no_grad():
                U = _make_orthonormal_noise_like(latents)
                base_norm = (latents.flatten(1).norm(dim=1) + 1e-12)
                U_norm = (U.flatten(1).norm(dim=1) + 1e-12)
                U = U * (base_norm / U_norm).view(-1, 1, 1, 1)

                if mode == 'replace':
                    out = (1.0 - strength) * latents + strength * U
                else:  # blend
                    out = latents + strength * U

            return out

        # ===== 回调：一步步注入多样性（带 Heun 预测端点） =====
        def diversity_callback(ppl, i, t, kw):
            ts = ppl.scheduler.timesteps
            t_cur = float(ts[i].item())
            t_next = float(ts[i + 1].item()) if i + 1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

            # 第 0 步：安全混入批内正交噪声
            if i == 0 and "latents" in kw and kw["latents"] is not None:
                kw["latents"] = _inject_init_ortho(
                    kw["latents"],
                    strength=float(getattr(args, "init_ortho", 0.0)),
                    mode=str(getattr(args, "init_ortho_mode", "blend")),
                )

            lat = kw.get("latents")
            if lat is None:
                return kw

            gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
            if gamma_sched <= 0:
                state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                state["prev_dt_unit"] = dt_unit
                return kw

            lat_new = lat.clone()

            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
            B = lat_vae_full.size(0)
            chunk = 2 if B >= 2 else 1

            prev_cpu = state.get("prev_latents_vae_cpu", None)

            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                v_est = None
                if prev_cpu is not None:
                    z_prev = prev_cpu[s:e].to(dev_vae, non_blocking=True)
                    z_det = z.detach()
                    total_diff = z_det - z_prev
                    prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                    prev_dt = state.get("prev_dt_unit", None)

                    if (prev_ctrl is not None) and (prev_dt is not None):
                        ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True)
                        base_move_prev = total_diff - ctrl_prev
                        v_est = base_move_prev / max(prev_dt, 1e-8)
                    else:
                        v_est = total_diff / max(dt_unit, 1e-8)

                def _decode_with_endpoint(z_var: torch.Tensor) -> torch.Tensor:
                    if v_est is not None:
                        z_pred = z_var + dt_unit * v_est
                    else:
                        z_pred = z_var
                    return _vae_decode_pixels(z_pred)

                with torch.enable_grad(), torch.backends.cudnn.flags(
                    enabled=False, benchmark=False, deterministic=False
                ):
                    imgs_chunk = checkpoint(_decode_with_endpoint, z, use_reentrant=False)

                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                current_logdet = float(_logs.get("logdet", 0.0))

                last_logdet = state.get("last_logdet", None)
                if (last_logdet is not None) and (current_logdet < last_logdet):
                    gamma_sched = 0.5 * gamma_sched
                state["last_logdet"] = current_logdet

                grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                grad_lat = torch.autograd.grad(
                    outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                    retain_graph=False, create_graph=False, allow_unused=False
                )[0]

                g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                div_disp = g_proj * dt_unit

                brown_std = _brownian_std_from_scheduler(ppl, i)
                eta = float(args.eta_sde)
                base_brown = eta * brown_std

                if args.noise_timing == 'early':
                    rho_t = args.rho0 * t_norm + args.rho1 * (1.0 - t_norm)
                else:
                    rho_t = args.rho1 * t_norm + args.rho0 * (1.0 - t_norm)

                base_disp = (v_est * dt_unit) if v_est is not None else z
                base_norm = _bn(base_disp)

                target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
                target_snr = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                target = torch.minimum(target_brown, target_snr)

                xi = torch.randn_like(g_proj)

                vnorm = _bn(v_est) if v_est is not None else None
                if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                    xi_eff = xi
                else:
                    xi_eff = project_partial_orth(xi, v_est, float(args.noise_partial_ortho))

                xi_norm = _bn(xi_eff)
                noise_disp = xi_eff / (xi_norm.view(-1, 1, 1, 1) + 1e-12) * target.view(-1, 1, 1, 1)

                disp_cap = cfg.gamma_max_ratio * _bn(base_disp)
                div_raw = _bn(div_disp)
                scale = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                delta_chunk = (gamma_sched * scale.view(-1, 1, 1, 1)) * div_disp + noise_disp

                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                lat_new[s:e] = lat_new[s:e] + delta_tr

                if "ctrl_cache" not in state:
                    state["ctrl_cache"] = []
                state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                if args.debug and (s == 0):
                    with torch.no_grad():
                        n_div = _bn(div_disp).mean().item()
                        n_noise = _bn(noise_disp).mean().item()
                        n_base = _bn(base_disp).mean().item()
                        print(f"[t={t_norm:.2f}] |base|={n_base:.4f} |div|={n_div:.4f} |noise|={n_noise:.4f} "
                              f"gamma={gamma_sched:.4f} brown={brown_std:.4f} rho_t={rho_t:.3f}")

                if dev_clip.type == 'cuda':
                    torch.cuda.synchronize(dev_clip)
                if dev_vae.type == 'cuda':
                    torch.cuda.synchronize(dev_vae)
                del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, z, v_est

            kw["latents"] = lat_new

            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

            if "ctrl_cache" in state:
                state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
                del state["ctrl_cache"]
            state["prev_dt_unit"] = dt_unit
            return kw

        # ===== 5) 遍历 concept / prompt / seed / guidance =====
        outputs_root = os.path.join(project_root, 'outputs')

        for concept in concept_list:
            prompts = spec[concept]
            if not isinstance(prompts, list):
                continue

            concept_slug = _slugify(str(concept))
            concept_dirname = f"{args.method}_{concept_slug or 'no_concept'}"
            base_out_dir = (
                args.out.strip()
                if (args.out and len(args.out.strip()) > 0)
                else os.path.join(outputs_root, concept_dirname)
            )
            imgs_root = os.path.join(base_out_dir, "imgs")
            eval_dir = os.path.join(base_out_dir, "eval")
            os.makedirs(imgs_root, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)
            _log(f"Concept={concept} -> base output dir: {imgs_root}", True)

            for prompt in prompts:
                prompt_slug = _slugify(prompt)
                _log(f"  Prompt: {prompt} (slug={prompt_slug})", True)

                for seed in seeds:
                    for g in guidances:
                        run_dir_name = f"{prompt_slug}_seed{seed}_g{g}_s{args.steps}"
                        out_dir = os.path.join(imgs_root, run_dir_name)
                        os.makedirs(out_dir, exist_ok=True)
                        _log(f"    Run dir: {out_dir}", True)

                        # 每个 run 重置状态
                        reset_state()

                        generator = torch.Generator(device=dev_tr) if dev_tr.type == 'cuda' else torch.Generator()
                        generator.manual_seed(seed)

                        _log(
                            f"    Start SDXL-Turbo pipeline() "
                            f"(concept={concept}, prompt_slug={prompt_slug}, seed={seed}, g={g}) ...",
                            args.debug,
                        )

                        latents_out = pipe(
                            prompt=prompt,
                            height=args.height,
                            width=args.width,
                            num_images_per_prompt=args.G,
                            num_inference_steps=args.steps,
                            guidance_scale=g,
                            negative_prompt=(args.negative if args.negative else None),
                            generator=generator,
                            callback_on_step_end=diversity_callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="latent",
                            return_dict=False,
                        )[0]  # latent tensor（在 transformer/unet 卡）

                        # ===== decode =====
                        latents_final = latents_out.to(dev_vae, non_blocking=True)

                        if args.enable_vae_tiling and hasattr(pipe, "vae"):
                            if hasattr(pipe.vae, "enable_slicing"):
                                pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, "enable_tiling"):
                                pipe.vae.enable_tiling()

                        with torch.inference_mode(), torch.backends.cudnn.flags(
                            enabled=False, benchmark=False, deterministic=False
                        ):
                            images = checkpoint(_vae_decode_pixels, latents_final, use_reentrant=False)

                        for i in range(images.size(0)):
                            save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

                        _log(
                            f"    Done (concept={concept}, prompt_slug={prompt_slug}, "
                            f"seed={seed}, g={g}). Saved to {out_dir}",
                            True,
                        )

        _log("All concepts/prompts/seeds/guidances done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
