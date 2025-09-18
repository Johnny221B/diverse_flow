#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT) [with A+B noise + init orthogonalization]

特点：
- 单次生成：指定 prompt，G 张图，固定 guidance 与 steps（可改）
- 回调：VAE 解码 -> CLIP 体积损失 -> VJP 回 latent -> 写回
- 确定性项（体积漂移）仅受 gamma(t)（t_gate+sched_shape）调度
- 噪声：SDE/Brownian 对齐 + 相对 SNR 目标（早强/晚强可配）
- 批内初始 latent 正交化（i==0 时对 kw["latents"] 进行）

输出：
- outputs/<method>_<prompt_slug>/{imgs,eval}/
"""

import os
import re
import sys
import time
import argparse
import traceback
from typing import Any, Dict, Optional

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
        for p in sm.parameters(recurse=False):
            return p.device
        for b in sm.buffers(recurse=False):
            return b.device
    return None

def inspect_pipe_devices(pipe):
    names = [
        "transformer",
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
        description='Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)'
    )

    # 生成参数
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=64)  # 默认改为 64
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=42)

    # 本地模型路径
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # 输出与方法名
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='noise')

    # 多样性目标（方法相关）
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)

    # 正交/门控
    ap.add_argument('--partial-ortho', type=float, default=0.95)   # 体积力对基流正交比例
    ap.add_argument('--t-gate', type=str, default='0.85,0.95')     # 仅用于确定性体积漂移
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # 噪声时间形态（新增）：early=前期强，late=后期强
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])

    # -------- 新增（A+B）噪声控制参数 --------
    ap.add_argument('--eta-sde', type=float, default=1.0, help='SDE/Brownian 噪声强度 (与 scheduler 对齐)')   # [A]
    ap.add_argument('--rho0', type=float, default=0.25, help='相对 SNR 目标（早期）')                           # [B]
    ap.add_argument('--rho1', type=float, default=0.05, help='相对 SNR 目标（末期）')                           # [B]
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0, help='噪声相对基流的正交比例')             # [B]
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4, help='基流范数过小则跳过噪声投影的阈值')        # [B]

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

    # TE 默认与 Transformer 同卡
    if args.device_text1 is None:
        args.device_text1 = args.device_transformer
    if args.device_text2 is None:
        args.device_text2 = args.device_transformer
    if args.device_text3 is None:
        args.device_text3 = args.device_transformer

    try:
        import torch.backends.cudnn as cudnn
        from torch.utils.checkpoint import checkpoint
        from diffusers import StableDiffusion3Pipeline
        from torchvision.utils import save_image  # noqa

        from diverse_flow.config import DiversityConfig
        from diverse_flow.clip_wrapper import CLIPWrapper
        from diverse_flow.volume_objective import VolumeObjective
        from diverse_flow.utils import project_partial_orth, batched_norm as _bn
        from diverse_flow.utils import sched_factor as time_sched_factor

        # 设备 + dtype
        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_clip= torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, clip={dev_clip}", args.debug)
        _log(f"Model dir: {args.model_dir}", args.debug)
        _log(f"CLIP JIT: {args.clip_jit}", args.debug)
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

        # ===== 1) CPU 加载，再手动上卡 =====
        model_dir = _resolve_model_dir(args.model_dir)
        _log("Loading SD3.5 (CPU) ...", args.debug)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True,
        )
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")
        print("scheduler:", pipe.scheduler.__class__.__name__)

        _log("Moving modules to target devices ...", args.debug)
        if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
        if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        if args.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                _log(f"enable_xformers failed: {e}", args.debug)

        inspect_pipe_devices(pipe)
        if hasattr(pipe, "transformer"):    assert_on(pipe.transformer, dev_tr)
        if hasattr(pipe, "text_encoder"):   assert_on(pipe.text_encoder,   dev_tr)
        if hasattr(pipe, "text_encoder_2"): assert_on(pipe.text_encoder_2, dev_tr)
        if hasattr(pipe, "text_encoder_3"): assert_on(pipe.text_encoder_3, dev_tr)
        if hasattr(pipe, "vae"):            assert_on(pipe.vae,            dev_vae)

        # ===== 2) CLIP & Volume objective =====
        _log("Loading CLIP ...", args.debug)
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
        )
        _log("CLIP ready.", args.debug)

        t0, t1 = args.t_gate.split(',')
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
            sched_shape=args.sched_shape, clip_image_size=224,
            leverage_alpha=0.5,
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # ===== 3) 状态 & 工具函数 =====
        state: Dict[str, Optional[torch.Tensor]] = {
            "prev_latents_vae_cpu": None,
            "prev_ctrl_vae_cpu":   None,
            "prev_dt_unit":        None,
            "prev_prev_latents_vae_cpu": None,
            "last_logdet":         None,
            "gamma_auto_done":     False,
        }

        def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

        def _beta_sched(t_norm: float, eps: float = 1e-2) -> float:
            # 现仅用于相对 SNR 插值（保持接口）
            t = float(t_norm)
            if args.noise_timing == 'early':
                base = min(1.0, t / (1.0 - t + eps))
            else:
                base = min(1.0, (1.0 - t) / (t + eps))
            return float(base)

        # [A] Brownian 尺度（尽可能对齐调度器）
        def _brownian_std_from_scheduler(ppl, i):
            sch = ppl.scheduler
            try:
                if hasattr(sch, "sigmas"):
                    s = sch.sigmas.float()
                    cur = s[i].item()
                    nxt = s[i+1].item() if i+1 < len(s) else s[i].item()
                    var = max(nxt**2 - cur**2, 0.0)
                    return float(var**0.5)
                elif hasattr(sch, "alphas_cumprod"):
                    ac = sch.alphas_cumprod.float()
                    cur = ac[i].item()
                    nxt = ac[i+1].item() if i+1 < len(ac) else ac[i].item()
                    sig2_cur = max(1.0 - cur, 0.0)
                    sig2_nxt = max(1.0 - nxt, 0.0)
                    var = max(sig2_nxt - sig2_cur, 0.0)
                    return float(var**0.5)
            except Exception:
                pass
            # 回退：用归一化步长 √Δt
            ts = sch.timesteps
            t_cur  = float(ts[i].item())
            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
            return float(dt_unit**0.5)

        # === 批内初噪正交化（你的实现，原样并入） ===
        def _batch_orthogonalize(x, eps: float = 1e-8):
            # x: [B, C, H, W]
            B = x.size(0)
            y = x.view(B, -1)  # [B, M]
            Q = torch.zeros_like(y)
            for i in range(B):
                v = y[i]
                for j in range(i):
                    v = v - (v @ Q[j]) * Q[j]
                n = v.norm(p=2) + eps
                Q[i] = v / n
            return Q.view_as(x)

        # ===== 回调：一步步注入多样性 =====
        def diversity_callback(ppl, i, t, kw):
            ts = ppl.scheduler.timesteps
            t_cur  = float(ts[i].item())
            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

            # ---------- 批内初始 latent 正交化（只在第 0 步） ----------
            if i == 0 and "latents" in kw and kw["latents"] is not None:
                with torch.no_grad():
                    kw["latents"] = _batch_orthogonalize(kw["latents"])

            lat = kw.get("latents")
            if lat is None:
                return kw

            # γ 仅用于确定性体积漂移（gating by t_gate）
            gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
            if gamma_sched <= 0:
                state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                state["prev_dt_unit"] = dt_unit
                return kw

            lat_new = lat.clone()

            # 把本步 latent 搬到 VAE 卡
            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
            B = lat_vae_full.size(0)
            chunk = 2 if B >= 2 else 1

            prev_cpu = state.get("prev_latents_vae_cpu", None)

            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                # —— 解码到像素 —— #
                with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

                # —— CLIP 卡上求体积损失对“像素”的梯度 ——
                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                current_logdet = float(_logs.get("logdet", 0.0))

                # 能量单调守门（若下降则削弱 gamma）
                last_logdet = state.get("last_logdet", None)
                if (last_logdet is not None) and (current_logdet < last_logdet):
                    gamma_sched = 0.5 * gamma_sched
                state["last_logdet"] = current_logdet

                grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                # —— VJP 回 latent —— #
                grad_lat = torch.autograd.grad(
                    outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                    retain_graph=False, create_graph=False, allow_unused=False
                )[0]  # [bs,C,h,w] on VAE

                # —— 基流速度估计：v_est = Δz / Δt ——
                v_est = None
                if prev_cpu is not None:
                    total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)
                    prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                    prev_dt   = state.get("prev_dt_unit", None)
                    if (prev_ctrl is not None) and (prev_dt is not None):
                        ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True)
                        base_move_prev = total_diff - ctrl_prev
                        v_est = base_move_prev / max(prev_dt, 1e-8)
                    else:
                        v_est = total_diff / max(dt_unit, 1e-8)

                # —— 体积力：对基流（部分/全）正交 —— #
                g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                div_disp = g_proj * dt_unit  # 确定性体积位移（γ 只作用这里）

                # ===================== 噪声项（A + B）=====================
                brown_std = _brownian_std_from_scheduler(ppl, i)  # 标量
                eta = float(args.eta_sde)
                base_brown = eta * brown_std

                if args.noise_timing == 'early':
                    rho_t = args.rho0 * t_norm + args.rho1 * (1.0 - t_norm)
                else:
                    rho_t = args.rho1 * t_norm + args.rho0 * (1.0 - t_norm)

                base_disp = (v_est * dt_unit) if v_est is not None else z
                base_norm = _bn(base_disp)  # [bs]

                target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
                target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                target = torch.minimum(target_brown, target_snr)  # [bs]

                xi = torch.randn_like(g_proj)

                vnorm = _bn(v_est) if v_est is not None else None
                if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                    xi_eff = xi
                else:
                    xi_eff = project_partial_orth(xi, v_est, float(args.noise_partial_ortho))

                xi_norm = _bn(xi_eff)  # [bs]
                noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)
                # =================== 噪声项（A + B）结束 ===================

                # 可选信赖域：仅限幅体积位移（不限制噪声）
                disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                div_raw   = _bn(div_disp)
                scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                # 最终位移写回：γ 只缩放体积位移；噪声直接叠加
                delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp

                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                lat_new[s:e] = lat_new[s:e] + delta_tr

                # —— 缓存控制位移（CPU）——
                if "ctrl_cache" not in state:
                    state["ctrl_cache"] = []
                state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                if args.debug and (s == 0):
                    with torch.no_grad():
                        n_div   = _bn(div_disp).mean().item()
                        n_noise = _bn(noise_disp).mean().item()
                        n_base  = _bn(base_disp).mean().item()
                        print(f"[t={t_norm:.2f}] |base|={n_base:.4f} |div|={n_div:.4f} |noise|={n_noise:.4f} "
                              f"gamma={gamma_sched:.4f} brown={brown_std:.4f} rho_t={rho_t:.3f}")

                if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

            kw["latents"] = lat_new

            # prev/prev_prev
            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

            # 控制位移与 dt
            if "ctrl_cache" in state:
                state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
                del state["ctrl_cache"]
            state["prev_dt_unit"] = dt_unit
            return kw

        # ===== 4) 输出目录 =====
        prompt_slug = _slugify(args.prompt)
        outputs_root = os.path.join(project_root, 'outputs')
        auto_dirname = f"{args.method}_{prompt_slug or 'no_prompt'}"
        base_out_dir = args.out.strip() if (args.out and len(args.out.strip()) > 0) else os.path.join(outputs_root, auto_dirname)
        out_dir = os.path.join(base_out_dir, "imgs")
        eval_dir = os.path.join(base_out_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        _log(f"Output dir: {out_dir}", True)

        # ===== 5) 生成 latent（不让管线内部 decode） =====
        generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
        generator.manual_seed(args.seed)

        _log("Start pipeline() ...", args.debug)
        latents_out = pipe(
            prompt=args.prompt,
            negative_prompt=(args.negative if args.negative else None),
            height=args.height, width=args.width,
            num_images_per_prompt=args.G,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            callback_on_step_end=diversity_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
            return_dict=False,
        )[0]  # -> latent tensor（在 transformer 卡）

        # ===== 6) 手动在 VAE 卡 decode 最终 latent =====
        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
        latents_final = latents_out.to(dev_vae, non_blocking=True)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            images = checkpoint(
                lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                latents_final, use_reentrant=False
            )

        from torchvision.utils import save_image
        for i in range(images.size(0)):
            save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

        _log(f"Done. Saved to {out_dir}", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()