#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)
[with A+B noise + safe init-orthogonal injection + Heun-style endpoint volume]
- 支持读取 JSON 文件进行批量生成
- 自动按照 {method}_{concept}/imgs/{prompt_slug} 分类存放
"""

import os
import re
import sys
import time
import json
import argparse
import traceback
from typing import Any, Dict, Optional
from collections import OrderedDict

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
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON spec (e.g. mini_color.json)')
    ap.add_argument('--prompt', type=str, default=None, help='Fallback single prompt if spec is not provided')
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=16)  
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=1111)

    # 本地模型路径
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # 输出与方法名
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='modified')

    # 多样性目标（方法相关）
    ap.add_argument('--gamma0', type=float, default=0.06)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.10)

    # 正交/门控
    ap.add_argument('--partial-ortho', type=float, default=0.95)   
    ap.add_argument('--t-gate', type=str, default='0.85,0.99')     
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # 噪声时间形态
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])

    # 噪声控制（A+B）
    ap.add_argument('--eta-sde', type=float, default=0.5)
    ap.add_argument('--rho0', type=float, default=0.25)
    ap.add_argument('--rho1', type=float, default=0.05)
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)
    ap.add_argument('--vae-chunk', type=int, default=4,
                    help='Sub-batch size for VAE decode inside callback (trade OOM vs speed)')
    ap.add_argument('--update-every', type=int, default=1,
                    help='Apply diversity gradient only every K steps (1=every step). '
                         'K=2 halves the CLIP/VAE overhead with minimal quality loss.')

    # Memory bank (offline diversity for small m)
    ap.add_argument('--memory-bank-size', type=int, default=0,
                    help='Memory bank size (0=disabled). When >0, diversity signal is '
                         'augmented with up to N cached features from previous prompts, '
                         'enabling effective diversity guidance even at small batch sizes.')
    ap.add_argument('--memory-bank-decay', type=float, default=0.0,
                    help='Exponential decay for older bank entries (0=uniform weight)')

    # 安全的“初噪正交化”注入
    ap.add_argument('--init-ortho', type=float, default=0.2)
    ap.add_argument('--init-ortho-mode', type=str, default='blend', choices=['blend','off','replace'])

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

    # ===== sys.path 注入 =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 解析 Spec 或者 Prompt
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as f:
            spec_data = json.load(f, object_pairs_hook=OrderedDict)
        # 兼容 { "color": ["prompt1", "prompt2"], "complex": [...] }
        concept_to_prompts = spec_data
    elif args.prompt:
        concept_to_prompts = {"single": [args.prompt]}
    else:
        raise ValueError("Must provide either --spec or --prompt")

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

        # ===== 1) CPU 加载，再手动上卡 =====
        model_dir = _resolve_model_dir(args.model_dir)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, torch_dtype=dtype, local_files_only=True,
        )
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")

        if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
        if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        if args.enable_xformers:
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception as e: _log(f"enable_xformers failed: {e}", args.debug)

        # ===== 2) CLIP & Volume objective =====
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
        )

        t0, t1 = args.t_gate.split(',')
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
            sched_shape=args.sched_shape, clip_image_size=224,
            leverage_alpha=0.5,
        )
        vol = VolumeObjective(clip, cfg)

        # Memory bank for offline diversity (small-m support)
        memory_bank = None
        if args.memory_bank_size > 0:
            from diverse_flow.memory_bank import OfflineMemoryBank
            memory_bank = OfflineMemoryBank(
                max_size=args.memory_bank_size,
                decay=args.memory_bank_decay,
                device="cpu",
            )
            _log(f"Memory bank enabled: max_size={args.memory_bank_size}, "
                 f"decay={args.memory_bank_decay}", True)

        # --------------------- 循环处理逻辑 ---------------------
        outputs_root = os.path.join(project_root, 'outputs')

        for concept, prompts in concept_to_prompts.items():
            # 1. 建立 Concept 目录： outputs/ourmethod_color/
            base_out_dir = os.path.join(outputs_root, f"{args.method}_{concept}")
            imgs_root_dir = os.path.join(base_out_dir, "imgs")
            eval_dir = os.path.join(base_out_dir, "eval")
            
            os.makedirs(imgs_root_dir, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)
            _log(f"\n=== Starting Concept: {concept} ===", True)
            _log(f"  Saving to: {base_out_dir}", True)

            for prompt_text in prompts:
                prompt_slug = _slugify(prompt_text)
                # 2. 建立 Prompt 子目录： outputs/ourmethod_color/imgs/a_red_apple/
                run_dir = os.path.join(imgs_root_dir, prompt_slug)

                # 断点续跑支持
                if os.path.exists(run_dir) and len(os.listdir(run_dir)) >= args.G:
                    _log(f"  [SKIP] '{prompt_text[:40]}...' already exists.", True)
                    continue
                
                os.makedirs(run_dir, exist_ok=True)
                _log(f"  [RUN] Prompt: '{prompt_text}'", True)

                # ===== 3) 重置 State =====
                state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_latents_vae_cpu": None,
                    "prev_ctrl_vae_cpu":   None,
                    "prev_dt_unit":        None,
                    "prev_prev_latents_vae_cpu": None,
                    "last_logdet":         None,
                    "gamma_auto_done":     False,
                }

                # 提取出的工具函数
                def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
                    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                    out = pipe.vae.decode(z / sf, return_dict=False)[0] 
                    return (out.float().clamp(-1,1) + 1.0) / 2.0        

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
                    ts = sch.timesteps
                    t_cur  = float(ts[i].item())
                    t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                    t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                    dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
                    return float(dt_unit**0.5)

                def _make_orthonormal_noise_like(x: torch.Tensor) -> torch.Tensor:
                    B = x.size(0)
                    M = x[0].numel()
                    y = torch.randn(B, M, device=x.device, dtype=torch.float32)  
                    Q, _ = torch.linalg.qr(y.t(), mode='reduced') 
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
                        else:  
                            out = latents + strength * U
                    return out

                # ===== 当前 Prompt 的回调函数 =====
                def diversity_callback(ppl, i, t, kw):
                    ts = ppl.scheduler.timesteps
                    t_cur  = float(ts[i].item())
                    t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                    t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                    t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
                    dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

                    if i == 0 and "latents" in kw and kw["latents"] is not None:
                        kw["latents"] = _inject_init_ortho(
                            kw["latents"],
                            strength=float(getattr(args, "init_ortho", 0.0)),
                            mode=str(getattr(args, "init_ortho_mode", "blend")),
                        )

                    lat = kw.get("latents")
                    if lat is None: return kw

                    gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
                    if gamma_sched <= 0:
                        state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                        state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                        state["prev_dt_unit"] = dt_unit
                        return kw

                    # ── update-every: skip diversity gradient on non-update steps ──
                    update_every = max(1, int(getattr(args, "update_every", 1)))
                    if update_every > 1 and (i % update_every) != 0:
                        state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                        state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                        state["prev_dt_unit"] = dt_unit
                        return kw

                    lat_new = lat.clone()
                    lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
                    B = lat_vae_full.size(0)
                    vae_chunk = max(1, min(int(getattr(args, "vae_chunk", 4)), B))

                    prev_cpu = state.get("prev_latents_vae_cpu", None)

                    # ── Build per-sample v_est (full batch) ──────────────────────────
                    v_est_all = None
                    if prev_cpu is not None:
                        z_prev_full = prev_cpu.to(dev_vae, non_blocking=True)
                        total_diff  = lat_vae_full.detach() - z_prev_full
                        prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                        prev_dt   = state.get("prev_dt_unit", None)
                        if (prev_ctrl is not None) and (prev_dt is not None):
                            ctrl_prev_full = prev_ctrl.to(dev_vae, non_blocking=True)
                            v_est_all = (total_diff - ctrl_prev_full) / max(prev_dt, 1e-8)
                        else:
                            v_est_all = total_diff / max(dt_unit, 1e-8)

                    # ── Pass A: decode ALL endpoint latents (no-grad) for global CLIP gradient ──
                    # Key fix: build the B×B Gram matrix over all images simultaneously,
                    # so every trajectory repels ALL others (not just its chunk-partner).
                    with torch.no_grad():
                        imgs_nograd_list = []
                        for s_a in range(0, B, vae_chunk):
                            e_a = min(B, s_a + vae_chunk)
                            z_chunk_a = lat_vae_full[s_a:e_a]
                            if v_est_all is not None:
                                z_pred_a = z_chunk_a + dt_unit * v_est_all[s_a:e_a]
                            else:
                                z_pred_a = z_chunk_a
                            imgs_nograd_list.append(_vae_decode_pixels(z_pred_a))
                        imgs_nograd_all = torch.cat(imgs_nograd_list, dim=0)  # [B,3,H,W]

                    # ── Pass B: global volume gradient (B×B Gram matrix) ───────────────
                    phi_mem = memory_bank.get(query_device=dev_clip) \
                              if memory_bank is not None else None
                    _loss, grad_img_all, _logs = vol.volume_loss_and_grad(
                        imgs_nograd_all.to(dev_clip), phi_memory=phi_mem)   # [B,3,H,W]
                    current_logdet = float(_logs.get("logdet", 0.0))

                    last_logdet = state.get("last_logdet", None)
                    if (last_logdet is not None) and (current_logdet < last_logdet):
                        gamma_sched = 0.5 * gamma_sched
                    state["last_logdet"] = current_logdet

                    grad_img_vae_all = grad_img_all.to(dev_vae, non_blocking=True)
                    del imgs_nograd_all

                    # ── Pass C: per-chunk VAE VJP (pixel grad → latent grad) ───────────
                    # Recompute each chunk WITH grad so autograd can backprop through VAE.
                    # IMPORTANT: z_pred_chunk must be created INSIDE torch.enable_grad()
                    # because the outer pipeline context is @torch.no_grad() and arithmetic
                    # ops outside enable_grad won't build a grad_fn.
                    ctrl_chunks = []
                    for s in range(0, B, vae_chunk):
                        e = min(B, s + vae_chunk)

                        z_chunk   = lat_vae_full[s:e].detach().clone().requires_grad_(True)
                        v_chunk_s = v_est_all[s:e].detach() if v_est_all is not None else None

                        def _decode_endpoint_chunk(z_var, _v=v_chunk_s, _dt=dt_unit):
                            z_pred = z_var + _dt * _v if _v is not None else z_var
                            return _vae_decode_pixels(z_pred)

                        with torch.enable_grad(), torch.backends.cudnn.flags(
                                enabled=False, benchmark=False, deterministic=False):
                            imgs_chunk = checkpoint(
                                _decode_endpoint_chunk, z_chunk, use_reentrant=False)

                        go = grad_img_vae_all[s:e].to(dev_vae).to(imgs_chunk.dtype)
                        grad_lat_chunk = torch.autograd.grad(
                            outputs=imgs_chunk, inputs=z_chunk,
                            grad_outputs=go,
                            retain_graph=False, create_graph=False
                        )[0]

                        v_chunk = v_est_all[s:e] if v_est_all is not None else None
                        g_proj  = project_partial_orth(grad_lat_chunk, v_chunk, cfg.partial_ortho) \
                                  if v_chunk is not None else grad_lat_chunk
                        div_disp = g_proj * dt_unit

                        # ── Noise injection ───────────────────────────────────────
                        brown_std = _brownian_std_from_scheduler(ppl, i)
                        eta       = float(args.eta_sde)
                        base_brown = eta * brown_std

                        if args.noise_timing == 'early':
                            rho_t = args.rho0 * t_norm + args.rho1 * (1.0 - t_norm)
                        else:
                            rho_t = args.rho1 * t_norm + args.rho0 * (1.0 - t_norm)

                        base_disp = (v_chunk * dt_unit) if v_chunk is not None \
                                    else lat_vae_full[s:e].detach()
                        base_norm = _bn(base_disp)

                        target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
                        target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                        target       = torch.minimum(target_brown, target_snr)

                        xi = torch.randn_like(g_proj)
                        vnorm = _bn(v_chunk) if v_chunk is not None else None
                        if (v_chunk is None) or (vnorm is None) or \
                                (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                            xi_eff = xi
                        else:
                            xi_eff = project_partial_orth(xi, v_chunk, float(args.noise_partial_ortho))

                        xi_norm    = _bn(xi_eff)
                        noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)

                        disp_cap = cfg.gamma_max_ratio * _bn(base_disp)
                        div_raw  = _bn(div_disp)
                        scale    = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                        delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp
                        delta_tr    = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                        lat_new[s:e] = lat_new[s:e] + delta_tr
                        ctrl_chunks.append(delta_chunk.detach().to("cpu"))

                    if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                    if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                    del grad_img_all, grad_img_vae_all

                    kw["latents"] = lat_new
                    state["prev_ctrl_vae_cpu"] = torch.cat(ctrl_chunks, dim=0)

                    state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                    state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
                    state["prev_dt_unit"] = dt_unit
                    return kw

                # ===== 4) 生成 Pipeline =====
                generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                generator.manual_seed(args.seed)

                latents_out = pipe(
                    prompt=prompt_text,
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
                )[0]

                # ===== 5) VAE 解码与保存 =====
                sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                latents_final = latents_out.to(dev_vae, non_blocking=True)

                with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    images = checkpoint(
                        lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                        latents_final, use_reentrant=False
                    )

                for i in range(images.size(0)):
                    save_image(images[i].cpu(), os.path.join(run_dir, f"{i:03d}.png"))

                # ===== 6) Update memory bank with final-image features =====
                if memory_bank is not None:
                    with torch.no_grad():
                        imgs_for_bank = images.detach().to(dev_clip, dtype=torch.float32)
                        phi_final = clip.encode_image_from_pixels(imgs_for_bank, size=224)
                        phi_final = F.normalize(phi_final, dim=-1)
                        memory_bank.update(phi_final)
                    _log(f"  [Bank] updated, size={memory_bank.size()}", True)

                # ===== 7) 彻底释放本轮 Prompt 占用的显存 =====
                del latents_out, latents_final, images
                torch.cuda.empty_cache()

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()