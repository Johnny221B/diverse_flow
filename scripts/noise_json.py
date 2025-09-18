#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)
[with A+B noise + safe init-orthogonal injection]

扩展：
- 支持 --spec 指向 JSON: { "truck": [...], "bus": [...], ... }
- 遍历 concept × prompt × guidance × seed
- 输出到 outputs/<method>_<concept>/imgs/<prompt_slug>_seed{SEED}_g{GUIDANCE}_s{STEPS}/
"""

import os
import re
import sys
import time
import json
import argparse
import traceback
from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict
from pathlib import Path

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

# 新增：解析 {concept: [prompts...]}
def _parse_concepts_spec(obj: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    if not isinstance(obj, dict):
        raise ValueError("Spec must be a JSON object: {concept: [prompts...]}")
    out = OrderedDict()
    for concept, plist in obj.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s); cleaned.append(s)
        if cleaned:
            out[concept] = cleaned
    if not out:
        raise ValueError("No valid {concept: [prompts...]} found in spec.")
    return out

def _build_root_out(project_root: str, method: str, concept: str) -> Tuple[str, str, str]:
    base = os.path.join(project_root, "outputs", f"{method}_{_slugify(concept)}")
    imgs = os.path.join(base, "imgs")
    eval_dir = os.path.join(base, "eval")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return base, imgs, eval_dir

def _prompt_run_dir(imgs_root: str, prompt: str, seed: int, guidance: float, steps: int) -> str:
    pslug = _slugify(prompt)
    return os.path.join(imgs_root, f"{pslug}_seed{seed}_g{guidance}_s{steps}")


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)'
    )

    # 新增：多 concept JSON；保留单 prompt 回退
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec not provided')
    ap.add_argument('--negative', type=str, default='')

    # 生成参数
    ap.add_argument('--G', type=int, default=64)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)

    # 新增：多 guidance/seed；保留单值回退（兼容原脚本）
    ap.add_argument('--guidances', type=float, nargs='+', default=[3.0, 5.0, 7.5])
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])
    ap.add_argument('--seed', type=int, default=42)

    # 本地模型路径
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # 输出与方法名
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='ourmethod')  # 统一为小写，按你描述

    # 多样性目标（方法相关，保持原样）
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--t-gate', type=str, default='0.85,0.95')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # 噪声时间形态
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])

    # 噪声控制（A+B）
    ap.add_argument('--eta-sde', type=float, default=1.0)
    ap.add_argument('--rho0', type=float, default=0.25)
    ap.add_argument('--rho1', type=float, default=0.05)
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)

    # 初噪正交化
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

    # ===== sys.path 注入（让脚本能 import diverse_flow） =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 解析 JSON / 单 prompt
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec_obj = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --spec (JSON with {concept:[prompts...]}) or --prompt")

    # TE 默认与 Transformer 同卡
    if args.device_text1 is None:
        args.device_text1 = args.device_transformer
    if args.device_text2 is None:
        args.device_text2 = args.device_transformer
    if args.device_text3 is None:
        args.device_text3 = args.device_transformer

    # 生效的 guidances / seeds
    guidances = args.guidances if args.guidances else [args.guidance]
    seeds = args.seeds if args.seeds else [args.seed]

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

        # ===== 一些内部工具（保持原有方法一致） =====
        def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

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
            y = torch.randn(B, M, device=x.device, dtype=torch.float32)  # [B,M]
            Q, _ = torch.linalg.qr(y.t(), mode='reduced')                 # [M,B]
            Qb = Q.t().to(dtype=x.dtype).contiguous().view_as(x)          # [B,C,H,W]
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

        # ====== 主过程：遍历 concept × prompt × guidance × seed ======
        for concept, prompt_list in concept_to_prompts.items():
            base_dir, imgs_root, eval_dir = _build_root_out(project_root, args.method, concept)
            _log(f"[OUT] base={base_dir}", True)

            for prompt_text in prompt_list:
                for g in (guidances if len(guidances)>0 else [args.guidance]):
                    for sd in (seeds if len(seeds)>0 else [args.seed]):

                        # —— 每次 run 的状态（与原脚本一致）——
                        state: Dict[str, Optional[torch.Tensor]] = {
                            "prev_latents_vae_cpu": None,
                            "prev_ctrl_vae_cpu":   None,
                            "prev_dt_unit":        None,
                            "prev_prev_latents_vae_cpu": None,
                            "last_logdet":         None,
                            "gamma_auto_done":     False,
                        }

                        def diversity_callback(ppl, i, t, kw):
                            ts = ppl.scheduler.timesteps
                            t_cur  = float(ts[i].item())
                            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
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
                                    imgs_chunk = (pipe.vae.decode(z / getattr(pipe.vae.config, "scaling_factor", 1.0), return_dict=False)[0].float().clamp(-1,1) + 1.0)/2.0

                                # —— CLIP 卡上求体积损失对“像素”的梯度 ——
                                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                                _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                                current_logdet = float(_logs.get("logdet", 0.0))

                                last_logdet = state.get("last_logdet", None)
                                if (last_logdet is not None) and (current_logdet < last_logdet):
                                    gamma_sched = 0.5 * gamma_sched
                                state["last_logdet"] = current_logdet

                                grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                                # —— VJP 回 latent —— #
                                grad_lat = torch.autograd.grad(
                                    outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                                    retain_graph=False, create_graph=False, allow_unused=False
                                )[0]

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

                                # 体积项（正交到基流）
                                from diverse_flow.utils import project_partial_orth, batched_norm as _bn
                                g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                                div_disp = g_proj * dt_unit

                                # 噪声项（A+B）
                                brown_std = _brownian_std_from_scheduler(ppl, i)
                                eta = float(args.eta_sde)
                                base_brown = eta * brown_std

                                if args.noise_timing == 'early':
                                    rho_t = args.rho0 * t_norm + args.rho1 * (1.0 - t_norm)
                                else:
                                    rho_t = args.rho1 * t_norm + args.rho0 * (1.0 - t_norm)

                                base_disp = (v_est * dt_unit) if v_est is not None else z
                                base_norm = _bn(base_disp)

                                import torch as _t
                                target_brown = _t.full_like(base_norm, fill_value=max(base_brown, 0.0))
                                target_snr   = _t.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                                target = _t.minimum(target_brown, target_snr)

                                xi = _t.randn_like(g_proj)
                                vnorm = _bn(v_est) if v_est is not None else None
                                if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                                    xi_eff = xi
                                else:
                                    xi_eff = project_partial_orth(xi, v_est, float(args.noise_partial_ortho))

                                xi_norm = _bn(xi_eff)
                                noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)

                                # 信赖域（仅限幅体积位移）
                                disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                                div_raw   = _bn(div_disp)
                                scale     = _t.minimum(_t.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                                delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp
                                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                                lat_new[s:e] = lat_new[s:e] + delta_tr

                                if "ctrl_cache" not in state:
                                    state["ctrl_cache"] = []
                                state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                                if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                                if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                                del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

                            kw["latents"] = lat_new

                            state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

                            if "ctrl_cache" in state:
                                import torch as _t
                                state["prev_ctrl_vae_cpu"] = _t.cat(state["ctrl_cache"], dim=0).to("cpu")
                                del state["ctrl_cache"]
                            state["prev_dt_unit"] = dt_unit
                            return kw

                        # ===== 输出目录（按 concept/prompt/seed/guidance）=====
                        run_dir = _prompt_run_dir(imgs_root, prompt_text, int(sd), float(g), int(args.steps))
                        os.makedirs(run_dir, exist_ok=True)
                        _log(f"[RUN] concept='{concept}' | prompt='{prompt_text}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}", True)

                        # ===== 生成 latent（不让管线内部 decode）=====
                        generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                        generator.manual_seed(int(sd))

                        latents_out = pipe(
                            prompt=prompt_text,
                            negative_prompt=(args.negative if args.negative else None),
                            height=args.height, width=args.width,
                            num_images_per_prompt=args.G,
                            num_inference_steps=args.steps,
                            guidance_scale=float(g),
                            generator=generator,
                            callback_on_step_end=diversity_callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="latent",
                            return_dict=False,
                        )[0]

                        # ===== 手动在 VAE 卡 decode 最终 latent =====
                        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                        latents_final = latents_out.to(dev_vae, non_blocking=True)
                        if args.enable_vae_tiling:
                            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

                        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                            images = (pipe.vae.decode(latents_final / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0

                        from torchvision.utils import save_image
                        for i in range(images.size(0)):
                            save_image(images[i].cpu(), os.path.join(run_dir, f"img_{i:03d}.png"))

                        # 显存友好：释放 run 级变量
                        del latents_out, latents_final, images

        _log("Done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()