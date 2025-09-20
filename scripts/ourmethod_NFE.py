#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diverse SD3.5 批量版（多 method × 多 steps × 多 seeds；固定 guidance & prompt）

需求实现：
- 支持多个 methods、多个 steps、多个 seeds 的笛卡尔积批量生成；
- guidance 与 prompt 固定；
- 输出目录结构：
    outputs/{method}_NFE/{eval,imgs}/
    其中 imgs 下按 seed 与 steps 建子目录：
    outputs/{method}_NFE/imgs/seed{seed}_steps{steps}/img_***.png
- 在 eval/ 下维护 summary.csv 与 summary.jsonl，记录每次 run 的元信息。

相对原脚本的改动：
- 新增 CLI：--methods、--steps-list、--seeds-list；保留 G/width/height 等；
- 管线（模型/设备/CLIP）初始化一次，之后循环不同 steps/seed 构建 cfg 与 vol 并生成；
- 输出目录改为 {method}_NFE 固定命名，并按 seed/steps 细分；
- 封装 run_one(...)，便于重复调用；
- 其余生成逻辑、正交注入、体积目标、噪声注入等保持不变。

注意：若你的 “method” 还会影响算法分支，请在 run_one 内部根据 method 做相应切换；
当前脚本仅将 method 用于目录命名（与原代码一致）。
"""

import os
import re
import sys
import time
import json
import csv
import argparse
import traceback
from typing import Any, Dict, Optional, List, Tuple

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


def _ensure_dirs(base_outputs: str, method: str, seed: int, steps: int) -> Tuple[str, str, str]:
    """返回 (root, eval_dir, imgs_subdir)"""
    root = os.path.join(base_outputs, f"{method}_NFE")
    eval_dir = os.path.join(root, "eval")
    imgs_dir = os.path.join(root, "imgs")
    imgs_sub = os.path.join(imgs_dir, f"seed{seed}_steps{steps}")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(imgs_sub, exist_ok=True)
    return root, eval_dir, imgs_sub


def _append_log(eval_dir: str, record: Dict[str, Any]):
    csv_path = os.path.join(eval_dir, "summary.csv")
    jsonl_path = os.path.join(eval_dir, "summary.jsonl")
    fieldnames = ["timestamp", "method", "steps", "seed", "guidance", "prompt", "negative", "G", "height", "width", "out_dir", "images"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in record.items()}
        w.writerow(row)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Diverse SD3.5 批量版（多 method × 多 steps × 多 seeds）'
    )

    # 固定项（一次指定，全局生效）
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=64)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--guidance', type=float, default=3.0)

    # 批量项（列表）
    ap.add_argument('--methods', type=str, required=True, help="逗号分隔，例如 'noise,ddim'。仅用于命名，若算法有分支请在代码里接入。")
    ap.add_argument('--steps-list', type=str, required=True, help="逗号分隔，例如 '20,30,50'")
    ap.add_argument('--seeds-list', type=str, required=True, help="逗号分隔，例如 '0,42,1234'")

    # 本地模型路径
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

    # 多样性/噪声等（保持原有参数）
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--t-gate', type=str, default='0.85,0.95')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2', 't1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early', 'late'])
    ap.add_argument('--eta-sde', type=float, default=1.0)
    ap.add_argument('--rho0', type=float, default=0.25)
    ap.add_argument('--rho1', type=float, default=0.05)
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)
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

    # 输出根目录（固定结构在此根目录下）
    ap.add_argument('--outputs', type=str, default=None, help='默认在 <project_root>/outputs')
    return ap.parse_args()


# -------------------- main & core --------------------

def main():
    args = parse_args()

    # ===== sys.path 注入 =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    outputs_root = args.outputs.strip() if (args.outputs and len(args.outputs.strip())>0) else os.path.join(project_root, 'outputs')
    os.makedirs(outputs_root, exist_ok=True)

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

        # ===== 1) CPU 加载，再手动上卡（只做一次） =====
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

        # ===== 2) CLIP（一次） =====
        _log("Loading CLIP ...", args.debug)
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
        )
        _log("CLIP ready.", args.debug)

        # ===== 3) 解析批量参数 =====
        methods: List[str] = [m.strip() for m in args.methods.split(',') if m.strip()]
        steps_list: List[int] = [int(s.strip()) for s in args.steps_list.split(',')]  # type: ignore
        seeds_list: List[int] = [int(s.strip()) for s in args.seeds_list.split(',')]  # type: ignore

        # ===== 4) 主循环 =====
        for method in methods:
            for steps in steps_list:
                for seed in seeds_list:
                    _run_one(
                        pipe=pipe,
                        clip=clip,
                        args=args,
                        method=method,
                        steps=steps,
                        seed=seed,
                        outputs_root=outputs_root,
                        dev_tr=dev_tr,
                        dev_vae=dev_vae,
                        dev_clip=dev_clip,
                        dtype=dtype,
                    )

        _log("All runs finished.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


# -------------------- run-one (核心生成，基本沿用原逻辑) --------------------

def _run_one(pipe, clip, args, method: str, steps: int, seed: int, outputs_root: str, dev_tr, dev_vae, dev_clip, dtype):
    from torch.utils.checkpoint import checkpoint
    from diverse_flow.config import DiversityConfig
    from diverse_flow.volume_objective import VolumeObjective
    from diverse_flow.utils import project_partial_orth, batched_norm as _bn
    from diverse_flow.utils import sched_factor as time_sched_factor

    debug = args.debug

    # ===== 输出目录（本次组合） =====
    root, eval_dir, imgs_sub = _ensure_dirs(outputs_root, method, seed, steps)
    _log(f"Output (method={method}, steps={steps}, seed={seed}): {imgs_sub}", True)

    # ===== 每次 steps 变化需要重建 cfg/vol =====
    t0, t1 = args.t_gate.split(',')
    cfg = DiversityConfig(
        num_steps=int(steps), tau=args.tau, eps_logdet=args.eps_logdet,
        gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
        partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
        sched_shape=args.sched_shape, clip_image_size=224,
        leverage_alpha=0.5,
    )
    vol = VolumeObjective(clip, cfg)

    # ===== 运行状态 =====
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
        Q, _ = torch.linalg.qr(y.t(), mode='reduced')  # [M, B]
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

    def diversity_callback(ppl, i, t, kw):
        from diverse_flow.utils import sched_factor as time_sched_factor
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
                imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

            # —— CLIP 卡上求体积损失对“像素”的梯度 ——
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

            from diverse_flow.utils import project_partial_orth, batched_norm as _bn
            g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
            div_disp = g_proj * dt_unit

            # 噪声项（A+B）
            def _bn(x):
                return torch.sqrt(torch.clamp(x.flatten(1).pow(2).sum(dim=1), min=1e-12))

            brown_std = _brownian_std_from_scheduler(pipe, i)
            eta = float(args.eta_sde)
            base_brown = eta * brown_std
            if args.noise_timing == 'early':
                rho_t = args.rho0 * t_norm + args.rho1 * (1.0 - t_norm)
            else:
                rho_t = args.rho1 * t_norm + args.rho0 * (1.0 - t_norm)

            base_disp = (v_est * dt_unit) if v_est is not None else z
            base_norm = _bn(base_disp)
            target_brown = torch.full_like(base_norm, fill_value=max(base_brown, 0.0))
            target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
            target = torch.minimum(target_brown, target_snr)

            xi = torch.randn_like(g_proj)
            vnorm = _bn(v_est) if v_est is not None else None
            if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(args.vnorm_threshold)):
                xi_eff = xi
            else:
                from diverse_flow.utils import project_partial_orth
                xi_eff = project_partial_orth(xi, v_est, float(args.noise_partial_ortho))

            xi_norm = _bn(xi_eff)
            noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)

            disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
            div_raw   = _bn(div_disp)
            scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

            delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp

            delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
            lat_new[s:e] = lat_new[s:e] + delta_tr

            if "ctrl_cache" not in state:
                state["ctrl_cache"] = []
            state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

            if args.debug and (s == 0):
                with torch.no_grad():
                    def _bn(x):
                        return torch.sqrt(torch.clamp(x.flatten(1).pow(2).sum(dim=1), min=1e-12))
                    n_div   = _bn(div_disp).mean().item()
                    n_noise = _bn(noise_disp).mean().item()
                    n_base  = _bn(base_disp).mean().item()
                    print(f"[t={t_norm:.2f}] |base|={n_base:.4f} |div|={n_div:.4f} |noise|={n_noise:.4f} "
                          f"gamma={gamma_sched:.4f} brown={brown_std:.4f} rho_t={rho_t:.3f}")

            if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
            if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
            del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

        kw["latents"] = lat_new

        state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
        state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
        if "ctrl_cache" in state:
            state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
            del state["ctrl_cache"]
        state["prev_dt_unit"] = dt_unit
        return kw

    # ===== 生成 latent =====
    generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
    generator.manual_seed(int(seed))

    _log("Start pipeline() ...", debug)
    latents_out = pipe(
        prompt=args.prompt,
        negative_prompt=(args.negative if args.negative else None),
        height=args.height, width=args.width,
        num_images_per_prompt=args.G,
        num_inference_steps=int(steps),
        guidance_scale=args.guidance,
        generator=generator,
        callback_on_step_end=diversity_callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="latent",
        return_dict=False,
    )[0]

    # ===== decode 最终 latent =====
    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
    latents_final = latents_out.to(dev_vae, non_blocking=True)

    if getattr(args, 'enable_vae_tiling', False):
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    from torch.utils.checkpoint import checkpoint
    with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
        images = checkpoint(
            lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
            latents_final, use_reentrant=False
        )

    from torchvision.utils import save_image
    img_paths = []
    for i in range(images.size(0)):
        p = os.path.join(imgs_sub, f"img_{i:03d}.png")
        save_image(images[i].cpu(), p)
        img_paths.append(p)

    # ===== 记录日志 =====
    _append_log(
        eval_dir,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "steps": int(steps),
            "seed": int(seed),
            "guidance": float(args.guidance),
            "prompt": args.prompt,
            "negative": args.negative,
            "G": int(args.G),
            "height": int(args.height),
            "width": int(args.width),
            "out_dir": imgs_sub,
            "images": img_paths,
        },
    )

    _log(f"Done. Saved {len(img_paths)} images to {imgs_sub}", True)


if __name__ == "__main__":
    main()
