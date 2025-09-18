#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation Study for SD3.5 + CLIP (early-noise, colored dots friendly)

Methods:
  - wo_OP     : remove orthogonal projection (lambda_op=0.0 & partial_ortho=0, OP 路径直接跳过)
  - wo_LR     : remove leverage (leverage_alpha=0.0)
  - ourmethod : baseline（保留 OP 与 LR）

输出目录（与评测脚本兼容）：
outputs/ablation/<method>/{imgs,eval}/
  imgs/<prompt_slug>_seed{seed}_g{guidance}_s{steps}/img_000.png ...

默认参数偏激进，用于放大失真现象（车头错位、尺寸波动等）：
  --gamma0 0.12 --t-gate 0.70,0.99 --partial-ortho 1.0
  --noise-timing early --noise-scale 0.9
  --steps 30 --guidance 3.0
"""

import os
import re
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
        if not hasattr(pipe, name): continue
        obj = getattr(pipe, name)
        if obj is None: report[name] = "None"; continue
        if isinstance(obj, nn.Module):
            dev = _first_device_of_module(obj)
            report[name] = str(dev) if dev is not None else "module(no params)"
        else:
            report[name] = "non-module"
    print("[pipe-devices]", report, flush=True)

def assert_on(m, want):
    if not isinstance(m, nn.Module): return
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
    if os.path.isfile(os.path.join(p, 'model_index.json')): return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files: return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Ablation Study (wo_OP / wo_LR / ourmethod) [early-noise]')

    # 生成
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=20)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])

    ap.add_argument('--methods', nargs='+', default=['wo_OP','wo_LR','ourmethod'])

    # 模型与设备
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae',         type=str, default='cuda:0')
    ap.add_argument('--device-clip',        type=str, default='cuda:0')
    ap.add_argument('--device-text1',       type=str, default='cuda:1')
    ap.add_argument('--device-text2',       type=str, default='cuda:1')
    ap.add_argument('--device-text3',       type=str, default='cuda:1')
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')

    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=1.0)
    ap.add_argument('--t-gate', type=str, default='0.70,0.99')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    ap.add_argument('--noise-timing', type=str, default='early', choices=['early','late'])
    ap.add_argument('--noise-scale', type=float, default=0.9)

    ap.add_argument('--lambda-op-default', type=float, default=0.95)
    ap.add_argument('--leverage-alpha-default', type=float, default=0.6)

    return ap.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()

    # sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # TE 默认与 Transformer 同卡
    if args.device_text1 is None: args.device_text1 = args.device_transformer
    if args.device_text2 is None: args.device_text2 = args.device_transformer
    if args.device_text3 is None: args.device_text3 = args.device_transformer

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

        # ---- 设备 + dtype
        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_te1 = torch.device(args.device_text1)
        dev_te2 = torch.device(args.device_text2)
        dev_te3 = torch.device(args.device_text3)
        dev_clip= torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, text1={dev_te1}, text2={dev_te2}, text3={dev_te3}, clip={dev_clip}", args.debug)
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

        # ---- Pipeline
        model_dir = _resolve_model_dir(args.model_dir)
        pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
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
        inspect_pipe_devices(pipe)
        if hasattr(pipe, "transformer"):    assert_on(pipe.transformer, dev_tr)
        if hasattr(pipe, "text_encoder"):   assert_on(pipe.text_encoder,   dev_tr)
        if hasattr(pipe, "text_encoder_2"): assert_on(pipe.text_encoder_2, dev_tr)
        if hasattr(pipe, "text_encoder_3"): assert_on(pipe.text_encoder_3, dev_tr)
        if hasattr(pipe, "vae"):            assert_on(pipe.vae,            dev_vae)

        # ---- CLIP & Volume
        clip = CLIPWrapper(
            impl="openai_clip", arch="ViT-B-32",
            jit_path=args.clip_jit, checkpoint_path=None,
            device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
        )

        # 噪声时间调度（早/晚）
        def _beta_time_sched(t_norm: float, eps: float = 1e-2) -> float:
            t = float(t_norm)
            if args.noise_timing == 'early':
                base = min(1.0, t / (1.0 - t + eps))
            else:
                base = min(1.0, (1.0 - t) / (t + eps))
            return float(args.noise_scale) * float(base)

        # OP 应用：lambda_op 控制去除平行分量强度；partial_ortho 作为混合权重
        def _apply_op(g: torch.Tensor, v: Optional[torch.Tensor], lambda_op: float, alpha: float) -> torch.Tensor:
            if (v is None) or (lambda_op <= 0.0) or (alpha <= 0.0):
                return g
            dot = (g * v).sum(dim=(1,2,3), keepdim=True)
            vv  = (v * v).sum(dim=(1,2,3), keepdim=True) + 1e-12
            g_para = (dot / vv) * v
            g_op = g - lambda_op * g_para
            return (1.0 - alpha) * g + alpha * g_op

        # 输出根目录
        prompt_slug = _slugify(args.prompt)
        base_root = os.path.join(project_root, "outputs", "ablation")

        # ---- 逐方法运行
        for method in args.methods:
            method = method.strip()
            method_dir = os.path.join(base_root, method)
            imgs_root  = os.path.join(method_dir, "imgs")
            eval_dir   = os.path.join(method_dir, "eval")
            os.makedirs(imgs_root, exist_ok=True)
            os.makedirs(eval_dir,  exist_ok=True)

            # per-method overrides
            if method == "wo_OP":
                lambda_op_use = 0.0          # 彻底关闭
                partial_ortho_use = 0.0      # 不参与 OP
                leverage_alpha_use = float(args.leverage_alpha_default)
                def do_op(g, v):  # 保证跳过
                    return g
            elif method == "wo_LR":
                lambda_op_use = float(args.lambda_op_default)
                partial_ortho_use = float(args.partial_ortho)
                leverage_alpha_use = 0.0     # 彻底关闭 LR
                def do_op(g, v):
                    return _apply_op(g, v, lambda_op_use, partial_ortho_use)
            else:  # ourmethod
                lambda_op_use = float(args.lambda_op_default)
                partial_ortho_use = float(args.partial_ortho)
                leverage_alpha_use = float(args.leverage_alpha_default)
                def do_op(g, v):
                    return _apply_op(g, v, lambda_op_use, partial_ortho_use)

            # VolumeObjective（按方法重建，确保 LR 生效/禁用）
            t0, t1 = [float(x) for x in args.t_gate.split(',')]
            cfg = DiversityConfig(
                num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
                gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
                partial_ortho=partial_ortho_use, t_gate=(t0, t1),
                sched_shape=args.sched_shape, clip_image_size=224,
                leverage_alpha=leverage_alpha_use,
            )
            vol = VolumeObjective(clip, cfg)

            _log(f"[ablation] method={method} | lambda_op={lambda_op_use} | partial_ortho={partial_ortho_use} | leverage_alpha={leverage_alpha_use}", True)

            for sd in args.seeds:
                run_dir = os.path.join(
                    imgs_root,
                    f"{prompt_slug}_seed{int(sd)}_g{float(args.guidance)}_s{int(args.steps)}"
                )
                os.makedirs(run_dir, exist_ok=True)
                _log(f"[RUN] {method} | seed={sd} -> {run_dir}", True)

                # per-run state
                state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_latents_vae_cpu": None,
                    "prev_ctrl_vae_cpu":   None,
                    "prev_dt_unit":        None,
                    "prev_prev_latents_vae_cpu": None,
                    "last_logdet":         None,
                }

                def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
                    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                    out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
                    return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

                def diversity_callback(ppl, i, t, kw):
                    ts = ppl.scheduler.timesteps
                    t_cur  = float(ts[i].item())
                    t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                    t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                    t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
                    dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

                    lat = kw.get("latents")
                    if lat is None: return kw

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

                        # decode -> grad (image) -> VJP
                        with torch.enable_grad(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                            imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)
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

                        # v_est
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

                        # 确定性项：方法化的 OP 路径（wo_OP 直接跳过）
                        g_det = do_op(grad_lat, v_est)
                        div_disp = g_det * dt_unit

                        # 噪声：白噪声 + 完全正交到基流；不乘 γ；前期（默认）更强
                        beta = _beta_time_sched(t_norm)
                        if (beta > 0.0) and (v_est is not None):
                            from diverse_flow.utils import project_partial_orth as ppo
                            xi = torch.randn_like(g_det)
                            xi = ppo(xi, v_est, 1.0)
                            noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi
                        else:
                            noise_disp = torch.zeros_like(g_det)

                        base_disp = (v_est * dt_unit) if v_est is not None else z
                        disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                        div_raw   = _bn(div_disp)
                        scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                        delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp

                        delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                        lat_new[s:e] = lat_new[s:e] + delta_tr

                        if "ctrl_cache" not in state: state["ctrl_cache"] = []
                        state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                        if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                        if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                        del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_det, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

                    kw["latents"] = lat_new
                    state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
                    state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
                    if "ctrl_cache" in state:
                        state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
                        del state["ctrl_cache"]
                    state["prev_dt_unit"] = dt_unit
                    return kw

                # run one
                generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                generator.manual_seed(int(sd))

                out = pipe(
                    prompt=args.prompt,
                    negative_prompt=(args.negative if args.negative else None),
                    height=args.height, width=args.width,
                    num_images_per_prompt=args.G,
                    num_inference_steps=args.steps,
                    guidance_scale=float(args.guidance),
                    generator=generator,
                    callback_on_step_end=diversity_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output_type="latent",
                    return_dict=False,
                )[0]

                # decode
                sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                latents_final = out.to(dev_vae, non_blocking=True)
                with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    images = checkpoint(
                        lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                        latents_final, use_reentrant=False
                    )
                from torchvision.utils import save_image
                for i_img in range(images.size(0)):
                    save_image(images[i_img].cpu(), os.path.join(run_dir, f"img_{i_img:03d}.png"))

                del out, latents_final, images

        _log("Ablation study (early-noise) finished.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()