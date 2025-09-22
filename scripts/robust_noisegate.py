import os
import re
import sys
import time
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


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Robust Noise-Gate sweep for Diverse SD3.5 (no-embeds)'
    )

    # 基本生成参数（guidance/steps 在脚本中固定）
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=20)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)

    # seeds: 逗号分隔
    ap.add_argument('--seeds', type=str, default='1111,2222,3333,4444',
                    help='多个种子，逗号分隔，如 1111,2222,3333')

    # 噪声门控列表：形如 "0.0:0.95,0.5:0.95,0.6:0.95,..."
    ap.add_argument('--noise-gates', type=str,
                    default='0.0:0.95,0.5:0.95,0.6:0.95,0.7:0.95,0.8:0.95,0.9:0.95')

    # 本地模型路径
    ap.add_argument('--model-dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))

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

def _parse_noise_gates(spec: str) -> List[Tuple[float, float]]:
    pairs = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' not in tok:
            raise ValueError(f"Bad noise-gates token: {tok}. 期望形如 '0.6:0.95'")
        a, b = tok.split(':', 1)
        pairs.append((float(a), float(b)))
    return pairs


def main():
    args = parse_args()

    # ===== sys.path 注入（让脚本能 import diverse_flow） =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 文本编码器默认与 transformer 同卡
    if args.device_text1 is None:
        args.device_text1 = args.device_transformer
    if args.device_text2 is None:
        args.device_text2 = args.device_transformer
    if args.device_text3 is None:
        args.device_text3 = args.device_transformer

    # 固定 guidance/steps
    FIXED_GUIDANCE = 3.0
    FIXED_STEPS = 30

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

        # ===== 3) 固定的多样性配置（除 t_gate 外）=====
        base_cfg_kwargs = dict(
            num_steps=FIXED_STEPS, tau=1.0, eps_logdet=1e-3,
            gamma0=0.12, gamma_max_ratio=0.3,
            partial_ortho=0.95,  # 体积力对基流的正交比例
            sched_shape='sin2', clip_image_size=224,
            leverage_alpha=0.5,
        )

        # ===== 4) 状态容器与回调（与单次调用相同） =====
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

        from diverse_flow.utils import batched_norm as _bn
        from diverse_flow.utils import project_partial_orth
        from diverse_flow.utils import sched_factor as time_sched_factor

        def diversity_callback_builder(cfg):
            # 返回一个闭包，绑定不同的 cfg（主要是不同 t_gate）
            def diversity_callback(ppl, i, t, kw):
                ts = ppl.scheduler.timesteps
                t_cur  = float(ts[i].item())
                t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
                dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

                # ---------- 第 0 步：安全混入批内正交噪声（保持与原脚本一致） ----------
                if i == 0 and "latents" in kw and kw["latents"] is not None:
                    lat0 = kw["latents"]
                    # 轻量的批内正交注入（可按需调强度/模式，这里沿用默认）
                    def _make_orthonormal_noise_like(x: torch.Tensor) -> torch.Tensor:
                        B = x.size(0)
                        M = x[0].numel()
                        y = torch.randn(B, M, device=x.device, dtype=torch.float32)
                        Q, _ = torch.linalg.qr(y.t(), mode='reduced')  # [M,B]
                        return Q.t().to(dtype=x.dtype).contiguous().view_as(x)

                    def _inject_init_ortho(latents: torch.Tensor, strength: float = 0.2, mode: str = 'blend') -> torch.Tensor:
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

                    kw["latents"] = _inject_init_ortho(lat0, strength=0.2, mode='blend')

                lat = kw.get("latents")
                if lat is None:
                    return kw

                gamma_sched = base_cfg_kwargs["gamma0"] * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
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

                    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                        imgs_chunk = torch.utils.checkpoint.checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

                    imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                    _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                    current_logdet = float(_logs.get("logdet", 0.0))

                    last_logdet = state.get("last_logdet", None)
                    if (last_logdet is not None) and (current_logdet < last_logdet):
                        gamma_sched_local = 0.5 * gamma_sched
                    else:
                        gamma_sched_local = gamma_sched
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

                    g_proj = project_partial_orth(grad_lat, v_est, base_cfg_kwargs["partial_ortho"]) if v_est is not None else grad_lat
                    div_disp = g_proj * dt_unit

                    brown_std = _brownian_std_from_scheduler(ppl, i)
                    eta = 1.0

                    # 这里保持原“relative SNR 目标”的实现（使用 rho0/rho1 默认值）
                    rho0, rho1 = 0.25, 0.05
                    # —— 采用 early 策略（与原脚本一致）
                    rho_t = rho0 * t_norm + rho1 * (1.0 - t_norm)

                    base_disp = (v_est * dt_unit) if v_est is not None else z
                    base_norm = _bn(base_disp)

                    target_brown = torch.full_like(base_norm, fill_value=max(eta * brown_std, 0.0))
                    target_snr   = torch.clamp(base_norm * max(rho_t, 0.0), min=0.0)
                    target = torch.minimum(target_brown, target_snr)

                    xi = torch.randn_like(g_proj)

                    vnorm = _bn(v_est) if v_est is not None else None
                    if (v_est is None) or (vnorm is None) or (float(vnorm.mean().item()) < float(1e-4)):
                        xi_eff = xi
                    else:
                        xi_eff = project_partial_orth(xi, v_est, float(1.0))

                    xi_norm = _bn(xi_eff)
                    noise_disp = xi_eff / (xi_norm.view(-1,1,1,1) + 1e-12) * target.view(-1,1,1,1)

                    disp_cap  = base_cfg_kwargs["gamma_max_ratio"] * _bn(base_disp)
                    div_raw   = _bn(div_disp)
                    scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                    delta_chunk = (gamma_sched_local * scale.view(-1,1,1,1)) * div_disp + noise_disp

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
                    state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
                    del state["ctrl_cache"]
                state["prev_dt_unit"] = dt_unit
                return kw

            # end diversity_callback
            return diversity_callback

        # 准备 CLIP 体积目标（每个 t_gate 共享）
        cfg_dummy = type('Cfg', (), {"t_gate": (0.85, 0.95), "sched_shape": 'sin2'})
        vol = VolumeObjective(clip, DiversityConfig(**{**base_cfg_kwargs, "t_gate": cfg_dummy.t_gate}))

        # 输出根目录
        study_root = os.path.join(project_root, 'outputs', 'add_robust_study', 'noise_gate', 'imgs')
        eval_root = os.path.join(project_root, 'outputs', 'add_robust_study', 'noise_gate', 'eval')
        os.makedirs(study_root, exist_ok=True)
        os.makedirs(eval_root, exist_ok=True)
        _log(f"Study root: {study_root}", True)

        # 解析参数
        seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
        gate_pairs = _parse_noise_gates(args.noise_gates)

        # 逐配置运行
        for seed in seeds:
            for (t0, t1) in gate_pairs:
                # 每组 t_gate 需要各自的 cfg 与回调
                cfg = DiversityConfig(**{**base_cfg_kwargs, "t_gate": (float(t0), float(t1))})
                # 若 VolumeObjective 支持动态更新配置：
                if hasattr(vol, "update_config"):
                    vol.update_config(cfg)
                callback = diversity_callback_builder(cfg)

                # 构造输出目录（图片直接写到该目录）
                leaf = f"seed{seed}_noisegate{t0:.1f}_{t1:.2f}"
                out_dir = os.path.join(study_root, leaf)
                os.makedirs(out_dir, exist_ok=True)
                _log(f"Output dir: {out_dir}", True)

                # 生成 latent（不让管线内部 decode）
                generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                generator.manual_seed(seed)

                state.clear()
                state.update({
                    "prev_latents_vae_cpu": None,
                    "prev_ctrl_vae_cpu":   None,
                    "prev_dt_unit":        None,
                    "prev_prev_latents_vae_cpu": None,
                    "last_logdet":         None,
                    "gamma_auto_done":     False,
                })

                _log(f"Start pipeline() | seed={seed} | t_gate=({t0:.2f},{t1:.2f})", True)
                latents_out = pipe(
                    prompt=args.prompt,
                    negative_prompt=(args.negative if args.negative else None),
                    height=args.height, width=args.width,
                    num_images_per_prompt=args.G,
                    num_inference_steps=FIXED_STEPS,
                    guidance_scale=FIXED_GUIDANCE,
                    generator=generator,
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output_type="latent",
                    return_dict=False,
                )[0]

                # 手动在 VAE 卡 decode 最终 latent
                sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                latents_final = latents_out.to(dev_vae, non_blocking=True)

                if args.enable_vae_tiling:
                    if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                    if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

                with torch.inference_mode(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    images = torch.utils.checkpoint.checkpoint(
                        lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                        latents_final, use_reentrant=False
                    )

                from torchvision.utils import save_image
                for i in range(images.size(0)):
                    save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

                _log(f"Done. Saved {images.size(0)} images to {out_dir}", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
