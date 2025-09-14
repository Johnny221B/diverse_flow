# -*- coding: utf-8 -*-
"""
SD3.5 + CLIP feature-driven diversity (stabilized)
- 训练自由、推理时控制：体积漂移(γ) + 质量保真的正交噪声(β)
- 结构稳控：前段全正交 + 硬上限 ||g_proj|| ≤ ||v_est|| + 早期位移帽
- 细节多样：末端噪声门控 + AR(1) 时间相关 + 批内正交 + 中带增强
- 可选：Heun 校正 + 最后两步 fp32 解码（拉 FID/PR precision）
用法与原脚本相同；新增若干参数见 parse_args。
"""
import os, sys, argparse, traceback, time, re
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils: device & logging ----------
def _gb(x): return f"{x/1024**3:.2f} GB"
def print_mem_all(tag: str, devices: list):
    import torch
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
    if not isinstance(m, nn.Module): return None
    for p in m.parameters(recurse=False): return p.device
    for b in m.buffers(recurse=False):    return b.device
    for sm in m.children():
        for p in sm.parameters(recurse=False): return p.device
        for b in sm.buffers(recurse=False):    return b.device
    return None

def inspect_pipe_devices(pipe):
    names = ["transformer","text_encoder","text_encoder_2","text_encoder_3","vae",
             "tokenizer","tokenizer_2","tokenizer_3","scheduler"]
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
        if str(p.device) != str(want): raise RuntimeError(f"Param on {p.device}, expected {want}")
    for b in m.buffers():
        if str(b.device) != str(want): raise RuntimeError(f"Buffer on {b.device}, expected {want}")

def _slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip()); s = re.sub(r'[^A-Za-z0-9._-]+','',s); s = re.sub(r'_{2,}','_',s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _resolve_model_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p,'model_index.json')): return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files: return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')

def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S")
    if debug: print(f"[{ts}] {s}", flush=True)

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser(description='Diverse SD3.5 (feature-driven + quality-preserving noise)')
    # generation
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=24)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1111)

    # paths
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='ourMethod')

    # method hyperparams
    ap.add_argument('--gamma0', type=float, default=0.075)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.25)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--full-ortho-until', type=float, default=0.80, help='t_norm < this → use 1.0 orthogonality')
    ap.add_argument('--t-gate', type=str, default='0.80,0.99', help='for deterministic volume drift γ(t)')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-4)

    # noise scheduling (new)
    ap.add_argument('--t-gate-noise', type=str, default='0.90,0.995', help='noise only near the end')
    ap.add_argument('--noise-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--noise-rho', type=float, default=0.75, help='AR(1) correlation, 0~0.6')
    ap.add_argument('--midband-lambda', type=float, default=0.25, help='boost mid-band in g_proj (0~0.4)')
    ap.add_argument('--early-cap-power', type=float, default=1.0, help='extra early cap on disp, 0.5~1.5')
    ap.add_argument('--noise-lowpass-k', type=int, default=3,
                help='Low-pass kernel size for noise shaping (odd int, 1 disables)')

    # devices
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae',         type=str, default='cuda:0')
    ap.add_argument('--device-clip',        type=str, default='cuda:0')
    ap.add_argument('--device-text1',       type=str, default='cuda:1')
    ap.add_argument('--device-text2',       type=str, default='cuda:1')
    ap.add_argument('--device-text3',       type=str, default='cuda:1')

    # scheduler/precision niceties
    ap.add_argument('--use-heun', action='store_true', help='swap to Heun scheduler')
    ap.add_argument('--fp32-last-decode', action='store_true', help='decode final latents in fp32')

    # memory & debug
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()

# ---------- main ----------
def main():
    args = parse_args()

    # import project modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)

    # default TE devices
    if args.device_text1 is None: args.device_text1 = args.device_transformer
    if args.device_text2 is None: args.device_text2 = args.device_transformer
    if args.device_text3 is None: args.device_text3 = args.device_transformer

    try:
        import torch, torch.backends.cudnn as cudnn
        from torch.utils.checkpoint import checkpoint
        from diffusers import StableDiffusion3Pipeline
        from torchvision.utils import save_image

        from diverse_flow.config import DiversityConfig
        from diverse_flow.clip_wrapper import CLIPWrapper
        from diverse_flow.volume_objective import VolumeObjective
        from diverse_flow.utils import project_partial_orth, batched_norm as _bn
        from diverse_flow.utils import sched_factor as time_sched_factor
        import torch

        # devices & dtype
        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_te1 = torch.device(args.device_text1)
        dev_te2 = torch.device(args.device_text2)
        dev_te3 = torch.device(args.device_text3)
        dev_clip= torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: TF={dev_tr}, VAE={dev_vae}, TE=[{dev_te1},{dev_te2},{dev_te3}], CLIP={dev_clip}", args.debug)
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

        # 1) load pipeline on CPU → move modules to target devices
        model_dir = _resolve_model_dir(args.model_dir)
        pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")

        # optional: swap to Heun
        if args.use_heun:
            try:
                from diffusers import FlowMatchHeunDiscreteScheduler
                pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(pipe.scheduler.config)
                _log("Switched scheduler to Heun.", True)
            except Exception as e:
                _log(f"[warn] Heun scheduler not available: {e}", True)

        # move parts
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

        # 2) CLIP & Volume objective
        clip = CLIPWrapper(impl="openai_clip", arch="ViT-B-32",
                           jit_path=args.clip_jit, checkpoint_path=None,
                           device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"))
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=tuple(map(float, args.t_gate.split(','))),
            sched_shape=args.sched_shape, clip_image_size=224,
            leverage_alpha=0.7,
        )
        vol = VolumeObjective(clip, cfg)

        # 3) callback state
        state = {"prev_latents_vae_cpu": None, "prev_ctrl_vae_cpu": None, "prev_dt_unit": None,
                 "last_logdet": None, "xi_prev": None, "xi_prev_cpu": None}

        # helpers
        def _vae_decode_pixels(z):
            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

        def _beta_monotone(t_norm: float, eps: float = 1e-2) -> float:
            return float(min(1.0, t_norm / (1.0 - t_norm + eps))) * 0.5  # 0.5 为你目前设置

        def _lowpass(x, k=3):
            pad = k // 2
            w = torch.ones(x.size(1), 1, k, k, device=x.device, dtype=x.dtype) / (k*k)
            return F.conv2d(x, w, padding=pad, groups=x.size(1))

        def _bandpass_mid(x):
            return _lowpass(x, k=7) - _lowpass(x, k=15)

        # 4) main callback
        t0n, t1n = map(float, args.t_gate_noise.split(','))
        def diversity_callback(ppl, i, t, kw):
            ts = ppl.scheduler.timesteps
            t_cur  = float(ts[i].item())
            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
            t_norm  = (t_cur - t_min) / (t_max - t_min + 1e-8)
            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

            lat = kw.get("latents")
            if lat is None: return kw

            gamma_sched = cfg.gamma0 * time_sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
            if gamma_sched <= 0:
                state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
                state["prev_dt_unit"] = dt_unit
                return kw

            lat_new = lat.clone()
            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
            B = lat_vae_full.size(0); chunk = 2 if B >= 2 else 1
            prev_cpu = state.get("prev_latents_vae_cpu", None)
            
            if state.get("xi_prev_cpu", None) is None:
                state["xi_prev_cpu"] = torch.zeros_like(lat_vae_full, dtype=torch.float32, device="cpu")

            with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                for s in range(0, B, chunk):
                    e = min(B, s+chunk)
                    z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                    # decode to pixels
                    with torch.enable_grad():
                        # 提醒：确保 z 是叶子且需要梯度
                        if not z.requires_grad:
                            z.requires_grad_(True)

                        # 建议首次先 *不使用 checkpoint*，直接前向，避免断图
                        imgs_chunk = _vae_decode_pixels(z)      # [bs,3,H,W] in [0,1]
                        assert imgs_chunk.requires_grad, "VAE decode returned tensor without grad_fn (graph detached)."

                    # —— CLIP 卡上求体积损失对“像素”的梯度 ——
                    imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                    _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                    current_logdet = float(_logs.get("logdet", 0.0))

                    # 能量单调守门
                    last_logdet = state.get("last_logdet", None)
                    if (last_logdet is not None) and (current_logdet < last_logdet):
                        gamma_sched = 0.5 * gamma_sched
                    state["last_logdet"] = current_logdet

                    # 把像素梯度搬回 VAE 卡并对齐 dtype
                    grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                    # —— VJP 回 latent —— #
                    grad_lat = torch.autograd.grad(
                        outputs=imgs_chunk,
                        inputs=z,
                        grad_outputs=grad_img_vae,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,                 # 先放宽，下面兜底
                    )[0]

                    # 兜底：若出现未用到或断图（grad_lat is None），再用“无 clamp + 全 FP32”重算一遍
                    if grad_lat is None:
                        with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                            z32 = z.detach().float().requires_grad_(True)
                            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                            # 线性范围（不 clamp），避免饱和导致的 0 梯度
                            dec = pipe.vae.decode(z32 / sf, return_dict=False)[0]   # [-1,1]
                            imgs_lin = (dec + 1.0) / 2.0                            # [0,1] 无 clamp
                            go = grad_img_vae.float()
                            grad_lat_32 = torch.autograd.grad(
                                outputs=imgs_lin, inputs=z32, grad_outputs=go,
                                retain_graph=False, create_graph=False, allow_unused=False
                            )[0]
                        grad_lat = grad_lat_32.to(z.dtype)

                    # estimate base velocity
                    v_est = None
                    if prev_cpu is not None:
                        total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)
                        prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                        prev_dt   = state.get("prev_dt_unit", None)
                        if (prev_ctrl is not None) and (prev_dt is not None):
                            base_move_prev = total_diff - prev_ctrl[s:e].to(dev_vae, non_blocking=True)
                            v_est = base_move_prev / max(prev_dt, 1e-8)
                        else:
                            v_est = total_diff / max(dt_unit, 1e-8)

                    # projection (full ortho early)
                    if v_est is not None:
                        po = 1.0 if t_norm < args.full_ortho_until else cfg.partial_ortho
                        g_proj = project_partial_orth(grad_lat, v_est, po)
                        # mid-band boost
                        if args.midband_lambda > 0:
                            g_proj = g_proj + args.midband_lambda * _bandpass_mid(g_proj)
                        # per-sample hard cap: ||g_proj|| ≤ ||v_est||
                        vnorm = _bn(v_est); gnorm = _bn(g_proj)
                        scale_g = torch.minimum(torch.ones_like(vnorm), vnorm / (gnorm + 1e-12))
                        g_proj = g_proj * scale_g.view(-1,1,1,1)
                    else:
                        g_proj = grad_lat

                    # deterministic displacement
                    div_disp = g_proj * dt_unit

                    # directional noise (end-gated + AR(1) + batch-orth)
                    beta = _beta_monotone(t_norm, eps=1e-2)
                    if (beta > 0.0) and (v_est is not None):
                        # 取出上一时刻噪声（按当前分块），若首次则为 0
                        xi_prev = state["xi_prev_cpu"][s:e].to(dev_vae, non_blocking=True, dtype=g_proj.dtype)

                        # 新噪声基：正交到基流 + 可选低通 + 标准化
                        eps = torch.randn_like(g_proj)
                        eps = project_partial_orth(eps, v_est, 1.0)  # 完全正交于基流
                        if getattr(args, "noise_lowpass_k", 3) and args.noise_lowpass_k > 1:
                            eps = _lowpass(eps, k=int(args.noise_lowpass_k))
                        eps = eps - eps.mean(dim=(1,2,3), keepdim=True)
                        eps = eps / (_bn(eps).view(-1,1,1,1) + 1e-12)

                        # AR(1) 相关噪声：xi_t = rho * xi_{t-1} + sqrt(1-rho^2) * eps
                        rho = float(getattr(args, "noise_rho", 0.9))
                        rho = max(0.0, min(0.999, rho))
                        xi = rho * xi_prev + (1.0 - rho**2) ** 0.5 * eps

                        # 位移形态噪声（不乘 γ）
                        noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi

                        # 写回缓存到 CPU，供下一步使用
                        state["xi_prev_cpu"][s:e] = xi.detach().to("cpu", dtype=torch.float32)
                    else:
                        noise_disp = torch.zeros_like(g_proj)

                    # trust region on deterministic part + early cap
                    base_disp = (v_est * dt_unit) if v_est is not None else z
                    disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                    if args.early_cap_power != 1.0:
                        disp_cap = disp_cap * ((1.0 - t_norm) ** args.early_cap_power)
                    div_raw   = _bn(div_disp)
                    scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                    # write back
                    delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp
                    delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                    lat_new[s:e] = lat_new[s:e] + delta_tr

                    # cache control for next-step base estimation
                    if "ctrl_cache" not in state: state["ctrl_cache"] = []
                    state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

                    if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
                    if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
                    del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk, delta_tr, v_est, z

            kw["latents"] = lat_new
            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
            if "ctrl_cache" in state:
                state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
                del state["ctrl_cache"]
            state["prev_dt_unit"] = dt_unit
            return kw

        # 5) outputs
        prompt_slug = _slugify(args.prompt)
        outputs_root = os.path.join(project_root, 'outputs')
        base_out_dir = args.out if (args.out and len(args.out.strip())>0) else os.path.join(outputs_root, f"{args.method}_{prompt_slug or 'no_prompt'}")
        out_dir = os.path.join(base_out_dir, "imgs")
        os.makedirs(out_dir, exist_ok=True)
        _log(f"Output dir: {out_dir}", True)

        # 6) run pipeline (latent output)
        generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
        generator.manual_seed(int(args.seed))

        latents_out = pipe(
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

        # 7) final decode (optionally fp32 to reduce artifacts / improve FID)
        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
        latents_final = latents_out.to(dev_vae, non_blocking=True)
        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        # with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
        #     if args.fp32-last-decode:   # NOTE: CLI flag uses hyphen; python var must use underscore
        #         # Python 变量名不能有 -，所以这里兼容处理：
        #         pass
        # 修正：将 flag 映射到本地变量
        fp32_decode = bool(getattr(args, 'fp32_last_decode', False))

        with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            if fp32_decode:
                images = checkpoint(lambda z: (pipe.vae.decode(z.float() / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                                    latents_final.float(), use_reentrant=False)
            else:
                images = checkpoint(lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                                    latents_final, use_reentrant=False)

        for i in range(images.size(0)):
            save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

        _log("Done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
