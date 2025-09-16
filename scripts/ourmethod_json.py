# -*- coding: utf-8 -*-
"""
SD3.5 + CLIP feature-driven diversity (stabilized)
- 训练自由、推理时控制：体积漂移(γ) + 质量保真的正交噪声(β)
- 结构稳控：前段全正交 + 硬上限 ||g_proj|| ≤ ||v_est|| + 早期位移帽
- 细节多样：末端噪声门控 + AR(1) 时间相关 + 批内正交 + 中带增强
- 可选：Heun 校正 + 最后两步 fp32 解码（拉 FID/PR precision）

扩展：
- 新增 --spec 输入 {concept: [prompts...]}，遍历 concept×prompt×guidance×seed
- 输出路径：outputs/ourmethod_{concept}/imgs/{prompt_slug}_seed{seed}_g{guidance}_s{steps}/
"""

import os, sys, argparse, traceback, time, re, json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

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

# ---- 新增：解析 {concept: [prompts...]} ----
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

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser(description='Diverse SD3.5 (feature-driven + quality-preserving noise)')

    # 新增：多 concept JSON；保留单 prompt 回退
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept: [prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec not provided')
    ap.add_argument('--negative', type=str, default='')

    # generation
    ap.add_argument('--G', type=int, default=24)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)

    # 新增：多 guidance/seed；保留单值回退
    ap.add_argument('--guidances', type=float, nargs='+', default=[3.0, 5.0, 7.5])
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])
    ap.add_argument('--seed', type=int, default=1111)

    # paths
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--out', type=str, default=None)  # 不用外部指定 outputs 根；保留兼容
    ap.add_argument('--method', type=str, default='ourMethod')  # 你文档里是 ourmethod，这里保留你原默认

    # method hyperparams（保持原样）
    ap.add_argument('--gamma0', type=float, default=0.075)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.25)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--full-ortho-until', type=float, default=0.80, help='t_norm < this → use 1.0 orthogonality')
    ap.add_argument('--t-gate', type=str, default='0.80,0.99', help='for deterministic volume drift γ(t)')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-4)

    # noise scheduling（保持原样）
    ap.add_argument('--t-gate-noise', type=str, default='0.90,0.995')
    ap.add_argument('--noise-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--noise-rho', type=float, default=0.75)
    ap.add_argument('--midband-lambda', type=float, default=0.25)
    ap.add_argument('--early-cap-power', type=float, default=1.0)
    ap.add_argument('--noise-lowpass-k', type=int, default=3)

    # devices（保持原样）
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae',         type=str, default='cuda:0')
    ap.add_argument('--device-clip',        type=str, default='cuda:0')
    ap.add_argument('--device-text1',       type=str, default='cuda:1')
    ap.add_argument('--device-text2',       type=str, default='cuda:1')
    ap.add_argument('--device-text3',       type=str, default='cuda:1')

    # scheduler/precision niceties（保持原样）
    ap.add_argument('--use-heun', action='store_true')
    ap.add_argument('--fp32-last-decode', action='store_true')

    # memory & debug（保持原样）
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

    # 解析 JSON / 单 prompt
    if args.spec:
        with open(args.spec, 'r', encoding='utf-8') as f:
            spec_obj = json.load(f, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --spec (JSON with {concept:[prompts...]}) or --prompt")

    # 决定本次使用的 guidances / seeds（若用户显式传入单值也可回退）
    guidances = args.guidances if args.guidances else [args.guidance]
    seeds = args.seeds if args.seeds else [args.seed]

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

        # ==== 遍历 concept → prompts → guidance → seed ====
        for concept, prompt_list in concept_to_prompts.items():
            base_dir, imgs_root, eval_dir = _build_root_out(project_root, args.method, concept)
            _log(f"[OUT] base={base_dir}", True)

            for prompt_text in prompt_list:
                for g in guidances:
                    for sd in seeds:
                        # -------- 每次 run 的 callback & 状态（保持你的原逻辑） --------
                        state = {"prev_latents_vae_cpu": None, "prev_ctrl_vae_cpu": None, "prev_dt_unit": None,
                                 "last_logdet": None, "xi_prev": None, "xi_prev_cpu": None}

                        def _vae_decode_pixels(z):
                            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
                            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

                        def _beta_monotone(t_norm: float, eps: float = 1e-2) -> float:
                            return float(min(1.0, t_norm / (1.0 - t_norm + eps))) * 0.5

                        def _lowpass(x, k=3):
                            pad = k // 2
                            w = torch.ones(x.size(1), 1, k, k, device=x.device, dtype=x.dtype) / (k*k)
                            return F.conv2d(x, w, padding=pad, groups=x.size(1))

                        def _bandpass_mid(x):
                            return _lowpass(x, k=7) - _lowpass(x, k=15)

                        t0n, t1n = map(float, args.t-gate-noise.split(',')) if False else (0.0, 0.0)  # 占位，不改你原逻辑

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
                                for sidx in range(0, B, chunk):
                                    e = min(B, sidx+chunk)
                                    z = lat_vae_full[sidx:e].detach().clone().requires_grad_(True)

                                    # decode to pixels
                                    with torch.enable_grad():
                                        if not z.requires_grad:
                                            z.requires_grad_(True)
                                        imgs_chunk = _vae_decode_pixels(z)
                                        assert imgs_chunk.requires_grad, "VAE decode returned tensor without grad_fn."

                                    # CLIP grad on pixels
                                    imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                                    _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                                    current_logdet = float(_logs.get("logdet", 0.0))
                                    last_logdet = state.get("last_logdet", None)
                                    if (last_logdet is not None) and (current_logdet < last_logdet):
                                        gamma_sched = 0.5 * gamma_sched
                                    state["last_logdet"] = current_logdet
                                    grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                                    # VJP to latent
                                    grad_lat = torch.autograd.grad(
                                        outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                                        retain_graph=False, create_graph=False, allow_unused=True
                                    )[0]
                                    if grad_lat is None:
                                        with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
                                            z32 = z.detach().float().requires_grad_(True)
                                            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                                            dec = pipe.vae.decode(z32 / sf, return_dict=False)[0]
                                            imgs_lin = (dec + 1.0) / 2.0
                                            go = grad_img_vae.float()
                                            grad_lat_32 = torch.autograd.grad(
                                                outputs=imgs_lin, inputs=z32, grad_outputs=go,
                                                retain_graph=False, create_graph=False, allow_unused=False
                                            )[0]
                                        grad_lat = grad_lat_32.to(z.dtype)

                                    # base velocity estimate
                                    v_est = None
                                    if prev_cpu is not None:
                                        total_diff = z - prev_cpu[sidx:e].to(dev_vae, non_blocking=True)
                                        prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
                                        prev_dt   = state.get("prev_dt_unit", None)
                                        if (prev_ctrl is not None) and (prev_dt is not None):
                                            base_move_prev = total_diff - prev_ctrl[sidx:e].to(dev_vae, non_blocking=True)
                                            v_est = base_move_prev / max(prev_dt, 1e-8)
                                        else:
                                            v_est = total_diff / max(dt_unit, 1e-8)

                                    # projection + mid-band boost + hard cap
                                    if v_est is not None:
                                        po = 1.0 if t_norm < args.full_ortho_until else cfg.partial_ortho
                                        g_proj = project_partial_orth(grad_lat, v_est, po)
                                        if args.midband_lambda > 0:
                                            def _lowpass(xx, k=3):
                                                pad = k // 2
                                                w = torch.ones(xx.size(1), 1, k, k, device=xx.device, dtype=xx.dtype) / (k*k)
                                                return F.conv2d(xx, w, padding=pad, groups=xx.size(1))
                                            g_proj = g_proj + args.midband_lambda * (_lowpass(g_proj, 7) - _lowpass(g_proj, 15))
                                        vnorm = _bn(v_est); gnorm = _bn(g_proj)
                                        scale_g = torch.minimum(torch.ones_like(vnorm), vnorm / (gnorm + 1e-12))
                                        g_proj = g_proj * scale_g.view(-1,1,1,1)
                                    else:
                                        g_proj = grad_lat

                                    div_disp = g_proj * dt_unit

                                    beta = _beta_monotone(t_norm, eps=1e-2)
                                    if (beta > 0.0) and (v_est is not None):
                                        xi_prev = state["xi_prev_cpu"][sidx:e].to(dev_vae, non_blocking=True, dtype=g_proj.dtype)
                                        eps = torch.randn_like(g_proj)
                                        eps = project_partial_orth(eps, v_est, 1.0)
                                        if getattr(args, "noise_lowpass_k", 3) and args.noise_lowpass_k > 1:
                                            pad = int(args.noise_lowpass_k) // 2
                                            w = torch.ones(eps.size(1), 1, int(args.noise_lowpass_k), int(args.noise_lowpass_k), device=eps.device, dtype=eps.dtype) / (int(args.noise_lowpass_k)**2)
                                            eps = F.conv2d(eps, w, padding=pad, groups=eps.size(1))
                                        eps = eps - eps.mean(dim=(1,2,3), keepdim=True)
                                        eps = eps / (_bn(eps).view(-1,1,1,1) + 1e-12)
                                        rho = float(getattr(args, "noise_rho", 0.9)); rho = max(0.0, min(0.999, rho))
                                        xi = rho * xi_prev + (1.0 - rho**2) ** 0.5 * eps
                                        noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi
                                        state["xi_prev_cpu"][sidx:e] = xi.detach().to("cpu", dtype=torch.float32)
                                    else:
                                        noise_disp = torch.zeros_like(g_proj)

                                    base_disp = (v_est * dt_unit) if v_est is not None else z
                                    disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                                    if args.early_cap_power != 1.0:
                                        disp_cap = disp_cap * ((1.0 - t_norm) ** args.early_cap_power)
                                    div_raw   = _bn(div_disp)
                                    scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                                    delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp
                                    delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                                    lat_new[sidx:e] = lat_new[sidx:e] + delta_tr

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

                        # -------- 运行一次 pipeline（latent 输出），并落到指定 run_dir --------
                        run_dir = _prompt_run_dir(imgs_root, prompt_text, int(sd), float(g), int(args.steps))
                        os.makedirs(run_dir, exist_ok=True)
                        _log(f"[RUN] concept='{concept}' | prompt='{prompt_text}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}", True)

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

                        # 最终 decode
                        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                        latents_final = latents_out.to(dev_vae, non_blocking=True)
                        if args.enable_vae_tiling:
                            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

                        fp32_decode = bool(getattr(args, 'fp32_last_decode', False))
                        with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                            if fp32_decode:
                                images = (pipe.vae.decode(latents_final.float() / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0
                            else:
                                images = (pipe.vae.decode(latents_final / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0

                        from torchvision.utils import save_image
                        for i in range(images.size(0)):
                            save_image(images[i].cpu(), os.path.join(run_dir, f"img_{i:03d}.png"))

                        # 释放临时张量引用
                        del latents_out, latents_final, images

        _log("Done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()