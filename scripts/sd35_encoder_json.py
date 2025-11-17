#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SD3.5 + (Inception v3 / DINOv2-HF / DINO-timm) encoder — batch runner (JSON prompts)
- 支持 --encoder {inception, dinohf, dino}
- 多 seed × 多 guidance，从 JSON 选 concept 的 prompt
- 输出：outputs/<method>_<concept>/imgs/<prompt_slug>_seed<SEED>_g<GUIDANCE>_s<STEPS>/img_***.png

注意：本脚本默认写入了你们机器上的本地权重路径：
- Inception v3: ~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth
- DINOv2 (HF): /home/cliu322/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415
可通过命令行参数覆盖。
"""

import os, re, sys, json, time, argparse, traceback
from typing import Any, Dict, Optional, List, Tuple, Union

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
    if not isinstance(m, nn.Module): return None
    for p in m.parameters(recurse=False): return p.device
    for b in m.buffers(recurse=False):   return b.device
    for sm in m.children():
        for p in sm.parameters(recurse=False): return p.device
        for b in sm.buffers(recurse=False):   return b.device
    return None

def inspect_pipe_devices(pipe):
    names = ["transformer","text_encoder","text_encoder_2","text_encoder_3","vae","tokenizer","tokenizer_2","tokenizer_3","scheduler"]
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

def assert_on(m: nn.Module, want):
    if not isinstance(m, nn.Module): return
    for p in m.parameters():
        if str(p.device) != str(want): raise RuntimeError(f"Param on {p.device}, expected {want}")
    for b in m.buffers():
        if str(b.device) != str(want): raise RuntimeError(f"Buffer on {b.device}, expected {want}")

def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S"); print(f"[{ts}] {s}", flush=True) if debug else None

def _slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip()); s = re.sub(r'[^A-Za-z0-9._-]+', '', s); s = re.sub(r'_{2,}', '_', s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _resolve_model_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')): return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files: return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')

# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(description='SD3.5 + (Inception/DINO) encoder batch runner (JSON prompts)')

    ap.add_argument('--encoder', type=str, default='inception', choices=['inception','dinohf','dino'])

    # prompts
    ap.add_argument('--prompts-json', type=str, required=True)
    ap.add_argument('--concept', type=str, required=True)
    ap.add_argument('--prompt-index', type=int, default=0)

    # generation
    ap.add_argument('--G', type=int, default=32)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidances', type=str, default='5.0')
    ap.add_argument('--seeds', type=str, default='1111,3333,4444,5555,6666,7777')

    # models
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    # 默认 Inception 与 DINOv2-HF 的本地路径（可覆盖）
    ap.add_argument('--inception-ckpt', type=str, default=os.path.expanduser('~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth'))
    ap.add_argument('--dino-hf-path', type=str, default='/home/cliu322/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415')
    # timm 版 DINO（如需）：
    ap.add_argument('--dino-ckpt', type=str, default=None)
    ap.add_argument('--dino-arch', type=str, default='vit_base_patch14_dinov2')
    ap.add_argument('--dino-pool', type=str, default='cls', choices=['cls','mean'])

    # output
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method', type=str, default='sd35_encoder')

    # diversity config
    ap.add_argument('--gamma0', type=float, default=0.06)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.95)
    ap.add_argument('--t-gate', type=str, default='0.30,0.95')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)

    # noise
    ap.add_argument('--noise-timing', type=str, default='early', choices=['early','late'])
    ap.add_argument('--eta-sde', type=float, default=1.0)
    ap.add_argument('--rho0', type=float, default=0.25)
    ap.add_argument('--rho1', type=float, default=0.05)
    ap.add_argument('--noise-partial-ortho', type=float, default=1.0)
    ap.add_argument('--vnorm-threshold', type=float, default=1e-4)

    # init-ortho
    ap.add_argument('--init-ortho', type=float, default=0.2)
    ap.add_argument('--init-ortho-mode', type=str, default='blend', choices=['blend','off','replace'])

    # devices
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae', type=str, default='cuda:0')
    ap.add_argument('--device-clip', type=str, default='cuda:0', help='作为 encoder 设备使用')

    # misc
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()

# -------------------- main --------------------

def main():
    args = parse_args()

    # sys.path 注入（让脚本能 import diverse_flow）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)

    # prompts
    with open(os.path.expanduser(args.prompts_json),'r',encoding='utf-8') as f:
        j = json.load(f)
    if args.concept not in j: raise KeyError(f"concept '{args.concept}' not in {args.prompts_json}")
    plist = j[args.concept]
    assert isinstance(plist, list) and len(plist)>0
    assert 0 <= args.prompt_index < len(plist)
    prompt = plist[args.prompt_index]

    # seeds/guidances
    seeds = [int(s) for s in str(args.seeds).split(',') if s.strip()!='']
    guidances = [float(g) for g in str(args.guidances).split(',') if g.strip()!='']

    try:
        import torch.backends.cudnn as cudnn
        from torch.utils.checkpoint import checkpoint
        from diffusers import StableDiffusion3Pipeline
        from torchvision.utils import save_image

        from diverse_flow.config import DiversityConfig
        from diverse_flow.inception_wrapper import InceptionV3Wrapper
        from diverse_flow.dino_hf_wrapper import DINOHFWrapper
        from diverse_flow.dino_wrapper import DINOEncoderWrapper
        from diverse_flow.volume_objective import VolumeObjective
        from diverse_flow.utils import project_partial_orth, batched_norm as _bn
        from diverse_flow.utils import sched_factor as time_sched_factor

        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_enc = torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type=='cuda' else torch.float32

        _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, encoder={dev_enc}")
        print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_enc])

        model_dir = _resolve_model_dir(args.model_dir)
        pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
        pipe.set_progress_bar_config(leave=True)
        pipe = pipe.to("cpu")

        if hasattr(pipe, "transformer"): pipe.transformer.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder"): pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"): pipe.vae.to(dev_vae, dtype=dtype)
        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
        if args.enable_xformers:
            try: pipe.enable_xformers_memory_efficient_attention()
            except Exception as e: _log(f"enable_xformers failed: {e}")
        inspect_pipe_devices(pipe)

        # 选择 encoder
        enc_size: int
        if args.encoder == 'inception':
            encoder = InceptionV3Wrapper(checkpoint_path=args.inception_ckpt, device=dev_enc if dev_enc.type=='cuda' else torch.device('cpu'))
            enc_size = 299
        elif args.encoder == 'dinohf':
            encoder = DINOHFWrapper(repo_or_path=os.path.expanduser(args.dino_hf_path), device=dev_enc if dev_enc.type=='cuda' else torch.device('cpu'), pool=args.dino_pool)
            enc_size = getattr(encoder, 'in_size', 224)
        else:  # 'dino' (timm)
            if not args.dino_ckpt:
                raise ValueError("--encoder dino 需要 --dino-ckpt 指向本地 .pth/.pt")
            encoder = DINOEncoderWrapper(arch=args.dino_arch, checkpoint_path=args.dino_ckpt, device=dev_enc if dev_enc.type=='cuda' else torch.device('cpu'), pool=args.dino_pool, in_size=224)
            enc_size = 224

        t0, t1 = [float(x) for x in args.t_gate.split(',')]
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=(t0, t1),
            sched_shape=args.sched_shape, clip_image_size=enc_size,
            leverage_alpha=0.5,
        )
        vol = VolumeObjective(encoder, cfg)

        # 状态 & 工具
        state: Dict[str, Optional[torch.Tensor]] = {"prev_latents_vae_cpu": None, "prev_ctrl_vae_cpu": None, "prev_dt_unit": None, "prev_prev_latents_vae_cpu": None, "last_logdet": None, "gamma_auto_done": False}

        def _vae_decode_pixels(z: torch.Tensor) -> torch.Tensor:
            sf = getattr(pipe.vae.config, 'scaling_factor', 1.0)
            out = pipe.vae.decode(z / sf, return_dict=False)[0]
            return (out.float().clamp(-1,1) + 1.0) / 2.0

        def _brownian_std_from_scheduler(ppl, i):
            sch = ppl.scheduler
            try:
                if hasattr(sch,'sigmas'):
                    s = sch.sigmas.float(); cur = s[i].item(); nxt = s[i+1].item() if i+1<len(s) else s[i].item(); var = max(nxt**2-cur**2,0.0); return float(var**0.5)
                elif hasattr(sch,'alphas_cumprod'):
                    ac = sch.alphas_cumprod.float(); cur = ac[i].item(); nxt = ac[i+1].item() if i+1<len(ac) else ac[i].item(); sig2_cur=max(1.0-cur,0.0); sig2_nxt=max(1.0-nxt,0.0); var=max(sig2_nxt-sig2_cur,0.0); return float(var**0.5)
            except Exception: pass
            ts = sch.timesteps; t_cur=float(ts[i].item()); t_next=float(ts[i+1].item()) if i+1<len(ts) else float(ts[-1].item()); t_max,t_min=float(ts[0].item()),float(ts[-1].item()); dt_unit=abs(t_cur-t_next)/(abs(t_max-t_min)+1e-8); return float(dt_unit**0.5)

        from diverse_flow.utils import project_partial_orth, batched_norm as _bn
        from diverse_flow.utils import sched_factor as time_sched_factor

        def _make_orthonormal_noise_like(x: torch.Tensor) -> torch.Tensor:
            B = x.size(0); M = x[0].numel(); y = torch.randn(B, M, device=x.device, dtype=torch.float32)
            Q,_ = torch.linalg.qr(y.t(), mode='reduced'); Qb = Q.t().to(dtype=x.dtype).contiguous().view_as(x); return Qb

        def _inject_init_ortho(latents: torch.Tensor, strength: float, mode: str='blend') -> torch.Tensor:
            if strength<=0 or mode=='off': return latents
            with torch.no_grad():
                U = _make_orthonormal_noise_like(latents)
                base_norm = (latents.flatten(1).norm(dim=1)+1e-12); U_norm = (U.flatten(1).norm(dim=1)+1e-12)
                U = U * (base_norm / U_norm).view(-1,1,1,1)
                out = (1.0-strength)*latents + strength*U if mode=='replace' else latents + strength*U
            return out

        def diversity_callback(ppl, i, t, kw):
            ts = ppl.scheduler.timesteps
            t_cur=float(ts[i].item()); t_next=float(ts[i+1].item()) if i+1<len(ts) else float(ts[-1].item()); t_max,t_min=float(ts[0].item()),float(ts[-1].item())
            t_norm=(t_cur-t_min)/(t_max-t_min+1e-8); dt_unit=abs(t_cur-t_next)/(abs(t_max-t_min)+1e-8)

            if i==0 and 'latents' in kw and kw['latents'] is not None:
                kw['latents'] = _inject_init_ortho(kw['latents'], strength=float(getattr(args,'init_ortho',0.0)), mode=str(getattr(args,'init_ortho_mode','blend')))

            lat = kw.get('latents')
            if lat is None: return kw

            gamma_sched = args.gamma0 * time_sched_factor(t_norm, (t0,t1), args.sched_shape)
            if gamma_sched <= 0:
                state['prev_prev_latents_vae_cpu'] = state.get('prev_latents_vae_cpu', None)
                state['prev_latents_vae_cpu'] = lat.detach().to('cpu')
                state['prev_dt_unit'] = dt_unit
                return kw

            lat_new = lat.clone()
            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone(); B = lat_vae_full.size(0); chunk = 2 if B>=2 else 1
            prev_cpu = state.get('prev_latents_vae_cpu', None)

            for s in range(0, B, chunk):
                e = min(B, s + chunk)
                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                # 关键：开启梯度环境解码
                with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    imgs_chunk = torch.utils.checkpoint.checkpoint(_vae_decode_pixels, z, use_reentrant=False)

                # 编码器侧体积梯度（注意：不要对 imgs_chunk 做 detach）
                imgs_enc = imgs_chunk.to(dev_enc, non_blocking=True)
                _loss, grad_img_enc, _logs = vol.volume_loss_and_grad(imgs_enc)

                # 回传到 VAE
                grad_img_vae = grad_img_enc.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)
                grad_lat = torch.autograd.grad(
                    outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
                    retain_graph=False, create_graph=False, allow_unused=False
                )[0]

                v_est=None
                if prev_cpu is not None:
                    total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)
                    prev_ctrl = state.get('prev_ctrl_vae_cpu', None); prev_dt = state.get('prev_dt_unit', None)
                    if (prev_ctrl is not None) and (prev_dt is not None):
                        ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True); base_move_prev = total_diff - ctrl_prev; v_est = base_move_prev / max(prev_dt,1e-8)
                    else:
                        v_est = total_diff / max(dt_unit,1e-8)

                g_proj = project_partial_orth(grad_lat, v_est, float(args.partial_ortho)) if v_est is not None else grad_lat
                div_disp = g_proj * dt_unit

                brown_std = _brownian_std_from_scheduler(ppl, i); eta=float(args.eta_sde); base_brown = eta * brown_std
                rho_t = (args.rho0 * t_norm + args.rho1 * (1.0-t_norm)) if args.noise_timing=='early' else (args.rho1 * t_norm + args.rho0 * (1.0-t_norm))
                base_disp = (v_est * dt_unit) if v_est is not None else z; base_norm = _bn(base_disp)
                target_brown = torch.full_like(base_norm, fill_value=max(base_brown,0.0)); target_snr = torch.clamp(base_norm * max(rho_t,0.0), min=0.0); target = torch.minimum(target_brown, target_snr)

                xi = torch.randn_like(g_proj)
                vnorm = _bn(v_est) if v_est is not None else None
                xi_eff = xi if (v_est is None or vnorm is None or float(vnorm.mean().item()) < float(args.vnorm_threshold)) else project_partial_orth(xi, v_est, float(args.noise_partial_ortho))
                xi_norm = _bn(xi_eff); noise_disp = xi_eff / (xi_norm.view(-1,1,1,1)+1e-12) * target.view(-1,1,1,1)

                disp_cap = float(args.gamma_max_ratio) * _bn(base_disp); div_raw = _bn(div_disp); scale = torch.minimum(torch.ones_like(disp_cap), disp_cap/(div_raw+1e-12))
                delta_chunk = (float(args.gamma0) * scale.view(-1,1,1,1)) * div_disp + noise_disp

                lat_new[s:e] = lat_new[s:e] + delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)

                if 'ctrl_cache' not in state: state['ctrl_cache']=[]
                state['ctrl_cache'].append(delta_chunk.detach().to('cpu'))

                if dev_enc.type=='cuda': torch.cuda.synchronize(dev_enc)
                if dev_vae.type=='cuda': torch.cuda.synchronize(dev_vae)
                del imgs_chunk, imgs_enc, grad_img_enc, grad_img_vae, grad_lat, g_proj, div_disp, noise_disp, delta_chunk

            kw['latents'] = lat_new
            state['prev_prev_latents_vae_cpu'] = state.get('prev_latents_vae_cpu', None)
            state['prev_latents_vae_cpu'] = lat_vae_full.detach().to('cpu')
            if 'ctrl_cache' in state:
                state['prev_ctrl_vae_cpu'] = torch.cat(state['ctrl_cache'], dim=0).to('cpu'); del state['ctrl_cache']
            state['prev_dt_unit'] = dt_unit
            return kw

        # outputs
        prompt_slug = _slugify(prompt)
        outputs_root = os.path.join(project_root, 'outputs')
        base_name = f"{args.method}_{args.concept}"
        base_out_dir = args.out.strip() if (args.out and len(args.out.strip())>0) else os.path.join(outputs_root, base_name)
        out_dir_root = os.path.join(base_out_dir, 'imgs'); eval_dir = os.path.join(base_out_dir, 'eval')
        os.makedirs(out_dir_root, exist_ok=True); os.makedirs(eval_dir, exist_ok=True)

        for g in guidances:
            for s in seeds:
                subdir = f"{prompt_slug}_seed{s}_g{g}_s{args.steps}"; run_dir = os.path.join(out_dir_root, subdir)
                os.makedirs(run_dir, exist_ok=True)
                _log(f"Run concept={args.concept} encoder={args.encoder} prompt=\"{prompt}\" seed={s} g={g}")

                generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator(); generator.manual_seed(int(s))
                latents = pipe(
                    prompt=prompt,
                    negative_prompt=None,
                    height=args.height, width=args.width,
                    num_images_per_prompt=args.G,
                    num_inference_steps=args.steps,
                    guidance_scale=float(g),
                    generator=generator,
                    callback_on_step_end=diversity_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output_type='latent', return_dict=False,
                )[0]

                sf = getattr(pipe.vae.config, 'scaling_factor', 1.0)
                zf = latents.to(dev_vae, non_blocking=True)
                if args.enable_vae_tiling:
                    if hasattr(pipe.vae,'enable_slicing'): pipe.vae.enable_slicing()
                    if hasattr(pipe.vae,'enable_tiling'): pipe.vae.enable_tiling()
                imgs = torch.utils.checkpoint.checkpoint(lambda zz: (pipe.vae.decode(zz/sf, return_dict=False)[0].float().clamp(-1,1)+1.0)/2.0, zf, use_reentrant=False)

                for i in range(imgs.size(0)):
                    save_image(imgs[i].cpu(), os.path.join(run_dir, f"img_{i:03d}.png"))
                _log(f"Saved: {run_dir}")

        _log("ALL DONE.")

    except Exception:
        print("\n=== FATAL ERROR ===\n"); traceback.print_exc(); sys.exit(1)

if __name__ == '__main__':
    main()
