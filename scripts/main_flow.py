# -*- coding: utf-8 -*-
"""
SD3.5 本地 + CLIP 多样性增强（方法一致版）  [patched]
- 仅传 prompt/negative_prompt，由管线内部生成文本特征
- 强制 3 个 Text Encoders 与 Transformer 同卡，避免维度/设备错位
- 回调：VAE 解码 -> CLIP 体积能量梯度 -> VJP 回到 latent -> 写回
- 输出 latent，我们手动在 VAE 卡 decode

补丁要点：
  * 体积漂移：仅受 gamma(t)（sched_factor gating）调度
  * 正交噪声：β(t) 晚期更强（p=2.0），幅度 scale=0.35，混合中低频
  * 正交性：体积梯度对基流部分/全正交；在强正交时混入 5% 平行分量缓和结构撕裂
  * 稳控：per-sample γ-gate；能量回退时 div_disp 轻度收敛
"""

import os, sys, argparse, traceback, time, json, re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch.nn as nn
import torch.nn.functional as F


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


# ---- 更健壮的设备审计（跳过非 nn.Module） ----
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


def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S")
    if debug: print(f"[{ts}] {s}", flush=True)


# ------------- 解析 {concept: [prompts...]} -------------
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


def parse_args():
    ap = argparse.ArgumentParser(description='Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)')

    # ---- 输入：新增 --spec 多 concept JSON；保留单 prompt 回退 ----
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept: [prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec not provided')
    ap.add_argument('--negative', type=str, default='')

    # Grid
    ap.add_argument('--G', type=int, default=4)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)                  # 单值回退
    ap.add_argument('--guidances', type=float, nargs='+', default=[3.0, 5.0, 7.5])  # 多值，默认 3.0/5.0/7.5
    ap.add_argument('--seed', type=int, default=42)                         # 单值回退
    ap.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444])  # 多值，默认四个

    # Model & output naming
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--out', type=str, default=None)  # 不再用于路径决定，仅保留兼容
    ap.add_argument('--method',type=str,default='ourmethod')  # 输出路径 uses outputs/{method}_{concept}

    # 方法超参（调整默认值）
    ap.add_argument('--gamma0', type=float, default=0.05)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.85)
    ap.add_argument('--t-gate', type=str, default='0.88,0.99')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-4)

    # 设备（保持原样）
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae',         type=str, default='cuda:0')
    ap.add_argument('--device-clip',        type=str, default='cuda:0')
    ap.add_argument('--device-text1',       type=str, default='cuda:1')
    ap.add_argument('--device-text2',       type=str, default='cuda:1')
    ap.add_argument('--device-text3',       type=str, default='cuda:1')

    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()

    # ===== sys.path 注入（让脚本能 import diverse_flow） =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 解析 JSON / 单 prompt
    if args.spec:
        with open(args.spec, 'r', encoding='utf-8') as f:
            spec_obj = json.load(f, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --spec (JSON with {concept:[prompts...]}) or --prompt")

    # TE 默认与 Transformer 同卡
    if args.device_text1 is None: args.device_text1 = args.device_transformer
    if args.device_text2 is None: args.device_text2 = args.device_transformer
    if args.device_text3 is None: args.device_text3 = args.device_transformer

    try:
        import torch
        import torch.backends.cudnn as cudnn
        from torch.utils.checkpoint import checkpoint
        from diffusers import StableDiffusion3Pipeline
        from torchvision.utils import save_image

        from diverse_flow.config import DiversityConfig
        from diverse_flow.clip_wrapper import CLIPWrapper
        from diverse_flow.volume_objective import VolumeObjective
        from diverse_flow.utils import project_partial_orth
        from diverse_flow.utils import sched_factor as time_sched_factor
        from diverse_flow.utils import batched_norm as _bn

        # 设备 + dtype
        dev_tr  = torch.device(args.device_transformer)
        dev_vae = torch.device(args.device_vae)
        dev_te1 = torch.device(args.device_text1)
        dev_te2 = torch.device(args.device_text2)
        dev_te3 = torch.device(args.device_text3)
        dev_clip= torch.device(args.device_clip)
        dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

        _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, text1={dev_te1}, text2={dev_te2}, text3={dev_te3}, clip={dev_clip}", args.debug)
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

        t0, t1 = args.t-gate.split(',') if False else args.t_gate.split(',')  # keep IDE calm
        t0, t1 = args.t_gate.split(',')
        cfg = DiversityConfig(
            num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
            gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
            partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
            sched_shape=args.sched_shape, clip_image_size=224,
            leverage_alpha=0.7,
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # ======= 目录：按 concept 分 outputs/{method}_{concept}/... =======
        for concept, prompts in concept_to_prompts.items():
            base_dir, imgs_root, eval_dir = _build_root_out(project_root, args.method, concept)
            _log(f"[OUT] base={base_dir}", True)

            # 对每个 prompt × guidance × seed 跑一次
            for prompt_text in prompts:
                for g in (args.guidances if args.guidances is not None else [args.guidance]):
                    for sd in (args.seeds if args.seeds is not None else [args.seed]):

                        # ===== 3) 回调定义（每次 run 重置 state，避免跨 run 污染） =====
                        state = {
                            "prev_latents_vae_cpu": None,
                            "prev_ctrl_vae_cpu":   None,
                            "prev_dt_unit":        None,
                            "prev_prev_latents_vae_cpu": None,
                            "last_logdet":         None,
                            "gamma_auto_done":     False,
                        }

                        def _vae_decode_pixels(z):
                            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
                            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

                        # 晚期更强的噪声调度
                        def _beta_monotone(t_norm: float, eps: float = 1e-2, p: float = 2.0, scale: float = 0.35) -> float:
                            base = min(1.0, t_norm / (1.0 - t_norm + eps))
                            return float((base ** p) * scale)

                        # 中低频混合（避免纯低频）
                        def _band_mixed(x, k_low=3, k_high=7, alpha=0.7):
                            pad_l = k_low // 2
                            pad_h = k_high // 2
                            w_low  = torch.ones(x.size(1), 1, k_low,  k_low,  device=x.device, dtype=x.dtype) / (k_low*k_low)
                            w_high = torch.ones(x.size(1), 1, k_high, k_high, device=x.device, dtype=x.dtype) / (k_high*k_high)
                            lp = F.conv2d(x, w_low,  padding=pad_l, groups=x.size(1))
                            hp = x - F.conv2d(x, w_high, padding=pad_h, groups=x.size(1))
                            return alpha*lp + (1.0-alpha)*hp

                        def diversity_callback(ppl, i, t, kw):
                            # 1) 从调度器拿“真实时间”和本步步长 Δt（严格 FM）
                            ts = ppl.scheduler.timesteps
                            t_cur  = float(ts[i].item())
                            t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
                            t_max, t_min = float(ts[0].item()), float(ts[-1].item())
                            # 规范化时间到 [1,0]
                            t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
                            # 单位化步长
                            dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)

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

                            import torch
                            import torch.backends.cudnn as cudnn
                            from torch.utils.checkpoint import checkpoint

                            for s in range(0, B, chunk):
                                e = min(B, s + chunk)
                                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

                                # —— 解码到像素 —— #
                                with torch.enable_grad(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                                    imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

                                # —— CLIP 卡上求体积损失对“像素”的梯度 —— 
                                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                                _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
                                current_logdet = float(_logs.get("logdet", 0.0))

                                # 能量单调守门（若下降：γ 半衰 + 位移轻度收敛）
                                last_logdet = state.get("last_logdet", None)
                                gate_decay = 1.0
                                if (last_logdet is not None) and (current_logdet < last_logdet):
                                    gamma_sched = 0.5 * gamma_sched
                                    gate_decay = 0.8
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

                                # 在强正交时混入少量平行分量，缓和结构撕裂
                                if (v_est is not None) and (cfg.partial_ortho >= 0.85):
                                    v_flat = v_est
                                    dot = (g_proj * v_flat).sum(dim=(1,2,3), keepdim=True)
                                    vv  = (v_flat * v_flat).sum(dim=(1,2,3), keepdim=True) + 1e-12
                                    g_para = (dot / vv) * v_flat
                                    eta = 0.05
                                    g_proj = (1.0 - eta) * g_proj + eta * g_para

                                # 平滑限幅 + per-sample γ gate
                                if v_est is not None:
                                    vnorm = _bn(v_est)                      # [B,1]
                                    gnorm = _bn(g_proj)                     # [B,1]
                                    ratio = vnorm / (gnorm + 1e-12)
                                    scale_g = torch.minimum(torch.ones_like(vnorm), torch.sqrt(torch.clamp(ratio, min=0.0)))
                                    g_proj = g_proj * scale_g.view(-1, 1, 1, 1)
                                    gamma_gate = torch.clamp(vnorm / (vnorm + gnorm + 1e-12), 0.3, 1.0).view(-1,1,1,1)
                                else:
                                    gamma_gate = 1.0

                                # 确定性体积位移
                                div_disp = g_proj * dt_unit
                                div_disp = gate_decay * div_disp  # 能量回退时轻度收敛

                                # 正交噪声（晚强 + 中低频混合）
                                beta = _beta_monotone(t_norm, eps=1e-2, p=2.0, scale=0.35)
                                if (beta > 0.0) and (v_est is not None):
                                    xi = torch.randn_like(g_proj)
                                    xi = project_partial_orth(xi, v_est, 1.0)  # 对基流全正交
                                    xi = _band_mixed(xi, k_low=3, k_high=7, alpha=0.7)
                                    xi = xi - xi.mean(dim=(1,2,3), keepdim=True)
                                    xi = xi / (_bn(xi).view(-1,1,1,1) + 1e-12)
                                    noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi
                                else:
                                    noise_disp = torch.zeros_like(g_proj)

                                # 信赖域：仅限体积位移幅度
                                base_disp = (v_est * dt_unit) if v_est is not None else z
                                disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                                div_raw   = _bn(div_disp)
                                scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                                # 最终位移写回：γ（含 per-sample gate）只缩放体积位移；噪声直接叠加
                                delta_chunk = (gamma_sched * gamma_gate * scale.view(-1,1,1,1)) * div_disp + noise_disp

                                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                                lat_new[s:e] = lat_new[s:e] + delta_tr

                                # —— 缓存控制位移（CPU）——
                                if "ctrl_cache" not in state:
                                    state["ctrl_cache"] = []
                                state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

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

                        # ===== 4) 生成 latent（不让管线内部 decode），并落到指定 run_dir =====
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
                        )[0]  # -> latent tensor（在 transformer 卡）

                        # ===== 5) 手动在 VAE 卡 decode 最终 latent =====
                        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
                        latents_final = latents_out.to(dev_vae, non_blocking=True)

                        if args.enable_vae_tiling:
                            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

                        with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                            images = checkpoint(lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                                                latents_final, use_reentrant=False)

                        # 保存到该组合的子目录
                        from torchvision.utils import save_image
                        for i in range(images.size(0)):
                            save_image(images[i].cpu(), os.path.join(run_dir, f"img_{i:03d}.png"))

                        # 清理临时张量引用
                        del latents_out, latents_final, images

        _log("Done.", True)

    except Exception as e:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
