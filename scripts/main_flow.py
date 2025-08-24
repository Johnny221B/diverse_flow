# -*- coding: utf-8 -*-
"""
SD3.5 本地 + CLIP 多样性增强（稳定版）
- 只传字符串 prompt/negative_prompt（不传 embeds），由管线内部生成对齐的文本特征
- 强制将 3 个 Text Encoders 与 Transformer 放在同一 GPU，避免维度/设备错位
- 回调：VAE 卡解码 -> CLIP 卡体积损失 -> 回 VAE 做 VJP -> δ 写回 Transformer 卡
- 末尾：让管线输出 latent（output_type='latent'），我们手动在 VAE 卡 decode，避免设备冲突
"""
import os, sys, argparse, traceback, time

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
import torch.nn as nn
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
        "tokenizer", "tokenizer_2", "tokenizer_3", "scheduler",  # 非模块，仅打印占位
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

def parse_args():
    ap = argparse.ArgumentParser(description='Diverse SD3.5 (no-embeds, TE=Transformer device, DEBUG)')
    # 生成参数
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=4)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--guidance', type=float, default=4.5)
    ap.add_argument('--seed', type=int, default=42)
    # 本地模型路径
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--out', type=str, default='./outs_sd3_diverse')
    # 多样性目标
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.8)
    ap.add_argument('--t-gate', type=str, default='0.25,0.9')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)
    # 设备（TE 会被强制与 Transformer 同卡）
    ap.add_argument('--device-transformer', type=str, default='cuda:0')
    ap.add_argument('--device-vae',         type=str, default='cuda:1')
    ap.add_argument('--device-clip',        type=str, default='cuda:2')
    ap.add_argument('--device-text1',       type=str, default=None)
    ap.add_argument('--device-text2',       type=str, default=None)
    ap.add_argument('--device-text3',       type=str, default=None)
    # 省显存 + 调试
    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')
    ap.add_argument('--debug', action='store_true')
    return ap.parse_args()

def _resolve_model_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')): return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files: return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')

def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S")
    if debug: print(f"[{ts}] {s}", flush=True)

def main():
    args = parse_args()

    # ===== sys.path 注入（让脚本能 import diverse_flow） =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

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
        from diverse_flow.utils import project_partial_orth, batched_norm, sched_factor

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
        pipe = pipe.to("cpu")  # 先全在 CPU，避免 from_pretrained 期间盲目占显存

        # 移动模块到目标设备
        _log("Moving modules to target devices ...", args.debug)
        if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
        if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
        if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

        # VAE 低显存模式
        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        # xFormers（可选）
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
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # ===== 3) 回调：VAE 解码 -> CLIP 体积损失 -> VJP -> 写回 =====
        state = {"prev_latents_vae": None}

        def _vae_decode_pixels(z):
            sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
            out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
            return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

        def diversity_callback(ppl, i, t, kw):
            steps = args.steps
            t_norm = 1.0 - (i / max(1, steps-1))
            gamma_sched = cfg.gamma0 * sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
            if gamma_sched <= 0:
                return kw

            lat = kw.get("latents", None)
            if lat is None:
                return kw

            import torch
            import torch.backends.cudnn as cudnn
            from torch.utils.checkpoint import checkpoint

            # ---- 预备：把当前 latents 复制出一份“将要返回”的张量（在 Transformer 卡）----
            lat_new = lat.clone()  # 避免对 kw['latents'] 做原地切片写入

            # VAE 卡上的拷贝（不建图，用作块内 leaf 的来源）
            lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
            B = lat_vae_full.size(0)
            chunk = 2 if B >= 2 else 1  # 两张一组

            # 上一步的 latent（保存在 CPU 的），用于速度近似
            prev_cpu = state.get("prev_latents_vae_cpu", None)

            for s in range(0, B, chunk):
                e = min(B, s + chunk)

                # 块内 leaf：z
                z = lat_vae_full[s:e].detach().clone().requires_grad_(True)  # leaf on VAE device

                # 仅为本块建立解码图（降低峰值）
                with torch.enable_grad(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

                # 速度近似方向 v_est（块内）
                v_est = None
                if prev_cpu is not None:
                    v_est = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)

                # 在 CLIP 卡上计算 dL/dimage
                imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
                _, grad_img_clip, _ = vol.volume_loss_and_grad(imgs_clip)
                grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

                # 直接做 VJP：dL/dz
                grad_lat = torch.autograd.grad(
                    outputs=imgs_chunk,
                    inputs=z,
                    grad_outputs=grad_img_vae,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # 正交投影 + trust region
                g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                v_norm = batched_norm(v_est) if v_est is not None else batched_norm(z)
                g_norm = batched_norm(g_proj)
                cap    = cfg.gamma_max_ratio * v_norm
                scale  = torch.minimum(torch.ones_like(cap), cap / (g_norm + 1e-12))
                delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * g_proj  # on VAE

                # 写入到“新的” transformer 侧 latent（一次性回填，避免原地别名）
                delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
                lat_new[s:e] = lat_new[s:e] + delta_tr

                # 尽早同步，帮助把真正的错误定位到上一行而不是下一次 API 调用
                if dev_clip.type == 'cuda':
                    torch.cuda.synchronize(dev_clip)
                if dev_vae.type == 'cuda':
                    torch.cuda.synchronize(dev_vae)

                # 释放块内中间变量（不再 empty_cache）
                del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, v_norm, g_norm, cap, scale, delta_chunk, delta_tr, v_est, z

            # 把新的 latent 交回管线
            kw['latents'] = lat_new

            # 把本步 latent 放到 CPU，减少 VAE 卡驻留
            state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
            return kw

        # ===== 4) 生成 latent（不让管线内部 decode） =====
        os.makedirs(args.out, exist_ok=True)
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

        # ===== 5) 手动在 VAE 卡 decode 最终 latent（避免设备冲突 & 降峰值） =====
        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
        latents_final = latents_out.to(dev_vae, non_blocking=True)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            images = checkpoint(lambda z: _vae_decode_pixels(z), latents_final, use_reentrant=False)

        # 保存
        for i in range(images.size(0)):
            save_image(images[i].cpu(), os.path.join(args.out, f"img_{i:03d}.png"))

        _log(f"Done. Saved to {args.out}", True)

    except Exception as e:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()