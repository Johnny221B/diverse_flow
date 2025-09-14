# # -*- coding: utf-8 -*-
# """
# SD3.5 本地 + CLIP 多样性增强（稳定版）
# - 只传字符串 prompt/negative_prompt（不传 embeds），由管线内部生成对齐的文本特征
# - 强制将 3 个 Text Encoders 与 Transformer 放在同一 GPU，避免维度/设备错位
# - 回调：VAE 卡解码 -> CLIP 卡体积损失 -> 回 VAE 做 VJP -> δ 写回 Transformer 卡
# - 末尾：让管线输出 latent（output_type='latent'），我们手动在 VAE 卡 decode，避免设备冲突
# """
# # source flowqd/bin/activate
# import os, sys, argparse, traceback, time

# def _gb(x): return f"{x/1024**3:.2f} GB"

# def print_mem_all(tag: str, devices: list):
#     import torch
#     lines = [f"[{tag}]"]
#     for d in devices:
#         if d.type == 'cuda':
#             free, total = torch.cuda.mem_get_info(d)
#             used = total - free
#             lines.append(f"  {d}: used={_gb(used)}, free={_gb(free)}")
#         else:
#             lines.append(f"  {d}: CPU")
#     print("\n".join(lines), flush=True)

# # ---- 更健壮的设备审计（跳过非 nn.Module） ----
# import torch.nn as nn
# def _first_device_of_module(m: nn.Module):
#     if not isinstance(m, nn.Module):
#         return None
#     for p in m.parameters(recurse=False):
#         return p.device
#     for b in m.buffers(recurse=False):
#         return b.device
#     for sm in m.children():
#         for p in sm.parameters(recurse=False):
#             return p.device
#         for b in sm.buffers(recurse=False):
#             return b.device
#     return None

# def inspect_pipe_devices(pipe):
#     names = [
#         "transformer",
#         "text_encoder", "text_encoder_2", "text_encoder_3",
#         "vae",
#         "tokenizer", "tokenizer_2", "tokenizer_3", "scheduler",  # 非模块，仅打印占位
#     ]
#     report = {}
#     for name in names:
#         if not hasattr(pipe, name):
#             continue
#         obj = getattr(pipe, name)
#         if obj is None:
#             report[name] = "None"
#             continue
#         if isinstance(obj, nn.Module):
#             dev = _first_device_of_module(obj)
#             report[name] = str(dev) if dev is not None else "module(no params)"
#         else:
#             report[name] = "non-module"
#     print("[pipe-devices]", report, flush=True)

# def assert_on(m, want):
#     if not isinstance(m, nn.Module):
#         return
#     for p in m.parameters():
#         if str(p.device) != str(want):
#             raise RuntimeError(f"Param on {p.device}, expected {want}")
#     for b in m.buffers():
#         if str(b.device) != str(want):
#             raise RuntimeError(f"Buffer on {b.device}, expected {want}")

# def parse_args():
#     ap = argparse.ArgumentParser(description='Diverse SD3.5 (no-embeds, TE=Transformer device, DEBUG)')
#     # 生成参数
#     ap.add_argument('--prompt', type=str, required=True)
#     ap.add_argument('--negative', type=str, default='')
#     ap.add_argument('--G', type=int, default=4)
#     ap.add_argument('--height', type=int, default=1024)
#     ap.add_argument('--width', type=int, default=1024)
#     ap.add_argument('--steps', type=int, default=10)
#     ap.add_argument('--guidance', type=float, default=3.0)
#     ap.add_argument('--seed', type=int, default=42)
#     # 本地模型路径
#     ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
#     ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
#     ap.add_argument('--out', type=str, default=None)
#     ap.add_argument('--method',type=str,default='ourMethod')
#     # 多样性目标
#     ap.add_argument('--gamma0', type=float, default=0.12)
#     ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
#     ap.add_argument('--partial-ortho', type=float, default=0.5)
#     ap.add_argument('--t-gate', type=str, default='0.25,0.9')
#     ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
#     ap.add_argument('--tau', type=float, default=1.0)
#     ap.add_argument('--eps-logdet', type=float, default=1e-3)
#     # 设备（TE 会被强制与 Transformer 同卡）
#     ap.add_argument('--device-transformer', type=str, default='cuda:0')
#     ap.add_argument('--device-vae',         type=str, default='cuda:1')
#     ap.add_argument('--device-clip',        type=str, default='cuda:2')
#     ap.add_argument('--device-text1',       type=str, default=None)
#     ap.add_argument('--device-text2',       type=str, default=None)
#     ap.add_argument('--device-text3',       type=str, default=None)
#     # 省显存 + 调试
#     ap.add_argument('--enable-vae-tiling', action='store_true')
#     ap.add_argument('--enable-xformers', action='store_true')
#     ap.add_argument('--debug', action='store_true')
#     return ap.parse_args()

# def _resolve_model_dir(path: str) -> str:
#     p = os.path.abspath(path)
#     if os.path.isfile(os.path.join(p, 'model_index.json')): return p
#     for root, _, files in os.walk(p):
#         if 'model_index.json' in files: return root
#     raise FileNotFoundError(f'Could not find model_index.json under {path}')

# def _log(s, debug=True):
#     ts = time.strftime("%H:%M:%S")
#     if debug: print(f"[{ts}] {s}", flush=True)

# def main():
#     args = parse_args()

#     # ===== sys.path 注入（让脚本能 import diverse_flow） =====
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(current_dir)
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)

#     # TE 默认与 Transformer 同卡
#     if args.device_text1 is None: args.device_text1 = args.device_transformer
#     if args.device_text2 is None: args.device_text2 = args.device_transformer
#     if args.device_text3 is None: args.device_text3 = args.device_transformer

#     try:
#         import torch
#         import torch.backends.cudnn as cudnn
#         from torch.utils.checkpoint import checkpoint
#         from diffusers import StableDiffusion3Pipeline
#         from torchvision.utils import save_image

#         from diverse_flow.config import DiversityConfig
#         from diverse_flow.clip_wrapper import CLIPWrapper
#         from diverse_flow.volume_objective import VolumeObjective
#         from diverse_flow.utils import project_partial_orth, batched_norm, sched_factor

#         # 设备 + dtype
#         dev_tr  = torch.device(args.device_transformer)
#         dev_vae = torch.device(args.device_vae)
#         dev_te1 = torch.device(args.device_text1)
#         dev_te2 = torch.device(args.device_text2)
#         dev_te3 = torch.device(args.device_text3)
#         dev_clip= torch.device(args.device_clip)
#         dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

#         _log(f"Devices: transformer={dev_tr}, vae={dev_vae}, text1={dev_te1}, text2={dev_te2}, text3={dev_te3}, clip={dev_clip}", args.debug)
#         _log(f"Model dir: {args.model_dir}", args.debug)
#         _log(f"CLIP JIT: {args.clip_jit}", args.debug)

#         print_mem_all("before-pipeline-call", [dev_tr, dev_vae, dev_clip])

#         # ===== 1) CPU 加载，再手动上卡 =====
#         model_dir = _resolve_model_dir(args.model_dir)
#         _log("Loading SD3.5 (CPU) ...", args.debug)
#         pipe = StableDiffusion3Pipeline.from_pretrained(
#             model_dir, torch_dtype=dtype, local_files_only=True,
#         )
#         pipe.set_progress_bar_config(leave=True)
#         pipe = pipe.to("cpu")  # 先全在 CPU，避免 from_pretrained 期间盲目占显存
        
#         print("scheduler:", pipe.scheduler.__class__.__name__)  # FlowMatchEulerDiscreteScheduler / FlowMatchHeunDiscreteScheduler

#         _log("Moving modules to target devices ...", args.debug)
#         if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
#         if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
#         if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
#         if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
#         if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

#         # VAE 低显存模式
#         if args.enable_vae_tiling:
#             if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
#             if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

#         # xFormers（可选）
#         if args.enable_xformers:
#             try:
#                 pipe.enable_xformers_memory_efficient_attention()
#             except Exception as e:
#                 _log(f"enable_xformers failed: {e}", args.debug)

#         inspect_pipe_devices(pipe)
#         if hasattr(pipe, "transformer"):    assert_on(pipe.transformer, dev_tr)
#         if hasattr(pipe, "text_encoder"):   assert_on(pipe.text_encoder,   dev_tr)
#         if hasattr(pipe, "text_encoder_2"): assert_on(pipe.text_encoder_2, dev_tr)
#         if hasattr(pipe, "text_encoder_3"): assert_on(pipe.text_encoder_3, dev_tr)
#         if hasattr(pipe, "vae"):            assert_on(pipe.vae,            dev_vae)

#         # ===== 2) CLIP & Volume objective =====
#         _log("Loading CLIP ...", args.debug)
#         clip = CLIPWrapper(
#             impl="openai_clip", arch="ViT-B-32",
#             jit_path=args.clip_jit, checkpoint_path=None,
#             device=dev_clip if dev_clip.type=='cuda' else torch.device("cpu"),
#         )
#         _log("CLIP ready.", args.debug)

#         t0, t1 = args.t_gate.split(',')
#         cfg = DiversityConfig(
#             num_steps=args.steps, tau=args.tau, eps_logdet=args.eps_logdet,
#             gamma0=args.gamma0, gamma_max_ratio=args.gamma_max_ratio,
#             partial_ortho=args.partial_ortho, t_gate=(float(t0), float(t1)),
#             sched_shape=args.sched_shape, clip_image_size=224,
#         )
#         vol = VolumeObjective(clip, cfg)
#         _log("Volume objective ready.", args.debug)

#         # ===== 3) 回调：VAE 解码 -> CLIP 体积损失 -> VJP -> 写回 =====
#         state = {
#             "prev_latents_vae_cpu": None,
#             "prev_ctrl_vae_cpu":   None,
#             "prev_dt_unit":        None,
#             "prev_prev_latents_vae_cpu": None,  # E3: 再多存一帧做二阶差分
#             "last_logdet":         None,        # E2: 能量单调守门
#             "gamma_auto_done":     False,       # E1: 是否完成首步自校准
#         }

#         def _vae_decode_pixels(z):
#             sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
#             out = pipe.vae.decode(z / sf, return_dict=False)[0]    # [-1,1]
#             return (out.float().clamp(-1,1) + 1.0) / 2.0           # [0,1]

#         def diversity_callback(ppl, i, t, kw):
#             # 1) 从调度器拿“真实时间”和本步步长 Δt（严格 FM）
#             ts = ppl.scheduler.timesteps              # 例如 tensor([..., ...])，单调递减
#             t_cur  = float(ts[i].item())
#             t_next = float(ts[i+1].item()) if i+1 < len(ts) else float(ts[-1].item())
#             t_max, t_min = float(ts[0].item()), float(ts[-1].item())
#             # 规范化时间到 [1,0]（和调度器一一对应）
#             t_norm = (t_cur - t_min) / (t_max - t_min + 1e-8)
#             # 单位化步长（与上面的规范化同尺度）
#             dt_unit = abs(t_cur - t_next) / (abs(t_max - t_min) + 1e-8)
            
#             lat = kw.get("latents")
#             if lat is None:
#                 return kw
            
#             print("DEBUG sched:", type(t_norm), t_norm, type(cfg.t_gate), cfg.t_gate, cfg.sched_shape, flush=True)

#             gamma_sched = cfg.gamma0 * sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
#             if gamma_sched <= 0:
#                 state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
#                 state["prev_latents_vae_cpu"] = lat.detach().to("cpu")
#                 state["prev_dt_unit"] = dt_unit
#                 return kw


#             # lat = kw.get("latents")
#             # if lat is None:
#             #     return kw
#             lat_new = lat.clone()

#             # 把本步 latent 搬到 VAE 卡（作为 leaf 的来源），分块降低峰值显存
#             lat_vae_full = lat.detach().to(dev_vae, non_blocking=True).clone()
#             B = lat_vae_full.size(0)
#             chunk = 2 if B >= 2 else 1

#             # 上一时刻 latent（CPU），用于估基流速度 v_est
#             prev_cpu = state.get("prev_latents_vae_cpu", None)

#             import torch
#             import torch.backends.cudnn as cudnn
#             from torch.utils.checkpoint import checkpoint

#             for s in range(0, B, chunk):
#                 e = min(B, s + chunk)

#                 z = lat_vae_full[s:e].detach().clone().requires_grad_(True)

#                 # —— 解码到像素（建图；checkpoint 降显存）——
#                 with torch.enable_grad(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
#                     imgs_chunk = checkpoint(lambda zz: _vae_decode_pixels(zz), z, use_reentrant=False)

#                 # —— CLIP 卡上求体积损失对“像素”的梯度 —— 
#                 imgs_clip = imgs_chunk.to(dev_clip, non_blocking=True)
#                 _loss, grad_img_clip, _logs = vol.volume_loss_and_grad(imgs_clip)
#                 current_logdet = float(_logs.get("logdet", 0.0))

#                 # E2: 能量单调守门（和上一帧比，若下降则削弱 gamma）
#                 last_logdet = state.get("last_logdet", None)
#                 if (last_logdet is not None) and (current_logdet < last_logdet):
#                     gamma_sched = 0.5 * gamma_sched  # 可取 0.5~0.7

#                 # 记录本步能量，供下一步比较
#                 state["last_logdet"] = current_logdet
#                 grad_img_vae = grad_img_clip.to(dev_vae, non_blocking=True).to(imgs_chunk.dtype)

#                 # —— VJP 穿回 VAE：得到对 latent 的梯度（把它视为控制“速度” u）——
#                 grad_lat = torch.autograd.grad(
#                     outputs=imgs_chunk, inputs=z, grad_outputs=grad_img_vae,
#                     retain_graph=False, create_graph=False, allow_unused=False
#                 )[0]  # [bs,C,h,w] on VAE

#                 # —— 基流速度估计：v_est = Δz / Δt —— 
#                 v_est = None
#                 if prev_cpu is not None:
#                     total_diff = z - prev_cpu[s:e].to(dev_vae, non_blocking=True)   # 本步总位移
#                     prev_ctrl = state.get("prev_ctrl_vae_cpu", None)
#                     prev_dt   = state.get("prev_dt_unit", None)

#                     if (prev_ctrl is not None) and (prev_dt is not None):
#                         # 对应本 chunk 的上一轮控制位移（位移已包含 dt，直接相减）
#                         ctrl_prev = prev_ctrl[s:e].to(dev_vae, non_blocking=True)
#                         base_move_prev = total_diff - ctrl_prev                    # 纯基流位移
#                         v_est = base_move_prev / max(prev_dt, 1e-8)                # 纯基流速度
#                     else:
#                         # 第一轮或无缓存时，回退为原有估计
#                         v_est = total_diff / max(dt_unit, 1e-8)

#                 # —— 质量保护：部分正交到基流速度 + 信赖域（位移尺度）——
#                 # from diverse_flow.utils import project_partial_orth, batched_norm
#                 # g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
#                 # u = g_proj  # 控制方向

#                 # # ==== E1: 首个触发步的自校准（把位移比拉到 0.1~0.3） ====
#                 # if not state.get("gamma_auto_done", False):
#                 #     v_norm = batched_norm(v_est) if v_est is not None else batched_norm(z)
#                 #     g_norm = batched_norm(u)
#                 #     ratio = (g_norm / (v_norm + 1e-12))  # [B,1]
#                 #     med = torch.median(ratio).item() if ratio.numel() > 0 else 1.0
#                 #     if med > 0:
#                 #         target = 0.2  # 目标位移比（0.1~0.3 皆可）
#                 #         gamma_sched = min(gamma_sched, target / med)
#                 #     state["gamma_auto_done"] = True

#                 # # ==== E3: 曲率守门（二阶差分大就收油） ====
#                 # kappa_scale = 1.0
#                 # pp = state.get("prev_prev_latents_vae_cpu", None)
#                 # p  = state.get("prev_latents_vae_cpu", None)
#                 # if (pp is not None) and (p is not None):
#                 #     pp = pp[s:e].to(dev_vae, non_blocking=True)
#                 #     p  = p [s:e].to(dev_vae, non_blocking=True)
#                 #     # 离散二阶差分：z - 2p + pp
#                 #     num = (z - 2.0*p + pp).flatten(1).norm(dim=1, keepdim=True)
#                 #     den = (p - pp).flatten(1).norm(dim=1, keepdim=True) + 1e-8
#                 #     kappa = (num / den)  # [bs,1]
#                 #     # 简单抑制：>0.5 时按 1/(1+kappa) 缩放
#                 #     kappa_scale = torch.clamp(1.0 / (1.0 + kappa), min=0.25, max=1.0)  # 0.25~1.0
#                 #     # 取 batch 中位数来缩放 gamma（也可以逐样本，但更复杂）
#                 #     kappa_scale = torch.median(kappa).item()
#                 #     if kappa_scale < 1.0:
#                 #         gamma_sched = gamma_sched * kappa_scale

#                 # # ==== 信赖域限幅（与你原来一致） ====
#                 # base_disp = (v_est * dt_unit) if v_est is not None else z
#                 # disp_cap  = cfg.gamma_max_ratio * batched_norm(base_disp)
#                 # raw_disp  = batched_norm(u) * dt_unit
#                 # scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (raw_disp + 1e-12))

#                 # delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * (u * dt_unit)

#                 # 体积力（先与基流部分正交）
#                 g_proj = project_partial_orth(grad_lat, v_est, cfg.partial_ortho) if v_est is not None else grad_lat
                
#                 # --- NEW: 早强-晚弱的定向噪声（位移形态；与基流全正交） ---
#                 beta_gate = sched_factor(
#                     t_norm,
#                     cfg.t_gate if getattr(cfg, "noise_use_same_gate", True) else getattr(cfg, "noise_t_gate", (0.0, 0.7)),
#                     cfg.sched_shape
#                 )
#                 beta = getattr(cfg, "noise_beta0", 0.0) * beta_gate
                
#                 if (beta > 0.0) and (v_est is not None):
#                     xi = torch.randn_like(g_proj)                       # 与 g_proj 同形状/设备/dtype
#                     xi = project_partial_orth(xi, v_est, 1.0)           # 对基流“完全正交”
#                     # 位移形态噪声：||noise|| ~ sqrt(2*beta)*sqrt(dt)
#                     noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi
#                 else:
#                     noise_disp = torch.zeros_like(g_proj)
                
#                 # 合并：体积位移 + 噪声位移（都在 latent 空间）
#                 u = g_proj + noise_disp
                
#                 # 与原逻辑一致：统一信赖域限幅（作用在“合力位移 u”上）
#                 base_disp = (v_est * dt_unit) if v_est is not None else z
#                 disp_cap  = cfg.gamma_max_ratio * batched_norm(base_disp)
#                 raw_disp  = batched_norm(u) * dt_unit
#                 scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (raw_disp + 1e-12))
                
#                 # 最终写回（位移）
#                 delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * (u * dt_unit)


#                 delta_tr = delta_chunk.to(lat_new.device, non_blocking=True).to(lat_new.dtype)
#                 lat_new[s:e] = lat_new[s:e] + delta_tr

#                 # —— 缓存本 chunk 的控制位移（CPU 上存上一轮用）——
#                 if "ctrl_cache" not in state:
#                     state["ctrl_cache"] = []
#                 state["ctrl_cache"].append(delta_chunk.detach().to("cpu"))

#                 if dev_clip.type == 'cuda': torch.cuda.synchronize(dev_clip)
#                 if dev_vae.type  == 'cuda': torch.cuda.synchronize(dev_vae)
#                 del imgs_chunk, imgs_clip, grad_img_clip, grad_img_vae, grad_lat, g_proj, u, base_disp, disp_cap, raw_disp, scale, delta_chunk, delta_tr, v_est, z

#             kw["latents"] = lat_new

#             # 先把上一帧挪到 prev_prev，再更新 prev
#             state["prev_prev_latents_vae_cpu"] = state.get("prev_latents_vae_cpu", None)
#             state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")

#             # （B）控制位移与 dt
#             if "ctrl_cache" in state:
#                 state["prev_ctrl_vae_cpu"] = torch.cat(state["ctrl_cache"], dim=0).to("cpu")
#                 del state["ctrl_cache"]
#             state["prev_dt_unit"] = dt_unit

#             return kw

#             # kw["latents"] = lat_new
#             # state["prev_latents_vae_cpu"] = lat_vae_full.detach().to("cpu")
#             # return kw
        
#         import re

#         def _slugify(text: str, maxlen: int = 120) -> str:
#             s = re.sub(r'\s+', '_', text.strip())
#             s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
#             s = re.sub(r'_{2,}', '_', s).strip('._-')
#             return s[:maxlen] if maxlen and len(s) > maxlen else s
        
#         prompt_slug = _slugify(args.prompt)
#         outputs_root = os.path.join(project_root, 'outputs')  # diverse_flow/outputs
#         auto_dirname = f"{args.method}_{prompt_slug or 'no_prompt'}"
#         base_out_dir = args.out if (args.out and len(args.out.strip()) > 0) else os.path.join(outputs_root, auto_dirname)
#         out_dir = os.path.join(base_out_dir, "imgs")
#         eval_dir = os.path.join(base_out_dir, "eval")

#         os.makedirs(out_dir, exist_ok=True)
#         os.makedirs(eval_dir, exist_ok=True)
#         _log(f"Output dir: {out_dir}", True)

#         # ===== 4) 生成 latent（不让管线内部 decode） =====
#         # os.makedirs(args.out, exist_ok=True)
#         generator = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
#         generator.manual_seed(args.seed)

#         _log("Start pipeline() ...", args.debug)
#         latents_out = pipe(
#             prompt=args.prompt,
#             negative_prompt=(args.negative if args.negative else None),
#             height=args.height, width=args.width,
#             num_images_per_prompt=args.G,
#             num_inference_steps=args.steps,
#             guidance_scale=args.guidance,
#             generator=generator,
#             callback_on_step_end=diversity_callback,
#             callback_on_step_end_tensor_inputs=["latents"],
#             output_type="latent",
#             return_dict=False,
#         )[0]  # -> latent tensor（在 transformer 卡）

#         # ===== 5) 手动在 VAE 卡 decode 最终 latent（避免设备冲突 & 降峰值） =====
#         sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
#         latents_final = latents_out.to(dev_vae, non_blocking=True)

#         if args.enable_vae_tiling:
#             if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
#             if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

#         with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
#             images = checkpoint(lambda z: _vae_decode_pixels(z), latents_final, use_reentrant=False)

#         # 保存
#         for i in range(images.size(0)):
#             save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

#         _log(f"Done. Saved to {out_dir}", True)

#     except Exception as e:
#         print("\n=== FATAL ERROR ===\n")
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""
SD3.5 本地 + CLIP 多样性增强（方法一致版）
- 仅传 prompt/negative_prompt，由管线内部生成文本特征
- 强制 3 个 Text Encoders 与 Transformer 同卡，避免维度/设备错位
- 回调：VAE 解码 -> CLIP 体积能量梯度 -> VJP 回到 latent -> 写回
- 输出 latent，我们手动在 VAE 卡 decode
- 方法要点：
  * 体积漂移：仅受 gamma(t)（sched_factor gating）调度
  * 正交噪声：β(t)=min(1, t/(1-t+eps))，单调退火；不乘 gamma，不受 t_gate
  * 正交性：体积梯度对基流部分/全正交（cfg.partial_ortho），噪声对基流全正交
  * 稳控：启用 leverage_alpha（能量层面重加权）；信赖域可保留但放轻
"""
import os, sys, argparse, traceback, time
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

def parse_args():
    ap = argparse.ArgumentParser(description='Diverse SD3.5 (no-embeds, TE=Transformer device, METHOD-CONSISTENT)')

    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=4)
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--method',type=str,default='ourMethod')
    
    ap.add_argument('--gamma0', type=float, default=0.07)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.2)   # 信赖域（先保留，可适当加大或后续移除）
    ap.add_argument('--partial-ortho', type=float, default=0.95)     # 建议更强的正交（0.8~1.0）
    ap.add_argument('--t-gate', type=str, default='0.90,0.98')       # 仅用于确定性体积漂移
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-4)

    ap.add_argument('--device-transformer', type=str, default='cuda:0')
    ap.add_argument('--device-vae',         type=str, default='cuda:1')
    ap.add_argument('--device-clip',        type=str, default='cuda:2')
    ap.add_argument('--device-text1',       type=str, default=None)
    ap.add_argument('--device-text2',       type=str, default=None)
    ap.add_argument('--device-text3',       type=str, default=None)

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
        
        print("scheduler:", pipe.scheduler.__class__.__name__)  # FlowMatchEulerDiscreteScheduler / FlowMatchHeunDiscreteScheduler

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
            leverage_alpha=0.6,
        )
        vol = VolumeObjective(clip, cfg)
        _log("Volume objective ready.", args.debug)

        # ===== 3) 回调：VAE 解码 -> CLIP 体积损失 -> VJP -> 写回 =====
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

        def _beta_monotone(t_norm: float, eps: float = 1e-2) -> float:
            # 早强-晚弱，幅度归一：β(1)=1, β(0)=0
            return float(min(1.0, t_norm / (1.0 - t_norm + eps))) * 0.3
        
        def _lowpass(x, k=3):
            pad = k // 2
            w = torch.ones(x.size(1), 1, k, k, device=x.device, dtype=x.dtype) / (k*k)
            return F.conv2d(x, w, padding=pad, groups=x.size(1))

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

                # E2: 能量单调守门（若下降则削弱 gamma）
                last_logdet = state.get("last_logdet", None)
                if (last_logdet is not None) and (current_logdet < last_logdet):
                    gamma_sched = 0.5 * gamma_sched
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
                g_proj = _lowpass(g_proj, k=5)
                if v_est is not None:
                    vnorm = _bn(v_est)          # [B,1]
                    gnorm = _bn(g_proj)         # [B,1]
                    scale_g = torch.minimum(torch.ones_like(vnorm), vnorm / (gnorm + 1e-12))
                    g_proj = g_proj * scale_g.view(-1, 1, 1, 1)
                div_disp = g_proj * dt_unit  # 确定性体积位移（γ 只作用这里）

                beta = _beta_monotone(t_norm, eps=1e-2)
                if (beta > 0.0) and (v_est is not None):
                    xi = torch.randn_like(g_proj)
                    xi = project_partial_orth(xi, v_est, 1.0)      # 先全正交到基流
                    xi = _lowpass(xi, k=5)                         # 低频化
                    xi = xi - xi.mean(dim=(1,2,3), keepdim=True)
                    xi = xi / (_bn(xi).view(-1,1,1,1) + 1e-12)
                    noise_disp = (2.0 * beta)**0.5 * (dt_unit**0.5) * xi
                else:
                    noise_disp = torch.zeros_like(g_proj)

                base_disp = (v_est * dt_unit) if v_est is not None else z
                disp_cap  = cfg.gamma_max_ratio * _bn(base_disp)
                div_raw   = _bn(div_disp)
                scale     = torch.minimum(torch.ones_like(disp_cap), disp_cap / (div_raw + 1e-12))

                # 最终位移写回：γ 只缩放体积位移；噪声直接叠加
                delta_chunk = (gamma_sched * scale.view(-1,1,1,1)) * div_disp + noise_disp

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
        
        import re
        def _slugify(text: str, maxlen: int = 120) -> str:
            s = re.sub(r'\s+', '_', text.strip())
            s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
            s = re.sub(r'_{2,}', '_', s).strip('._-')
            return s[:maxlen] if maxlen and len(s) > maxlen else s
        
        prompt_slug = _slugify(args.prompt)
        outputs_root = os.path.join(project_root, 'outputs')
        auto_dirname = f"{args.method}_{prompt_slug or 'no_prompt'}"
        base_out_dir = args.out if (args.out and len(args.out.strip()) > 0) else os.path.join(outputs_root, auto_dirname)
        out_dir = os.path.join(base_out_dir, "imgs")
        eval_dir = os.path.join(base_out_dir, "eval")

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        _log(f"Output dir: {out_dir}", True)

        # ===== 4) 生成 latent（不让管线内部 decode） =====
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

        # ===== 5) 手动在 VAE 卡 decode 最终 latent =====
        sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
        latents_final = latents_out.to(dev_vae, non_blocking=True)

        if args.enable_vae_tiling:
            if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

        with torch.inference_mode(), cudnn.flags(enabled=False, benchmark=False, deterministic=False):
            images = checkpoint(lambda z: (pipe.vae.decode(z / sf, return_dict=False)[0].float().clamp(-1,1) + 1.0) / 2.0,
                                latents_final, use_reentrant=False)

        # 保存
        from torchvision.utils import save_image
        for i in range(images.size(0)):
            save_image(images[i].cpu(), os.path.join(out_dir, f"img_{i:03d}.png"))

        _log(f"Done. Saved to {out_dir}", True)

    except Exception as e:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()