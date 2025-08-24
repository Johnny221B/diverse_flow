# ============================
# FILE: scripts/run_sampler.py
# ============================
# NOTE: set CUDA_VISIBLE_DEVICES BEFORE importing torch to respect the chosen GPU.
# source flowqd/bin/activate
import os
import argparse


def parse_args():
    ap = argparse.ArgumentParser(description="DiverseFlow – CLIP Volume Sampler (local weights, CUDA-selectable)")
    ap.add_argument('--cuda', type=int, default=0, help='CUDA device index (use -1 for CPU)')
    ap.add_argument('--G', type=int, default=4, help='group size')
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--num-steps', type=int, default=20)

    # CLIP (local)
    ap.add_argument('--clip-impl', type=str, default='open_clip', choices=['open_clip','openai_clip'])
    ap.add_argument('--clip-arch', type=str, default='ViT-L-14')
    ap.add_argument('--clip-checkpoint', type=str, default=None, help='local checkpoint for open_clip (e.g., .pt/.safetensors)')
    ap.add_argument('--clip-jit', type=str, default=None, help='local JIT .pt for openai_clip')

    # Base flow (local)
    ap.add_argument('--flow-class', type=str, default=None,
                    help='Python path "pkg.mod:ClassName" for your velocity net (must match your local code)')
    ap.add_argument('--flow-ckpt', type=str, default=None, help='Local weight file for your flow model (.pt/.ckpt/.safetensors)')
    ap.add_argument('--flow-kwargs', type=str, default='{}',
                    help='JSON dict of constructor kwargs for your ClassName, e.g. "{\"channels\":64}"')

    ap.add_argument('--out-dir', type=str, default='./outs_demo')

    # Diversity knobs
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--t-gate', type=str, default='0.25,0.9')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--update-every', type=int, default=1)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)
    ap.add_argument('--partial-ortho', type=float, default=0.8)
    return ap.parse_args()


def main():
    args = parse_args()

    # Select CUDA device via env BEFORE importing torch
    if args.cuda is not None and args.cuda >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU

    import json
    import torch
    import torchvision.utils as vutils
    from diverse_flow.config import DiversityConfig
    from diverse_flow.clip_wrapper import CLIPWrapper
    from diverse_flow.sampler import DiverseFlowSampler
    from diverse_flow.base_flow import DummyFlow  # fallback demo
    from diverse_flow.my_flow import MyFlowFromModule  # your real loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[run] device = {device}")

    # Prepare start/target images for demo (replace with your own start state)
    G = args.G
    H = W = args.image_size
    x_start = torch.rand(G,3,H,W, device=device)
    x_target = torch.rand(1,3,H,W, device=device).expand_as(x_start)

    # Base flow (local weights). If --flow-class is given, load your real model; otherwise use DummyFlow.
    if args.flow_class is not None and args.flow_ckpt is not None:
        ctor_kwargs = json.loads(args.flow_kwargs) if args.flow_kwargs else {}
        base = MyFlowFromModule(class_path=args.flow_class,
                                ckpt_path=args.flow_ckpt,
                                ctor_kwargs=ctor_kwargs,
                                device=device)
        print(f"[run] loaded base flow: {args.flow_class} from {args.flow_ckpt}")
    else:
        base = DummyFlow(x_target).to(device)
        print("[run] using DummyFlow (demo)")

    # CLIP wrapper (local weights only)
    clip_wrap = CLIPWrapper(
        impl=args.clip_impl,
        arch=args.clip_arch,
        checkpoint_path=args.clip_checkpoint,
        jit_path=args.clip_jit,
        device=device,
    )

    # Config
    tg1, tg2 = [float(x) for x in args.t_gate.split(',')]
    cfg = DiversityConfig(
        num_steps=args.num_steps,
        gamma0=args.gamma0,
        gamma_max_ratio=args.gamma_max_ratio,
        partial_ortho=args.partial_ortho,
        t_gate=(tg1, tg2),
        sched_shape=args.sched_shape,
        update_every=args.update_every,
        tau=args.tau,
        eps_logdet=args.eps_logdet,
        device=device
    )

    sampler = DiverseFlowSampler(base, clip_wrap, cfg)
    x_final, logs = sampler.sample(x_start)

    os.makedirs(args.out_dir, exist_ok=True)
    vutils.save_image(x_final, os.path.join(args.out_dir, 'group_final.png'), nrow=G)
    print(f"[run] saved {os.path.join(args.out_dir, 'group_final.png')}")
    print("[run] last min/mean angle deg:",
          next((v for v in reversed(logs['min_angle_deg']) if v==v), float('nan')),
          next((v for v in reversed(logs['mean_angle_deg']) if v==v), float('nan')))


if __name__ == "__main__":
    main()


# ============================
# FILE: diverse_flow/my_flow.py
# ============================
from typing import Optional, Dict, Any
import importlib
import os
import torch
from safetensors.torch import load_file as load_safetensors
from .base_flow import BaseFlow


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.safetensors', '.sft']:
        return load_safetensors(path)
    state = torch.load(path, map_location='cpu')
    # common wrappers
    for key in ['state_dict', 'model', 'net', 'ema_state_dict']:
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            return state[key]
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unsupported checkpoint structure: {type(state)} from {path}")


def _instantiate(class_path: str, **kwargs):
    """Instantiate a class from 'pkg.module:ClassName' string."""
    if ':' not in class_path:
        raise ValueError("class_path must be 'pkg.mod:ClassName'")
    mod_name, cls_name = class_path.split(':', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)


class MyFlowFromModule(BaseFlow):
    """
    Generic wrapper to load your local velocity network and expose a .velocity(x,t,cond)->v API.

    Args:
        class_path: 'pkg.mod:ClassName' of your model (must be importable locally)
        ckpt_path:  path to local weights (.pt/.ckpt/.safetensors)
        ctor_kwargs: dict of constructor kwargs for your ClassName
        device: torch.device

    Your model is expected to implement forward(x, t, **cond)->v OR forward(x, t)->v.
    If your model expects scaled timesteps (e.g., 0..999), override self.t_map.
    """
    def __init__(self, class_path: str, ckpt_path: str, ctor_kwargs: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.net = _instantiate(class_path, **(ctor_kwargs or {}))
        sd = _load_state_dict_any(ckpt_path)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if len(missing)+len(unexpected) > 0:
            print(f"[MyFlow] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        self.net.eval()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        # default maps t in [1,0] to itself (continuous); override if your model expects int steps.
        self.t_map = lambda t: t

    @torch.no_grad()
    def velocity(self, x: torch.Tensor, t_scalar: float, cond: Optional[Dict]=None) -> torch.Tensor:
        t_in = self.t_map(t_scalar)
        if cond is None:
            try:
                return self.net(x, t_in)
            except TypeError:
                return self.net(x, t_in, {})
        else:
            return self.net(x, t_in, **cond) if isinstance(cond, dict) else self.net(x, t_in, cond)

    # Helper to use discrete timesteps; call e.g. myflow.use_linear_timesteps(1000) after init.
    def use_linear_timesteps(self, n_steps:int=1000):
        self.t_map = lambda t: int(round((1.0 - max(0.0, min(1.0, t))) * (n_steps-1)))
        return self


# ============================
# FILE: scripts/run_sd3_medium_diverse.py
# ============================
# Tailored to your local layout:
#   - SD3.5-Medium under: ./diverse_flow/models/stable-diffusion-3.5-medium
#   - OpenAI CLIP JIT under: /home/yangyz/.cache/clip/ViT-B-32.pt
#   - Reasoning code under:  ./diverse_flow/reasoning/*.py
#
# Example run:
#   HF_HUB_OFFLINE=1 \
#   python scripts/run_sd3_medium_diverse.py \
#     --cuda 0 \
#     --prompt "a cozy cabin in a snowy forest" \
#     --G 4 --height 1024 --width 1024 --steps 28 --guidance 4.5

import os
import argparse
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="SD3.5-M + CLIP-Volume diversity (local ModelScope + local CLIP)")
    ap.add_argument('--cuda', type=int, default=0, help='CUDA device index (-1 for CPU)')
    # Defaults set to your paths
    ap.add_argument('--model-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'diverse_flow', 'models', 'stable-diffusion-3.5-medium')),
                    help='Local dir of SD3.5-Medium (ModelScope/diffusers folder)')
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--negative', type=str, default='')
    ap.add_argument('--G', type=int, default=24, help='images per prompt (default: 24)')
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--guidance', type=float, default=4.5)  # common default for SD3.5-M
    ap.add_argument('--seed', type=int, default=42)
    # diversity knobs
    ap.add_argument('--gamma0', type=float, default=0.12)
    ap.add_argument('--gamma-max-ratio', type=float, default=0.3)
    ap.add_argument('--partial-ortho', type=float, default=0.8)
    ap.add_argument('--t-gate', type=str, default='0.25,0.9')
    ap.add_argument('--sched-shape', type=str, default='sin2', choices=['sin2','t1mt'])
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--eps-logdet', type=float, default=1e-3)
    # local CLIP (your JIT path)
    ap.add_argument('--clip-impl', type=str, default='openai_clip', choices=['openai_clip','open_clip'])
    ap.add_argument('--clip-arch', type=str, default='ViT-B-32')  # for logging only when using JIT
    ap.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'))
    ap.add_argument('--clip-checkpoint', type=str, default=None, help='for open_clip state_dict if you switch impl')
    ap.add_argument('--out', type=str, default='./outs_sd3_diverse')
    return ap.parse_args()


def _resolve_model_dir(path: str) -> str:
    """Return a directory that actually contains model_index.json; if not at root, try one level deeper."""
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')):
        return p
    # search depth 2
    for root, dirs, files in os.walk(p):
        if 'model_index.json' in files:
            return root
    raise FileNotFoundError(f"Could not find model_index.json under {path}. If this is a ModelScope repo, pass the diffusers subfolder.")


def main():
    args = parse_args()

    # Respect chosen CUDA before importing torch
    if args.cuda is not None and args.cuda >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Add your reasoning module path so imports work without packaging
    THIS_DIR = os.path.dirname(__file__)
    REASONING_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'diverse_flow', 'reasoning'))
    if REASONING_DIR not in sys.path:
        sys.path.insert(0, REASONING_DIR)

    import torch
    from diffusers import StableDiffusion3Pipeline
    from torchvision.utils import save_image
    from functools import partial

    # Import your local modules from reasoning/
    from config import DiversityConfig
    from clip_wrapper import CLIPWrapper
    from volume_objective import VolumeObjective
    from utils import project_partial_orth, batched_norm, sched_factor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_dir = _resolve_model_dir(args.model_dir)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=dtype, local_files_only=True
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    # CLIP wrapper: OpenAI JIT at your path by default
    clip_wrap = CLIPWrapper(
        impl=args.clip_impl,
        arch=args.clip_arch,
        checkpoint_path=args.clip_checkpoint,
        jit_path=args.clip_jit,
        device=device,
    )

    cfg = DiversityConfig(
        num_steps=args.steps,
        gamma0=args.gamma0,
        gamma_max_ratio=args.gamma_max_ratio,
        partial_ortho=args.partial_ortho,
        t_gate=tuple(float(x) for x in args.t_gate.split(',')),
        sched_shape=args.sched_shape,
        tau=args.tau,
        eps_logdet=args.eps_logdet,
        device=device,
    )
    vol = VolumeObjective(clip_wrap, cfg)

    @torch.no_grad()
    def diversity_callback(pipe, step_index, timestep, callback_kwargs, *, vol: VolumeObjective, cfg: DiversityConfig, steps: int):
        latents = callback_kwargs.get('latents', None)
        model_out = callback_kwargs.get('noise_pred', None) or callback_kwargs.get('model_output', None)
        if latents is None:
            return callback_kwargs
        t_norm = 1.0 - (step_index / max(1, steps-1))
        gamma_sched = cfg.gamma0 * sched_factor(t_norm, cfg.t_gate, cfg.sched_shape)
        if gamma_sched <= 0:
            return callback_kwargs

        latents_req = latents.detach().clone().requires_grad_(True)
        sf = getattr(pipe.vae.config, 'scaling_factor', 1.0)
        imgs = pipe.vae.decode(latents_req / sf, return_dict=False)[0]
        imgs = (imgs.float().clamp(-1,1) + 1.0) / 2.0
        with torch.enable_grad():
            loss, grad_img, vlogs = vol.volume_loss_and_grad(imgs)
        grad_lat = torch.autograd.grad(loss, latents_req, retain_graph=False, create_graph=False)[0]

        g_proj = grad_lat
        if model_out is not None and model_out.shape == latents.shape:
            g_proj = project_partial_orth(grad_lat, model_out, cfg.partial_ortho)
        v_norm = batched_norm(model_out) if model_out is not None else batched_norm(latents)
        g_norm = batched_norm(g_proj)
        cap = cfg.gamma_max_ratio * v_norm
        scale = torch.minimum(torch.ones_like(cap), cap / (g_norm + 1e-12))
        delta = (gamma_sched * scale.view(-1,1,1,1)) * g_proj
        callback_kwargs['latents'] = (latents + delta).to(latents.dtype)
        return callback_kwargs

    os.makedirs(args.out, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipe(
        prompt=[args.prompt],
        negative_prompt=[args.negative] if args.negative else None,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_images_per_prompt=args.G,
        generator=generator,
        callback_on_step_end=partial(diversity_callback, vol=vol, cfg=cfg, steps=args.steps),
        callback_on_step_end_tensor_inputs=["latents","noise_pred"],
    ).images

    import torchvision
    import numpy as np
    import PIL.Image
    import torch as _torch

    imgs = _torch.stack([_torch.from_numpy(np.array(im)).permute(2,0,1) for im in images]).float()/255.0
    grid = torchvision.utils.make_grid(imgs, nrow=args.G)
    save_image(grid, os.path.join(args.out, 'sd3m_diverse_grid.png'))
    print('[run] saved', os.path.join(args.out, 'sd3m_diverse_grid.png'))


if __name__ == '__main__':
    main()
