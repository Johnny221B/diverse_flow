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