# scripts/baseline_cads_fixed.py
# -*- coding: utf-8 -*-
"""
Baseline: CADS on SD-3.5 (Flow-Matching) — sampling-time only
- Starts from random latents
- Applies CADS each step via Diffusers' callback_on_step_end
- Only needs your local SD-3.5 weights

Robust model loader supports:
  (A) Diffusers directory with model_index.json (direct path)
  (B) Parent directory two levels up that contains the file (ModelScope/HF cache style)
  (C) Hugging Face snapshot directory under .../snapshots/<hash>/
  (D) Single-file checkpoint (*.safetensors/*.ckpt) via from_single_file
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline

# Ensure repo root on sys.path when invoked as "python scripts/...py"
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Prefer lowercase package name 'cads'
try:
    from cads import CADS
except Exception:
    try:
        from CADS import CADS  # fallback if user named folder in caps
    except Exception:
        from cads.cads import CADSConditionAnnealer as CADS


def _log(msg: str):
    print(msg, flush=True)


def slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline CADS on SD-3.5 (Flow-Matching)")

    # I/O
    p.add_argument("--model", type=str,
                   default="models/stable-diffusion-3.5-medium",
                   help=(
                       "Path to SD-3.5: diffusers dir / parent dir / snapshots/<hash>/ / single .safetensors"
                   ))
    p.add_argument("--out", type=str, default="outputs", help="Root output dir")

    # Prompts
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative", type=str, default="")

    # Sampling
    p.add_argument("--G", type=int, default=4, help="Images per prompt")
    p.add_argument("--steps", type=int, default=40, help="Inference steps")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])

    # CADS hyperparams
    g = p.add_argument_group("CADS")
    g.add_argument("--tau1", type=float, default=0.6)
    g.add_argument("--tau2", type=float, default=0.9)
    g.add_argument("--cads-s", type=float, default=0.10, dest="cads_s")
    g.add_argument("--psi", type=float, default=1.0)
    g.add_argument("--no-rescale", action="store_true")
    g.add_argument("--dynamic-cfg", action="store_true")

    # Device placement (optional)
    d = p.add_argument_group("Devices")
    d.add_argument("--device-transformer", type=str, default=None)
    d.add_argument("--device-vae", type=str, default=None)
    d.add_argument("--device-clip", type=str, default=None, help="Alias for text encoders")
    d.add_argument("--device-text1", type=str, default=None)
    d.add_argument("--device-text2", type=str, default=None)
    d.add_argument("--device-text3", type=str, default=None)

    # Debug
    p.add_argument("--debug", action="store_true")
    return p


def select_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


# ---------------- Robust loader ----------------

def _resolve_model_dir(root: Path) -> Path:
    """Return a directory that contains model_index.json.
    Accepts either the exact directory or a parent of it (search up to two
    levels deep). Also handles HuggingFace cache layouts like .../snapshots/<hash>/.
    """
    # 1) exact dir
    if (root / "model_index.json").exists():
        return root

    if root.exists() and root.is_dir():
        # 2) HF cache snapshots
        snaps = root / "snapshots"
        if snaps.is_dir():
            for depth1 in snaps.iterdir():
                if (depth1 / "model_index.json").exists():
                    return depth1
                if depth1.is_dir():
                    for depth2 in depth1.iterdir():
                        if (depth2 / "model_index.json").exists():
                            return depth2
        # 3) generic search up to two levels deep
        for depth1 in root.iterdir():
            if (depth1 / "model_index.json").exists():
                return depth1
            if depth1.is_dir():
                for depth2 in depth1.iterdir():
                    if (depth2 / "model_index.json").exists():
                        return depth2

    raise FileNotFoundError(
        f"Could not find model_index.json under '{root}'. Please point --model to the directory that contains it, "
        f"or to a HF snapshot folder (.../snapshots/<hash>/). If you only have a single .safetensors/.ckpt file, "
        f"pass that file path to --model instead."
    )


def load_sd35(model_path: str, torch_dtype):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path not found: {p}")

    # Case A: single file (.safetensors/.ckpt)
    if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt"}:
        _log(f"Loading single-file checkpoint: {p}")
        pipe = DiffusionPipeline.from_single_file(p.as_posix(), torch_dtype=torch_dtype)
        if not isinstance(pipe, StableDiffusion3Pipeline):
            pipe = StableDiffusion3Pipeline(**pipe.components)
        return pipe

    # Case B: directory — resolve where model_index.json lives (2-level deep)
    idx_dir = _resolve_model_dir(p)
    _log(f"Using diffusers folder: {idx_dir}")
    return StableDiffusion3Pipeline.from_pretrained(
        idx_dir.as_posix(), torch_dtype=torch_dtype, local_files_only=True
    )


# --------------- Utilities ----------------

def inspect_pipe_devices(pipe):
    def dev_of(m):
        try:
            for prm in m.parameters():
                return prm.device
        except Exception:
            return None
        return None

    parts = {
        "transformer": getattr(pipe, "transformer", None),
        "vae": getattr(pipe, "vae", None),
        "text_encoder": getattr(pipe, "text_encoder", None),
        "text_encoder_2": getattr(pipe, "text_encoder_2", None),
        "text_encoder_3": getattr(pipe, "text_encoder_3", None),
    }
    for k, m in parts.items():
        if m is None:
            continue
        _log(f"- {k}: {dev_of(m)}")


def move_if_exists(mod, device: str | None):
    if mod is not None and device is not None:
        mod.to(torch.device(device))


def save_images_and_grid(images: List[Image.Image], out_dir: Path, grid_cols: int | None = None):
    imgs_dir = out_dir / "imgs"
    eval_dir = out_dir / "eval"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        img.save(imgs_dir / f"{i:02d}.png")

    n = len(images)
    if n == 0:
        return
    if grid_cols is None:
        grid_cols = int(math.ceil(math.sqrt(n)))
    grid_rows = int(math.ceil(n / grid_cols))

    w, h = images[0].size
    grid = Image.new("RGB", (grid_cols * w, grid_rows * h))
    for idx, img in enumerate(images):
        r, c = divmod(idx, grid_cols)
        grid.paste(img, (c * w, r * h))
    grid.save(eval_dir / "grid.png")


# --------------- Main ----------------

def main():
    args = build_parser().parse_args()
    torch_dtype = select_dtype(args.dtype)

    tag = f"CADS_{slugify(args.prompt)}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Resolving model from: {args.model}")
    pipe = load_sd35(args.model, torch_dtype=torch_dtype)

    move_if_exists(getattr(pipe, "transformer", None), args.device_transformer)
    move_if_exists(getattr(pipe, "vae", None), args.device_vae)

    te_dev1 = args.device_text1 or args.device_clip
    te_dev2 = args.device_text2 or args.device_clip
    te_dev3 = args.device_text3 or args.device_clip
    move_if_exists(getattr(pipe, "text_encoder", None), te_dev1)
    move_if_exists(getattr(pipe, "text_encoder_2", None), te_dev2)
    move_if_exists(getattr(pipe, "text_encoder_3", None), te_dev3)

    if all(x is None for x in [args.device_transformer, args.device_vae, te_dev1, te_dev2, te_dev3]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)

    if args.debug:
        _log("Pipeline devices:")
        inspect_pipe_devices(pipe)

    cads = CADS(
        num_inference_steps=args.steps,
        tau1=args.tau1, tau2=args.tau2,
        s=args.cads_s, psi=args.psi,
        rescale=(not args.no_rescale), dynamic_cfg=args.dynamic_cfg,
        seed=args.seed,
    )

    try:
        exec_dev = str(pipe._execution_device)
    except Exception:
        exec_dev = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=exec_dev).manual_seed(int(args.seed))

    _log("Start sampling ...")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=(args.negative if args.negative else None),
        height=args.height, width=args.width,
        num_images_per_prompt=args.G,
        num_inference_steps=args.steps,
        guidance_scale=float(args.guidance),
        generator=generator,
        callback_on_step_end=cads,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    images = result.images if hasattr(result, "images") else result
    _log(f"Generated {len(images)} images. Saving to: {out_dir}")
    save_images_and_grid(images, out_dir)
    _log("Done.")


if __name__ == "__main__":
    main()