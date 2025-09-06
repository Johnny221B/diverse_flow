# -*- coding: utf-8 -*-
# =============================
# File: flow_base/scripts/cads_eval_grid.py
# 仅两处改动：
#   1) 兼容读取 DIMCIM 的 dense_prompts JSON；
#   2) 改写落盘路径为 <outdir>/{method}_{concept}_CIMDIM/{coarse_imgs,dense_imgs,eval}/<prompt_slug>/00.png...
# 其余（模型/设备/CADS回调/参数）保持不变。
# =============================

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline

# ---------- Repo root on sys.path（保持原样） ----------
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Try CADS imports (lower/upper/fallback)
try:
    from cads import CADS
except Exception:
    try:
        from CADS import CADS
    except Exception:
        from cads.cads import CADSConditionAnnealer as CADS


# ---------------------------
# Utilities（原函数保持）
# ---------------------------
def _log(msg: str):
    print(msg, flush=True)

def slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _parse_concepts_spec(spec: Dict[str, Any]) -> "OrderedDict[str, List[str]]":
    # 原有：{ "dog": ["a dog", ...], "truck": [...] }
    if not isinstance(spec, dict):
        raise ValueError("[CADS] Spec must be a JSON object: {concept: [prompts...]}")
    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()
    for concept, plist in spec.items():
        if not isinstance(concept, str) or not isinstance(plist, (list, tuple)):
            continue
        seen, cleaned = set(), []
        for p in plist:
            if isinstance(p, str):
                s = p.strip()
                if s and s not in seen:
                    seen.add(s); cleaned.append(s)
        if cleaned:
            concept_to_prompts[concept] = cleaned
    if not concept_to_prompts:
        raise ValueError("[CADS] No valid {concept: [prompts...]} found in spec.")
    return concept_to_prompts

# --- DIMCIM 兼容：解析 dense_prompts JSON ---
def _parse_dimcim_dense(path: Path) -> Tuple[str, List[str], List[str]]:
    """
    期望结构：
    {
      "concept": "bus",
      "coarse_dense_prompts": [
        { "coarse_prompt": "...",
          "dense_prompts": [
            {"attribute_type": "...", "attribute": "...", "dense_prompt": "..."},
            ...
          ]
        }, ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    concept = (data.get("concept") or "unknown").strip() or "unknown"
    coarse, dense = [], []
    rows = data.get("coarse_dense_prompts", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid DIMCIM JSON: 'coarse_dense_prompts' must be a list.")
    for r in rows:
        if not isinstance(r, dict): continue
        cp = r.get("coarse_prompt")
        if isinstance(cp, str) and cp.strip():
            coarse.append(cp.strip())
        dlist = r.get("dense_prompts", [])
        if isinstance(dlist, list):
            for d in dlist:
                if isinstance(d, dict):
                    dp = d.get("dense_prompt")
                    if isinstance(dp, str) and dp.strip():
                        dense.append(dp.strip())
    # 去重保序
    def _uniq(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return concept, _uniq(coarse), _uniq(dense)

# ---------- 原输出路径（保留给旧 spec 用） ----------
def _outputs_root_legacy(method: str, concept: str) -> Tuple[Path, Path, Path]:
    base = _REPO / "outputs" / f"{method}_{slugify(concept)}"
    eval_dir = base / "eval"
    imgs_dir = base / "imgs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, imgs_dir

# --- DIMCIM 兼容：新输出结构根目录 ---
def _outputs_root_dimcim(out_root: Path, method: str, concept: str) -> Tuple[Path, Path, Path, Path]:
    """
    <out_root>/{method}_{concept}_CIMDIM/{eval, coarse_imgs, dense_imgs}
    """
    base = out_root / f"{method}_{concept}_CIMDIM"
    eval_dir  = base / "eval"
    coarse_d  = base / "coarse_imgs"
    dense_d   = base / "dense_imgs"
    for d in (eval_dir, coarse_d, dense_d):
        d.mkdir(parents=True, exist_ok=True)
    return base, eval_dir, coarse_d, dense_d

def _save_images(imgs: List[Image.Image], out_dir: Path, wh: Tuple[int, int]):
    W, H = int(wh[0]), int(wh[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.size != (W, H):
            img = img.resize((W, H), resample=Image.BICUBIC)
        img.save(out_dir / f"{i:02d}.png")

def _select_dtype(name: str):
    if name == "fp16": return torch.float16
    if name == "bf16": return torch.bfloat16
    return torch.float32

def _move_if_exists(mod, device: str | None):
    if mod is not None and device is not None:
        mod.to(torch.device(device))

def _generators_for_K(exec_device: str, base_seed: int, K: int) -> List[torch.Generator]:
    return [torch.Generator(device=exec_device).manual_seed(int(base_seed) + i) for i in range(K)]


# ---------------------------
# Robust SD3.5 loader（原样）
# ---------------------------
def _resolve_model_dir(root: Path) -> Path:
    if (root / "model_index.json").exists():
        return root
    if root.is_dir():
        snaps = root / "snapshots"
        if snaps.is_dir():
            for d in snaps.rglob("model_index.json"):
                return d.parent
        for d in root.rglob("model_index.json"):
            return d.parent
    raise FileNotFoundError(
        f"[CADS] model_index.json not found under '{root}'. "
        f"Point --model to a diffusers dir (or HF snapshot subdir)."
    )

def load_sd35(model_path: str, torch_dtype):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"[CADS] model path not found: {p}")
    if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt"}:
        _log(f"[CADS] Loading single-file checkpoint: {p}")
        pipe = DiffusionPipeline.from_single_file(p.as_posix(), torch_dtype=torch_dtype)
        if not isinstance(pipe, StableDiffusion3Pipeline):
            pipe = StableDiffusion3Pipeline(**pipe.components)
        return pipe
    idx = _resolve_model_dir(p)
    _log(f"[CADS] Using diffusers folder: {idx}")
    return StableDiffusion3Pipeline.from_pretrained(idx.as_posix(), torch_dtype=torch_dtype, local_files_only=True)


# ---------------------------
# Args（原样）
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CADS grid evaluation on SD-3.5 (Flow-Matching)")

    # Model / output
    p.add_argument("--model", type=str, default="models/stable-diffusion-3.5-medium")
    p.add_argument("--method", type=str, default="cads", help="outputs/{method}_{concept}")
    p.add_argument("--out", type=str, default="outputs", help="输出根目录（旧版未用；DIMCIM 模式会使用此目录）")

    # Prompts（保持原参名；用 --spec 传 DIMCIM json 也可）
    p.add_argument("--spec", type=str, default=None,
                   help="Path to JSON: {concept: [prompts...]} 或 DIMCIM dense_prompts JSON（自动识别）")
    p.add_argument("--prompt", type=str, default=None, help="Fallback if --spec not provided")
    p.add_argument("--negative", type=str, default="low quality, blurry")

    # Grid（原样）
    p.add_argument("--G", type=int, default=4, help="images per prompt (group size)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidances", type=float, nargs="+", default=None, help="e.g., 3.0 7.5 12.0")
    p.add_argument("--guidance", type=float, default=7.5, help="used if --guidances omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444])

    # Resolution（原样）
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # Precision（原样）
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])

    # Devices（原样）
    p.add_argument("--device-transformer", type=str, default=None)
    p.add_argument("--device-vae", type=str, default=None)
    p.add_argument("--device-clip", type=str, default=None)
    p.add_argument("--device-text1", type=str, default=None)
    p.add_argument("--device-text2", type=str, default=None)
    p.add_argument("--device-text3", type=str, default=None)

    # CADS hyperparams（原样）
    g = p.add_argument_group("CADS")
    g.add_argument("--tau1", type=float, default=0.6)
    g.add_argument("--tau2", type=float, default=0.9)
    g.add_argument("--cads-s", type=float, default=0.10, dest="cads_s")
    g.add_argument("--psi", type=float, default=1.0)
    g.add_argument("--no-rescale", action="store_true")
    g.add_argument("--dynamic-cfg", action="store_true")

    # Debug（原样）
    p.add_argument("--debug", action="store_true")
    return p


# ---------------------------
# Main（仅在解析/落盘分支做 DIMCIM 兼容，其他保持原样）
# ---------------------------
def main():
    args = build_parser().parse_args()
    torch_dtype = _select_dtype(args.dtype)

    # 解析 prompts 来源
    dimcim_mode = False
    coarse_prompts: List[str] = []
    dense_prompts:  List[str] = []
    concept_to_prompts: "OrderedDict[str, List[str]]" = OrderedDict()

    if args.spec:
        # 尝试识别是否为 DIMCIM dense_prompts JSON
        try:
            with open(args.spec, "r", encoding="utf-8") as fp:
                maybe = json.load(fp)
            if isinstance(maybe, dict) and "coarse_dense_prompts" in maybe:
                dimcim_mode = True
                concept = (maybe.get("concept") or "unknown").strip() or "unknown"
                cpt, coarse_prompts, dense_prompts = _parse_dimcim_dense(Path(args.spec))
                assert cpt == concept
            else:
                concept_to_prompts = _parse_concepts_spec(maybe)
        except Exception:
            # 回退到原逻辑
            with open(args.spec, "r", encoding="utf-8") as fp:
                spec = json.load(fp)
            concept_to_prompts = _parse_concepts_spec(spec)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide either --spec (JSON file) or --prompt")

    guidances = args.guidances if args.guidances is not None else [args.guidance]
    guidances = [float(g) for g in guidances]

    # ---- Build pipeline once（原样） ----
    _log(f"[CADS] Resolving model from: {args.model}")
    pipe = load_sd35(args.model, torch_dtype=torch_dtype)

    # Optional device placement（原样）
    _move_if_exists(getattr(pipe, "transformer", None), args.device_transformer)
    _move_if_exists(getattr(pipe, "vae", None), args.device_vae)
    te_dev1 = args.device_text1 or args.device_clip
    te_dev2 = args.device_text2 or args.device_clip
    te_dev3 = args.device_text3 or args.device_clip
    _move_if_exists(getattr(pipe, "text_encoder", None), te_dev1)
    _move_if_exists(getattr(pipe, "text_encoder_2", None), te_dev2)
    _move_if_exists(getattr(pipe, "text_encoder_3", None), te_dev3)

    if all(x is None for x in [args.device_transformer, args.device_vae, te_dev1, te_dev2, te_dev3]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)

    if args.debug:
        def dev_of(m):
            try:
                for prm in m.parameters():
                    return prm.device
            except Exception:
                return None
        parts = {
            "transformer": getattr(pipe, "transformer", None),
            "vae": getattr(pipe, "vae", None),
            "text_encoder": getattr(pipe, "text_encoder", None),
            "text_encoder_2": getattr(pipe, "text_encoder_2", None),
            "text_encoder_3": getattr(pipe, "text_encoder_3", None),
        }
        for k, m in parts.items():
            if m is not None:
                _log(f"- {k}: {dev_of(m)}")

    try:
        exec_dev = str(pipe._execution_device)
    except Exception:
        exec_dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 采样循环 ----
    if dimcim_mode:
        # --- DIMCIM 新落盘结构 ---
        out_root = Path(args.out)  # 使用 --out 作为根
        base_dir, eval_dir, coarse_dir, dense_dir = _outputs_root_dimcim(out_root, args.method, concept)
        _log(f"[CADS] outputs base: {base_dir}")
        _log(f"[CADS] eval dir:     {eval_dir}")
        _log(f"[CADS] coarse root:  {coarse_dir}")
        _log(f"[CADS] dense root:   {dense_dir}")

        # Grid: prompt × guidance × seed（保持原有多 guidance/seed 行为）
        def _run_one_prompt(ptxt: str, parent_dir: Path):
            pslug = slugify(ptxt)  # 仅用 prompt 命名（不再加 seed/g/steps）
            run_dir = parent_dir / pslug
            for g in guidances:
                cads = CADS(
                    num_inference_steps=int(args.steps),
                    tau1=float(args.tau1), tau2=float(args.tau2),
                    s=float(args.cads_s), psi=float(args.psi),
                    rescale=(not args.no_rescale),
                    dynamic_cfg=bool(args.dynamic_cfg),
                    seed=int(args.seeds[0] if args.seeds else 1234),
                )
                for sd in args.seeds:
                    gens = _generators_for_K(exec_dev, int(sd), int(args.G))
                    _log(f"[CADS] sampling: concept='{concept}' | prompt='{ptxt}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}")
                    try:
                        result = pipe(
                            prompt=ptxt,
                            negative_prompt=(args.negative if args.negative else None),
                            height=int(args.height), width=int(args.width),
                            num_images_per_prompt=int(args.G),
                            num_inference_steps=int(args.steps),
                            guidance_scale=float(g),
                            generator=gens,
                            callback_on_step_end=cads,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="pil",
                        )
                    except TypeError:
                        result = pipe(
                            prompt=ptxt,
                            negative_prompt=(args.negative if args.negative else None),
                            num_images_per_prompt=int(args.G),
                            num_inference_steps=int(args.steps),
                            guidance_scale=float(g),
                            generator=gens,
                            callback_on_step_end=cads,
                            callback_on_step_end_tensor_inputs=["latents"],
                            output_type="pil",
                        )
                    images = result.images if hasattr(result, "images") else result
                    _save_images(images, run_dir, (args.width, args.height))

        # 先 COARSE 再 DENSE（与我们 DIMCIM 约定一致）
        for cp in coarse_prompts: _run_one_prompt(cp, coarse_dir)
        for dp in dense_prompts:  _run_one_prompt(dp, dense_dir)

    else:
        # ---- 旧 spec 落盘结构（完全保持原行为） ----
        for concept, prompts in concept_to_prompts.items():
            base_dir, eval_dir, imgs_root = _outputs_root_legacy(args.method, concept)
            _log(f"[CADS] outputs base: {base_dir}")
            _log(f"[CADS] eval dir:     {eval_dir}")
            _log(f"[CADS] imgs root:    {imgs_root}")

            for ptxt in prompts:
                for g in guidances:
                    for sd in args.seeds:
                        cads = CADS(
                            num_inference_steps=int(args.steps),
                            tau1=float(args.tau1), tau2=float(args.tau2),
                            s=float(args.cads_s), psi=float(args.psi),
                            rescale=(not args.no_rescale),
                            dynamic_cfg=bool(args.dynamic_cfg),
                            seed=int(sd),
                        )
                        gens = _generators_for_K(exec_dev, int(sd), int(args.G))
                        run_dir = imgs_root / f"{slugify(ptxt)}_seed{sd}_g{g}_s{args.steps}"
                        _log(f"[CADS] sampling: concept='{concept}' | prompt='{ptxt}' | seed={sd} | guidance={g} | steps={args.steps} -> {run_dir}")
                        try:
                            result = pipe(
                                prompt=ptxt,
                                negative_prompt=(args.negative if args.negative else None),
                                height=int(args.height), width=int(args.width),
                                num_images_per_prompt=int(args.G),
                                num_inference_steps=int(args.steps),
                                guidance_scale=float(g),
                                generator=gens,
                                callback_on_step_end=cads,
                                callback_on_step_end_tensor_inputs=["latents"],
                                output_type="pil",
                            )
                        except TypeError:
                            result = pipe(
                                prompt=ptxt,
                                negative_prompt=(args.negative if args.negative else None),
                                num_images_per_prompt=int(args.G),
                                num_inference_steps=int(args.steps),
                                guidance_scale=float(g),
                                generator=gens,
                                callback_on_step_end=cads,
                                callback_on_step_end_tensor_inputs=["latents"],
                                output_type="pil",
                            )
                        images = result.images if hasattr(result, "images") else result
                        _save_images(images, run_dir, (args.width, args.height))

    _log("[CADS] Done.")


if __name__ == "__main__":
    main()