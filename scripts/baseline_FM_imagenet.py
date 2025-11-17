#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FM Baseline for SD3.5 (multi-seed, multi-guidance)

新增：
- 支持 --spec 指向 JSON: { "truck":[...], "bus":[...], ... }
- 支持 --imagenet-json 指向 ImageNet-400 prompts: { "0":"...", "1":"...", ... }
- 遍历 (a) concept × prompt × guidance × seed 的 grid 模式
- 或 (b) ImageNet-400 单图模式：400 类 × 1 prompt × 1 image
- 输出 (grid):
    outputs/<method>_<concept>/{eval,imgs}/
    imgs/<prompt_slug>_seed<SEED>_g<GUIDANCE>_s<STEPS>/img_000.png...
  输出 (ImageNet-400):
    outputs/imagenet_400/<method>/cls_000.png ... cls_399.png

保持不变：
- 管线输出 latent，在 VAE 卡上手动 decode
- 其它生成超参/设备参数

示例 (grid):
python fm_sd35.py \
  --spec ./specs/prompt.json \
  --G 4 --steps 30 --guidances 3.0 5.0 7.5 --seeds 1111 2222 3333 4444 \
  --model-dir ../models/stable-diffusion-3.5-medium \
  --device-transformer cuda:1 --device-vae cuda:0 \
  --method fm

示例 (ImageNet-400):
python fm_sd35.py \
  --imagenet-json ./imagenet_400_prompts.json \
  --steps 30 --guidance 5.0 --seed 0 \
  --model-dir ../models/stable-diffusion-3.5-medium \
  --device-transformer cuda:1 --device-vae cuda:0 \
  --method fm
"""

import os, re, sys, time, json, argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


# -------------- small utils --------------

def _slugify(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'\s+', '_', text.strip())
    s = re.sub(r'[^A-Za-z0-9._-]+', '', s)
    s = re.sub(r'_{2,}', '_', s).strip('._-')
    return s[:maxlen] if maxlen and len(s) > maxlen else s

def _resolve_model_dir(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, 'model_index.json')):
        return p
    for root, _, files in os.walk(p):
        if 'model_index.json' in files:
            return root
    raise FileNotFoundError(f'Could not find model_index.json under {path}')

def _log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

def _first_device_of_module(m: nn.Module) -> Optional[torch.device]:
    if not isinstance(m, nn.Module): return None
    for p in m.parameters(recurse=False): return p.device
    for b in m.buffers(recurse=False): return b.device
    for sm in m.children():
        for p in sm.parameters(recurse=False): return p.device
        for b in sm.buffers(recurse=False): return b.device
    return None

def inspect_pipe_devices(pipe):
    names = ["transformer","text_encoder","text_encoder_2","text_encoder_3","vae","scheduler"]
    rep = {}
    for n in names:
        if hasattr(pipe, n):
            obj = getattr(pipe, n)
            if isinstance(obj, nn.Module):
                dev = _first_device_of_module(obj)
                rep[n] = str(dev) if dev is not None else "module(no params)"
            else:
                rep[n] = "non-module"
    _log(f"[pipe-devices] {rep}")

def parse_list_maybe(s_or_list, tp=float) -> List:
    # 兼容：既支持 "3.0,5.0" 这样的字符串，也支持 nargs='+' 传入的 list
    if s_or_list is None:
        return []
    if isinstance(s_or_list, (list, tuple)):
        return [tp(x) for x in s_or_list]
    s = str(s_or_list)
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            vals.append(tp(tok))
    return vals

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

def _outputs_root(project_root: str, method: str, concept: str) -> Tuple[str, str, str]:
    base = os.path.join(project_root, 'outputs', f"{method}_{_slugify(concept)}")
    imgs = os.path.join(base, "imgs")
    eval_dir = os.path.join(base, "eval")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return base, imgs, eval_dir

def _prompt_run_dir(imgs_root: str, prompt: str, seed: int, guidance: float, steps: int) -> str:
    pslug = _slugify(prompt)
    return os.path.join(imgs_root, f"{pslug}_seed{seed}_g{guidance}_s{steps}")


# -------------- args --------------

def parse_args():
    ap = argparse.ArgumentParser(description="FM Baseline (SD3.5) with manual VAE decode + multi seed/guidance + ImageNet-400")

    # 新增：ImageNet-400 prompt JSON（优先级最高）
    ap.add_argument('--imagenet-json', type=str, default=None,
                    help='Path to ImageNet-400 JSON: { "0": "...", "1": "...", ... }')

    # 多 concept JSON；保留单 prompt 回退
    ap.add_argument('--spec', type=str, default=None, help='Path to JSON: {concept:[prompts...]}')
    ap.add_argument('--prompt', type=str, default=None, help='Single prompt if --spec/--imagenet-json not provided')
    ap.add_argument('--negative', type=str, default='')

    # 生成参数
    ap.add_argument('--G', type=int, default=1, help="images per prompt (grid mode)")
    ap.add_argument('--height', type=int, default=512)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)

    # 多 guidance/seed（既支持 nargs 也支持逗号分隔）
    ap.add_argument('--guidances', nargs='*', default=[3.0, 5.0, 7.5],
                    help='e.g. 3.0 5.0 7.5 OR "3.0,5.0,7.5" (grid mode)')
    ap.add_argument('--guidance', type=float, default=3.0,
                    help='Single guidance used in ImageNet-400 mode (and as fallback if --guidances empty).')

    ap.add_argument('--seeds', nargs='*', default=[1111, 2222, 3333, 4444],
                    help='e.g. 1111 2222 3333 4444 OR "1111,2222,..." (grid mode)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Base seed for ImageNet-400 mode; if None, use first element of --seeds.')

    # 模型路径
    ap.add_argument('--model-dir', type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'stable-diffusion-3.5-medium')))

    # 设备（与项目脚本一致）
    ap.add_argument('--device-transformer', type=str, default='cuda:1')
    ap.add_argument('--device-vae', type=str, default='cuda:0')
    ap.add_argument('--device-text1', type=str, default='cuda:1')
    ap.add_argument('--device-text2', type=str, default='cuda:1')
    ap.add_argument('--device-text3', type=str, default='cuda:1')

    ap.add_argument('--enable-vae-tiling', action='store_true')
    ap.add_argument('--enable-xformers', action='store_true')

    # 方法名（用于目录前缀）
    ap.add_argument('--method', type=str, default='standardFM')

    # 兼容：保留旧的 --out，但此脚本按你的规范总是写到 ../outputs/<method>_<concept>/ 或 ../outputs/imagenet_400/<method>/
    ap.add_argument('--out', type=str, default=None)

    return ap.parse_args()


# -------------- main --------------

def main():
    args = parse_args()

    # 解析 seeds / guidances（兼容空格或逗号）
    seeds_list = [int(x) for x in parse_list_maybe(args.seeds, int)]
    guidances_list = [float(x) for x in parse_list_maybe(args.guidances, float)]
    if not guidances_list:
        guidances_list = [float(args.guidance)]

    # TE 与 Transformer 同卡（保持一致）
    if args.device_text1 is None: args.device_text1 = args.device_transformer
    if args.device_text2 is None: args.device_text2 = args.device_transformer
    if args.device_text3 is None: args.device_text3 = args.device_transformer

    from diffusers import StableDiffusion3Pipeline
    from torchvision.utils import save_image

    dev_tr  = torch.device(args.device_transformer)
    dev_vae = torch.device(args.device_vae)
    dtype   = torch.bfloat16 if dev_tr.type == 'cuda' else torch.float32

    # 加载模型并放置到目标设备
    model_dir = _resolve_model_dir(args.model-dir if hasattr(args, "model-dir") else args.model_dir)
    # 上面一行保险；绝大多数情况用 args.model_dir
    model_dir = _resolve_model_dir(args.model_dir)
    _log(f"Loading SD3.5 from: {model_dir}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=dtype, local_files_only=True,
    )
    pipe = pipe.to("cpu")
    if hasattr(pipe, "transformer"):    pipe.transformer.to(dev_tr,  dtype=dtype)
    if hasattr(pipe, "text_encoder"):   pipe.text_encoder.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_2"): pipe.text_encoder_2.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "text_encoder_3"): pipe.text_encoder_3.to(dev_tr, dtype=dtype)
    if hasattr(pipe, "vae"):            pipe.vae.to(dev_vae,        dtype=dtype)

    if args.enable_xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception as e: _log(f"enable_xformers failed: {e}")

    if args.enable_vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):  pipe.vae.enable_tiling()

    inspect_pipe_devices(pipe)

    # 输出根（scripts 与 outputs 同级）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_root = os.path.join(project_root, 'outputs')

    # VAE scaling_factor
    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)

    # ===================== 模式 B：ImageNet-400 单图评估 =====================
    if args.imagenet_json is not None:
        imagenet_path = Path(args.imagenet_json)
        if not imagenet_path.exists():
            raise FileNotFoundError(f"ImageNet-400 json not found: {imagenet_path}")

        with imagenet_path.open("r", encoding="utf-8") as fp:
            cls_to_prompt: Dict[str, str] = json.load(fp, object_pairs_hook=OrderedDict)

        base_seed = args.seed if args.seed is not None else (seeds_list[0] if len(seeds_list) > 0 else 0)
        guidance = float(args.guidance)

        _log(f"[FM-ImageNet400] base_seed={base_seed}, guidance={guidance}, steps={args.steps}")
        _log(f"[FM-ImageNet400] #classes = {len(cls_to_prompt)}")

        out_root = os.path.join(outputs_root, "imagenet_400", args.method)
        os.makedirs(out_root, exist_ok=True)
        _log(f"[FM-ImageNet400] output dir: {out_root}")

        for cls_id_str, prompt_text in sorted(cls_to_prompt.items(), key=lambda kv: int(kv[0])):
            cls_id = int(cls_id_str)
            cur_seed = int(base_seed) + cls_id

            _log(f"[FM-ImageNet400] cls={cls_id:03d} seed={cur_seed} prompt='{prompt_text}'")

            gen = torch.Generator(device=dev_tr) if dev_tr.type == 'cuda' else torch.Generator()
            gen.manual_seed(cur_seed)

            t0 = time.time()
            latents = pipe(
                prompt=prompt_text,
                negative_prompt=(args.negative if args.negative else None),
                height=args.height, width=args.width,
                num_images_per_prompt=1,
                num_inference_steps=args.steps,
                guidance_scale=guidance,
                generator=gen,
                output_type="latent",
                return_dict=False,
            )[0]

            latents_vae = latents.to(dev_vae, non_blocking=True)
            with torch.inference_mode():
                images = pipe.vae.decode(latents_vae / sf, return_dict=False)[0]   # [-1,1]
                images = (images.float().clamp(-1,1) + 1.0) / 2.0                  # [0,1]

            img = images[0].cpu()
            save_path = os.path.join(out_root, f"cls_{cls_id:03d}.png")
            save_image(img, save_path)
            t1 = time.time()
            _log(f"[FM-ImageNet400] saved -> {save_path} | time={t1-t0:.2f}s")

            del latents, latents_vae, images, img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _log("[FM-ImageNet400] Done.")
        return

    # ===================== 模式 A：原始 grid 多 prompt × guidance × seed =====================

    # 解析输入来源（此时没有 imagenet-json）
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fp:
            spec_obj = json.load(fp, object_pairs_hook=OrderedDict)
        concept_to_prompts = _parse_concepts_spec(spec_obj)
    elif args.prompt:
        concept_to_prompts = OrderedDict([("single", [args.prompt])])
    else:
        raise ValueError("Provide --imagenet-json or --spec (JSON with {concept:[prompts...]}) or --prompt")

    seeds = seeds_list
    guidances = guidances_list

    # 遍历 concept × prompt × guidance × seed
    for concept, prompts in concept_to_prompts.items():
        base_root, imgs_root, eval_root = _outputs_root(project_root, args.method, concept)
        _log(f"[OUT] base={base_root}")

        for prompt_text in prompts:
            for gval in guidances:
                for s in seeds:
                    # run 目录：imgs/<prompt_slug>_seed{SEED}_g{GUIDANCE}_s{STEPS}/
                    run_dir = _prompt_run_dir(imgs_root, prompt_text, int(s), float(gval), int(args.steps))
                    os.makedirs(run_dir, exist_ok=True)

                    _log(f"FM: concept='{concept}' | prompt='{prompt_text}' | seed={s} | g={gval} | steps={args.steps}")

                    # 生成器
                    gen = torch.Generator(device=dev_tr) if dev_tr.type=='cuda' else torch.Generator()
                    gen.manual_seed(int(s))

                    # 只出 latent，再到 VAE 卡 decode
                    t0 = time.time()
                    latents = pipe(
                        prompt=prompt_text,
                        negative_prompt=(args.negative if args.negative else None),
                        height=args.height, width=args.width,
                        num_images_per_prompt=args.G,
                        num_inference_steps=args.steps,
                        guidance_scale=float(gval),
                        generator=gen,
                        output_type="latent",
                        return_dict=False,
                    )[0]

                    latents_vae = latents.to(dev_vae, non_blocking=True)
                    with torch.inference_mode():
                        images = pipe.vae.decode(latents_vae / sf, return_dict=False)[0]   # [-1,1]
                        images = (images.float().clamp(-1,1) + 1.0) / 2.0                  # [0,1]

                    # 保存
                    for i in range(images.size(0)):
                        save_image(images[i].cpu(), os.path.join(run_dir, f"img_{i:03d}.png"))
                    t1 = time.time()

                    _log(f"Saved {images.size(0)} images -> {run_dir} | time={t1-t0:.2f}s")

                    # 适度清理
                    del latents, latents_vae, images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
