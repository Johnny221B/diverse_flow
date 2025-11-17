#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDXL / SDXL-Turbo baseline (no diversity pipeline), JSON-driven prompts.

默认：
- Backbone: stabilityai/sdxl-turbo （可通过 --model-dir 改成任意 SDXL 模型）
- 不做任何 volume / 多样性注入，纯 pipeline 采样。

JSON 结构示例：
{
    "bowl": [
        "a photo of a bowl",
        "a close-up photo of a bowl"
    ],
    "truck": [
        "a close-up photo of a truck"
    ]
}

输出结构：
  outputs/<method>_<concept>/
    ├─ eval/
    └─ imgs/
         └─ <prompt_slug>_seed<SEED>_g<GUIDANCE>_s<STEPS>/
               img_000.png ...

例如：
  method  = sd35_inception
  concept = bowl
  prompt  = "a photo of a bowl" -> prompt_slug = a_photo_of_a_bowl
  seed    = 1111
  g       = 5.0
  steps   = 30

  -> outputs/sd35_inception_bowl/imgs/a_photo_of_a_bowl_seed1111_g5.0_s30/img_000.png
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from typing import List

import torch


# -------------------- utils --------------------

def _gb(x): return f"{x/1024**3:.2f} GB"


def print_mem_all(tag: str, devices: List[torch.device]):
    lines = [f"[{tag}]"]
    for d in devices:
        if d.type == "cuda":
            free, total = torch.cuda.mem_get_info(d)
            used = total - free
            lines.append(f"  {d}: used={_gb(used)}, free={_gb(free)}")
        else:
            lines.append(f"  {d}: CPU")
    print("\n".join(lines), flush=True)


def _log(s, debug=True):
    ts = time.strftime("%H:%M:%S")
    if debug:
        print(f"[{ts}] {s}", flush=True)


def _slugify(text: str, maxlen: int = 120) -> str:
    """
    把 prompt 转成文件夹友好的形式：
    "a photo of a bowl" -> "a_photo_of_a_bowl"
    """
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s[:maxlen] if maxlen and len(s) > maxlen else s


def _resolve_model_dir(path: str) -> str:
    """
    尝试把 path 当成本地目录，找到 model_index.json；
    找不到就抛错，让上层用 repo id 的方式加载远程模型。
    """
    p = os.path.abspath(path)
    if os.path.isfile(os.path.join(p, "model_index.json")):
        return p
    for root, _, files in os.walk(p):
        if "model_index.json" in files:
            return root
    raise FileNotFoundError(f"Could not find model_index.json under {path}")


# -------------------- args --------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="SDXL / SDXL-Turbo baseline with JSON-driven prompts (no diversity pipeline)"
    )

    # JSON spec
    ap.add_argument(
        "--spec",
        type=str,
        required=True,
        help="JSON 文件路径，例如 /data2/toby/OSCAR/specs/prompt.json，结构如 {\"bowl\": [\"a photo of a bowl\", ...]}",
    )
    ap.add_argument(
        "--concepts",
        type=str,
        default=None,
        help='要使用的 concept，逗号分隔，如 "bowl,truck"；不填则使用 JSON 里的全部键',
    )

    # 生成参数
    ap.add_argument("--G", type=int, default=16, help="每个 prompt 生成的图片数")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4, help="SDXL-Turbo 常用 4 步")
    ap.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="SDXL-Turbo 官方推荐 0.0 关闭 CFG，如需 CFG 可改",
    )

    # guidance：单个 + 多个
    ap.add_argument("--guidances", type=str, default=None,
                    help='多个 guidance，逗号分隔，如 "0.0,1.0,2.0"，不填则用 --guidance')

    # seed：单个 + 多个
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='多个 seed，逗号分隔，如 "1111,3333,4444,5555,6666,7777"，不填则用 --seed',
    )

    # 模型路径 / repo id
    ap.add_argument(
        "--model-dir",
        type=str,
        default="stabilityai/sdxl-turbo",
        help="本地路径或 HF repo id（如 stabilityai/sdxl-turbo 或 SDXL base 模型）",
    )

    # 输出与设备
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--method",
        type=str,
        default="sdxlturbo",
        help="方法名：用于输出目录前缀，例如 sd35_inception",
    )
    ap.add_argument("--device", type=str, default="cuda:0")

    # negative prompt + 其它
    ap.add_argument("--negative", type=str, default="", help="统一的 negative prompt")

    # 省显存 + 调试
    ap.add_argument("--enable-xformers", action="store_true")
    ap.add_argument("--debug", action="store_true")

    return ap.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()

    # sys.path 注入（项目根目录推断）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 解析 seeds
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [args.seed]

    # 解析 guidances
    if args.guidances is not None:
        guidances = [float(g) for g in args.guidances.split(",") if g.strip() != ""]
    else:
        guidances = [args.guidance]

    # 读取 JSON spec
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # 解析 concepts
    if args.concepts is not None:
        concept_list = [c.strip() for c in args.concepts.split(",") if c.strip() != ""]
    else:
        concept_list = list(spec.keys())

    # 过滤：只保留 JSON 里存在的 concept
    concept_list = [c for c in concept_list if c in spec]
    if len(concept_list) == 0:
        raise ValueError("No valid concept found in spec for --concepts")

    try:
        from diffusers import StableDiffusionXLPipeline

        dev = torch.device(args.device)
        dtype = torch.float16 if dev.type == "cuda" else torch.float32

        _log(f"Device: {dev}", args.debug)
        _log(f"Model dir / repo: {args.model_dir}", args.debug)
        print_mem_all("before-pipeline-call", [dev])

        # ===== 1) 加载 SDXL / SDXL-Turbo（本地 / 远程） =====
        local_only = True
        model_id_or_path = args.model_dir
        try:
            model_id_or_path = _resolve_model_dir(args.model_dir)
        except Exception:
            local_only = False

        _log("Loading StableDiffusionXLPipeline ...", args.debug)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
            local_files_only=local_only,
        )
        if args.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                _log(f"enable_xformers failed: {e}", args.debug)

        pipe = pipe.to(dev)
        pipe.set_progress_bar_config(leave=True)
        print("scheduler:", pipe.scheduler.__class__.__name__)

        # ===== 2) 遍历 concept / prompt / seed / guidance =====
        outputs_root = os.path.join(project_root, "outputs")

        for concept in concept_list:
            prompts = spec[concept]
            if not isinstance(prompts, list):
                continue

            concept_slug = _slugify(str(concept))
            # method_concept
            concept_dirname = f"{args.method}_{concept_slug or 'no_concept'}"
            base_out_dir = (
                args.out.strip()
                if (args.out and len(args.out.strip()) > 0)
                else os.path.join(outputs_root, concept_dirname)
            )
            imgs_root = os.path.join(base_out_dir, "imgs")
            eval_dir = os.path.join(base_out_dir, "eval")
            os.makedirs(imgs_root, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)
            _log(f"Concept={concept} -> base output dir: {imgs_root}", True)

            for prompt in prompts:
                prompt_slug = _slugify(prompt)
                _log(f"  Prompt: {prompt} (slug={prompt_slug})", True)

                for seed in seeds:
                    for g in guidances:
                        run_dir_name = f"{prompt_slug}_seed{seed}_g{g}_s{args.steps}"
                        out_dir = os.path.join(imgs_root, run_dir_name)
                        os.makedirs(out_dir, exist_ok=True)
                        _log(f"    Run dir: {out_dir}", True)

                        generator = (
                            torch.Generator(device=dev)
                            if dev.type == "cuda"
                            else torch.Generator()
                        )
                        generator.manual_seed(seed)

                        _log(
                            f"    Start SDXL sampling (concept={concept}, "
                            f"prompt_slug={prompt_slug}, seed={seed}, g={g}) ...",
                            args.debug,
                        )

                        out = pipe(
                            prompt=prompt,
                            negative_prompt=(
                                None if args.negative == "" else args.negative
                            ),
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.steps,
                            guidance_scale=g,
                            num_images_per_prompt=args.G,
                            generator=generator,
                        )
                        images = out.images  # list of PIL

                        for i, img in enumerate(images):
                            img.save(os.path.join(out_dir, f"img_{i:03d}.png"))

                        _log(
                            f"    Done (concept={concept}, prompt_slug={prompt_slug}, "
                            f"seed={seed}, g={g}). Saved to {out_dir}",
                            True,
                        )

        _log("All concepts/prompts/seeds/guidances done.", True)

    except Exception:
        print("\n=== FATAL ERROR ===\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
