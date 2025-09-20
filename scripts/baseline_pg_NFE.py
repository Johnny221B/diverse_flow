# scripts/baseline_pg_batch.py
# -*- coding: utf-8 -*-
"""
SD3.5 + Particle Guidance 批量版（多 steps × 多 seeds；固定 prompt & cfg）。

输出目录规范：
    outputs/pg_NFE/{eval,imgs}/
    └─ imgs/
       └─ seed{seed}_steps{steps}/
          └─ img_***.png
并在 eval/ 下维护 summary.csv 与 summary.jsonl

相对 main_particle_flow.py 的差异：
- 新增 --steps-list、--seeds-list（逗号分隔）
- 固定 guidance（这里是 --cfg）、prompt；遍历 (steps, seed) 组合
- 每个 steps 重建一次 FlowSampler（因为 steps/cfg 在 FlowSamplerConfig 中）
- 每个 seed 设置独立随机种子
- 统一输出到 pg_NFE 结构
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import torch
from Particle_Guidance.flow_backend import FlowSamplerConfig, FlowSampler
from Particle_Guidance.particle_guidance import PGConfig
from Particle_Guidance.utils import seed_everything

# 允许从仓库根运行
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD3.5 + Particle Guidance (batch)")
    # 基本
    p.add_argument("--model", type=str, required=True, help="Path to SD3.5 directory (Diffusers layout)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative", type=str, default=None)
    p.add_argument("--G", type=int, default=32, help="Images per prompt")
    p.add_argument("--cfg", type=float, default=3.0, help="CFG scale (fixed across the batch)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")

    # 批量
    p.add_argument("--steps-list", type=str, required=True, help="逗号分隔，例如 '20,30,40'")
    p.add_argument("--seeds-list", type=str, required=True, help="逗号分隔，例如 '0,42,1234'")

    # PG 超参
    p.add_argument("--pg-mode", type=str, choices=["ode", "sde"], default="ode")
    p.add_argument("--pg-alpha-scale", type=float, default=30.0, help="alpha_t = alpha_scale * sigma(t)^2")
    p.add_argument("--pg-sigma-a", type=float, default=0.5, help="sigma(t)=a*sqrt(t/(1-t))")
    p.add_argument("--pg-bandwidth", type=float, default=None, help="fixed bandwidth; None -> median trick")

    # 性能
    p.add_argument("--fp16", action="store_true", help="Use float16 where possible")
    p.add_argument("--attention-slicing", action="store_true")

    # 输出根目录
    p.add_argument("--outputs", type=str, default=None, help="输出根目录（默认 <repo_root>/outputs）")
    return p.parse_args()


def ensure_dirs(outputs_root: Path, seed: int, steps: int):
    root = outputs_root / "pg_NFE"
    eval_dir = root / "eval"
    imgs_dir = root / "imgs"
    sub = imgs_dir / f"seed{seed}_steps{steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    return root, eval_dir, sub


def append_log(eval_dir: Path, record: dict):
    csv_path = eval_dir / "summary.csv"
    jsonl_path = eval_dir / "summary.jsonl"
    fieldnames = [
        "timestamp", "method", "steps", "seed", "guidance", "prompt",
        "G", "height", "width", "out_dir", "images"
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in record.items()}
        w.writerow(row)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = build_args()
    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    outputs_root = Path(args.outputs).resolve() if args.outputs else Path(REPO_DIR) / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    steps_list: List[int] = [int(s.strip()) for s in args.steps_list.split(',') if s.strip()]
    seeds_list: List[int] = [int(s.strip()) for s in args.seeds_list.split(',') if s.strip()]

    # PG 固定配置（与 steps 无关）
    pg_cfg_base = dict(
        group_size=args.G,
        mode=args.pg_mode,
        alpha_scale=args.pg_alpha_scale,
        sigma_a=args.pg_sigma_a,
        bandwidth=args.pg_bandwidth,
    )

    method_name = "pg"

    for steps in steps_list:
        # 依 steps（与 cfg/shape）构建 sampler
        pg_cfg = PGConfig(**pg_cfg_base)
        fs_cfg = FlowSamplerConfig(
            model_path=args.model,
            device=str(device),
            dtype=dtype,
            steps=int(steps),
            cfg_scale=float(args.cfg),
            height=int(args.height),
            width=int(args.width),
            enable_attention_slicing=bool(args.attention_slicing),
        )
        sampler = FlowSampler(fs_cfg, pg_cfg)

        for seed in seeds_list:
            # 随机源
            seed_everything(int(seed))

            # 目录
            root, eval_dir, sub = ensure_dirs(outputs_root, seed=int(seed), steps=int(steps))
            print(f"[PG] Output dir: {sub}")

            # 生成
            images = sampler.generate(
                prompt=args.prompt,
                negative_prompt=args.negative,
                num_images=args.G,
                seed=int(seed),
            )

            # 保存
            img_paths = []
            for i, img in enumerate(images):
                p = sub / f"img_{i:03d}.png"
                img.save(p)
                img_paths.append(str(p))
            print(f"[PG] Saved {len(img_paths)} images to: {sub}")

            # 记录
            append_log(
                eval_dir,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": method_name,
                    "steps": int(steps),
                    "seed": int(seed),
                    "guidance": float(args.cfg),
                    "prompt": args.prompt,
                    "G": int(args.G),
                    "height": int(args.height),
                    "width": int(args.width),
                    "out_dir": str(sub),
                    "images": img_paths,
                },
            )

    print("[PG] All runs finished.")


if __name__ == "__main__":
    main()
