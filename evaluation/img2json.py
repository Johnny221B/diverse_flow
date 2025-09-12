#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DIMCIM JSONs from generated images.

Outputs (written to <outputs>/<method>_<concept>_CIMDIM/eval/):
- table_coarse_prompt_generated_images_paths.json   # list[str]  (exactly as the template)
- table_dense_prompt_generated_images_paths.json    # list[ {prompt, attribute_type, attribute, image_paths[]} ]

Assumed layout (same as generation phase):
<outputs>/<method>_<concept>_CIMDIM/
  ├─ coarse_imgs/<slug(coarse_prompt)>/00.png..19.png
  └─ dense_imgs/<slug(dense_prompt)>/00.png..19.png
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def prompt_slug(prompt: str) -> str:
    """与生成时一致：去标点、去连字符、空格->下划线、保留大小写。"""
    s = prompt.strip()
    s = re.sub(r"[^\w\s-]", "", s)   # 移除标点但保留字母/数字/空格/-
    s = s.replace("-", "")           # 连字符去掉
    s = re.sub(r"\s+", "_", s)       # 空格 -> 下划线
    return s.strip("_")

def list_images(dirp: Path, absolute: bool = True) -> List[str]:
    if not dirp.exists():
        return []
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(sorted((p for p in dirp.glob(f"*{ext}")), key=lambda x: x.name))
    if absolute:
        return [str(p.resolve()) for p in files]
    # 若想要相对 <method>_<concept>_CIMDIM/，可改 absolute=False
    return [str(p.relative_to(dirp.parents[2]).as_posix()) for p in files]

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_dimcim_jsons(outputs_root: Path, method: str, concept: str, prompts_json: Path,
                       use_absolute_paths: bool = True, skip_missing: bool = True):
    base = outputs_root / f"{method}_{concept}_CIMDIM"
    coarse_root = base / "coarse_imgs"
    dense_root  = base / "dense_imgs"
    eval_dir    = base / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(prompts_json)
    pairs = data.get("coarse_dense_prompts", [])
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("prompts JSON 缺少 'coarse_dense_prompts' 列表。")

    # -------- Coarse: 按模板要求 => 纯 list[str] --------
    coarse_paths: List[str] = []
    for row in pairs:
        c_prompt = row.get("coarse_prompt", "")
        if not c_prompt:
            continue
        c_dir = coarse_root / prompt_slug(c_prompt)
        imgs = list_images(c_dir, absolute=use_absolute_paths)
        if imgs or not skip_missing:
            coarse_paths.extend(imgs)

    save_json(coarse_paths, eval_dir / "bus_coarse_prompt_generated_images_paths.json")

    # -------- Dense: list[dict]，附带属性信息 --------
    dense_entries: List[Dict[str, Any]] = []
    for row in pairs:
        dlist = row.get("dense_prompts", [])
        if not isinstance(dlist, list):
            continue
        for d in dlist:
            d_prompt = d.get("dense_prompt", "")
            if not d_prompt:
                continue
            d_dir  = dense_root / prompt_slug(d_prompt)
            imgs   = list_images(d_dir, absolute=use_absolute_paths)
            if imgs or not skip_missing:
                dense_entries.append({
                    "prompt": d_prompt,
                    "attribute_type": d.get("attribute_type", None),
                    "attribute": d.get("attribute", None),
                    "image_paths": imgs,
                })

    save_json(dense_entries, eval_dir / "bus_dense_prompt_generated_images_paths.json")

    print(f"[OK] Coarse JSON -> {eval_dir / 'bus_coarse_prompt_generated_images_paths.json'}  (images={len(coarse_paths)})")
    print(f"[OK] Dense  JSON -> {eval_dir / 'bus_dense_prompt_generated_images_paths.json'}   (prompts={len(dense_entries)})")

def main():
    ap = argparse.ArgumentParser(description="Assemble DIMCIM JSONs (coarse & dense).")
    ap.add_argument("--outputs", required=True, help="outputs 根目录（包含 <method>_<concept>_CIMDIM/）")
    ap.add_argument("--method",  required=True, help="方法名，如 pg / cads / dpp")
    ap.add_argument("--concept", required=True, help="概念名，如 bus")
    ap.add_argument("--prompts-json", required=True, help="DIMCIM prompts 表（第三个文件）")
    ap.add_argument("--relative", action="store_true", help="写相对路径（默认写绝对路径）")
    ap.add_argument("--no-skip-missing", action="store_true", help="即使该 prompt 没有图片也写记录")
    args = ap.parse_args()

    build_dimcim_jsons(
        outputs_root=Path(args.outputs),
        method=args.method,
        concept=args.concept,
        prompts_json=Path(args.prompts-json) if hasattr(args, 'prompts-json') else Path(args.prompts_json),
        use_absolute_paths=(not args.relative),
        skip_missing=(not args.no_skip_missing),
    )

if __name__ == "__main__":
    main()