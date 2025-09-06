#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把生成好的图片路径汇总成两份 JSON：
- eval/table_coarse_prompt_generated_images_paths.json  # 纯字符串列表（与 coarse example 一致）
- eval/table_dense_prompt_generated_images_paths.json   # 列表[ {prompt, attribute_type, attribute, image_paths: [...]}, ... ]

输入目录规范（与你之前约定一致）：
outputs-root/
  └── {method}_{concept}_CIMDIM/
      ├── dense_imgs/
      │   └── <prompt_slug>/
      │       ├── 00.png ...（任意数量、任意扩展名）
      ├── coarse_imgs/
      │   └── <prompt_slug>/
      │       ├── 00.png ...
      └── eval/   # 本脚本会写入两份 JSON 到这里

使用示例：
python build_eval_jsons.py \
  --outputs-root ./outputs \
  --method pg \
  --concept bus \
  --dense-prompts-json ./COCO-DIMCIM/dense_prompts/bus_dense_prompts.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _find_image_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def _slug_to_prompt(slug: str) -> str:
    # 仅用于兜底展示；真实 prompt 仍以 JSON 元数据为准（若提供）
    return slug.replace("_", " ").strip()


def _ensure_eval_dir(outputs_root: Path, method: str, concept: str) -> Path:
    base = outputs_root / f"{method}_{concept}_CIMDIM"
    eval_dir = base / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------
# Coarse: 写成 “字符串路径的扁平列表”
# ---------------------------
def write_coarse_eval_json(outputs_root: Path, method: str, concept: str) -> Path:
    base = _ensure_eval_dir(outputs_root, method, concept)
    coarse_root = base / "coarse_imgs"

    all_paths: List[str] = []
    if coarse_root.exists():
        for prompt_dir in sorted([p for p in coarse_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for img in _find_image_files(prompt_dir):
                # 写相对 {method}_{concept}_CIMDIM 的路径，方便可移植
                all_paths.append(str(img.relative_to(base).as_posix()))

    outpath = base / "eval" / "table_coarse_prompt_generated_images_paths.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(all_paths, f, ensure_ascii=False, indent=2)
    print(f"[coarse] wrote {outpath} ({len(all_paths)} images).")
    return outpath


# ---------------------------
# Dense: 写成 “对象列表（含 prompt、attribute_*、image_paths）”
# ---------------------------
def _guess_dense_json(outputs_root: Path, concept: str) -> Optional[Path]:
    # 尝试常见位置（可被 --dense-prompts-json 覆盖）
    candidates = [
        outputs_root.parent / "COCO-DIMCIM" / "dense_prompts" / f"{concept}_dense_prompts.json",
        Path.cwd() / "COCO-DIMCIM" / "dense_prompts" / f"{concept}_dense_prompts.json",
        Path.cwd() / f"{concept}_dense_prompts.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_dense_meta(dense_json_path: Path) -> List[Dict]:
    """
    兼容几种常见格式，抽取为：
    [{ "prompt": <dense_prompt>, "attribute_type": ..., "attribute": ... }, ...]
    """
    with open(dense_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: List[Dict] = []
    if isinstance(data, dict):
        if "dense_prompts" in data and isinstance(data["dense_prompts"], list):
            lst = data["dense_prompts"]
        elif "prompts" in data and isinstance(data["prompts"], list):
            lst = data["prompts"]
        else:
            # 极端兜底：收集所有 list 值
            lst = []
            for v in data.values():
                if isinstance(v, list):
                    lst.extend(v)
        for it in lst:
            if isinstance(it, dict):
                entries.append({
                    "prompt": it.get("dense_prompt") or it.get("prompt") or "",
                    "attribute_type": it.get("attribute_type"),
                    "attribute": it.get("attribute"),
                })
            elif isinstance(it, str):
                entries.append({"prompt": it, "attribute_type": None, "attribute": None})
    elif isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                entries.append({
                    "prompt": it.get("dense_prompt") or it.get("prompt") or "",
                    "attribute_type": it.get("attribute_type"),
                    "attribute": it.get("attribute"),
                })
            elif isinstance(it, str):
                entries.append({"prompt": it, "attribute_type": None, "attribute": None})
    return entries


def write_dense_eval_json(
    outputs_root: Path,
    method: str,
    concept: str,
    dense_prompts_json: Optional[Path] = None
) -> Path:
    base = _ensure_eval_dir(outputs_root, method, concept)
    dense_root = base / "dense_imgs"

    # 读 meta（若能找到）
    dense_json_path = dense_prompts_json or _guess_dense_json(outputs_root, concept)
    dense_meta: Optional[List[Dict]] = _load_dense_meta(dense_json_path) if dense_json_path and dense_json_path.exists() else None

    def _norm(s: str) -> str:
        return (s or "").lower().strip().replace(" ", "_")

    rows: List[Dict] = []

    # 为了顺序稳定：若有 meta，按其顺序；否则按文件夹名
    if dense_meta:
        # 先建立从 slug -> 图片路径列表 的索引
        slug_to_imgs: Dict[str, List[str]] = {}
        if dense_root.exists():
            for prompt_dir in sorted([p for p in dense_root.iterdir() if p.is_dir()], key=lambda p: p.name):
                imgs = [str(p.relative_to(base).as_posix()) for p in _find_image_files(prompt_dir)]
                slug_to_imgs[prompt_dir.name] = imgs

        for m in dense_meta:
            slug = _norm(m.get("prompt", ""))
            imgs = slug_to_imgs.get(slug, [])
            rows.append({
                "prompt": m.get("prompt", _slug_to_prompt(slug)),
                "attribute_type": m.get("attribute_type"),
                "attribute": m.get("attribute"),
                "image_paths": imgs,
            })
    else:
        # 没有 meta 的兜底：仅从目录恢复 prompt 文本
        if dense_root.exists():
            for prompt_dir in sorted([p for p in dense_root.iterdir() if p.is_dir()], key=lambda p: p.name):
                imgs = [str(p.relative_to(base).as_posix()) for p in _find_image_files(prompt_dir)]
                rows.append({
                    "prompt": _slug_to_prompt(prompt_dir.name),
                    "attribute_type": None,
                    "attribute": None,
                    "image_paths": imgs,
                })

    outpath = base / "eval" / "table_dense_prompt_generated_images_paths.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    total_imgs = sum(len(r["image_paths"]) for r in rows)
    print(f"[dense]  wrote {outpath} ({len(rows)} prompts, {total_imgs} images).")
    return outpath


def main():
    ap = argparse.ArgumentParser(description="Aggregate generated image paths to eval JSONs.")
    ap.add_argument("--outputs-root", required=True, help="根输出目录（包含 {method}_{concept}_CIMDIM/）")
    ap.add_argument("--method", required=True, help="如 pg")
    ap.add_argument("--concept", required=True, help="如 bus")
    ap.add_argument("--dense-prompts-json", default=None, help="可选：{concept}_dense_prompts.json 路径，用于填充 attribute 信息")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    dense_json = Path(args.dense_prompts_json) if args.dense_prompts_json else None

    write_coarse_eval_json(outputs_root, args.method, args.concept)
    write_dense_eval_json(outputs_root, args.method, args.concept, dense_json)


if __name__ == "__main__":
    main()
