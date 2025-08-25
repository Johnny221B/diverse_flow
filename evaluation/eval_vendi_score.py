#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vendi-Score evaluator (official package) with robust torchvision fix.

- Uses `pip install vendi_score[images]` and the official `image_utils` helpers.
- If your torchvision is new (>=0.13) and you see the
  "init_weights expected False but got True" error, this script
  automatically falls back to a **compat path** that builds Inception v3
  with the modern `weights=` API and computes Vendi via `vendi.score_dual`.
- Supports whole-folder or per-subfolder evaluation; outputs JSON.
"""

# python eval_vendi_score.py --images /mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourMethod_a_cozy_cabin_in_a_snowy_forest --device cuda --save /mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourMethod_a_cozy_cabin_in_a_snowy_forest/vendi.json
# python eval_vendi_score.py --images /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_a_cozy_cabin_in_a_snowy_forest --device cuda --save /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_a_cozy_cabin_in_a_snowy_forest/vendi.json

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vendi_score import image_utils, vendi

# -------------------------------
# IO helpers
# -------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def load_pils(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs

# -------------------------------
# Compat fallback for modern torchvision
# -------------------------------

def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    """Compute Vendi using torchvision Inception v3 (weights API) -> pooled 2048-d -> vendi.score_dual.
    This bypasses vendi_score.image_utils.get_inception to avoid the `init_weights` conflict.
    """
    import torchvision
    from torchvision.models import Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    # IMPORTANT: With weights!=None, torchvision expects aux_logits=True (see wrapper checks).
    model = torchvision.models.inception_v3(weights=weights, aux_logits=True, transform_input=True)
    # Replace classifier with identity to expose pooled features (2048-d)
    model.fc = nn.Identity()
    model.eval().to(device)

    feats: List[torch.Tensor] = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            batch = [tfm(img) for img in imgs[i : i + bs]]
            x = torch.stack(batch, dim=0).to(device)
            f = model(x)  # [B, 2048] because fc=Identity
            if isinstance(f, (tuple, list)):
                f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
    X = torch.cat(feats, dim=0).numpy()
    return float(vendi.score_dual(X, normalize=False))

# -------------------------------
# Core evaluators (official vendi_score + fallback)
# -------------------------------

def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    pix_vs = float(image_utils.pixel_vendi_score(imgs))

    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device))
    except Exception as e:
        # torchvision >=0.13 changed API (weights, init_weights). If official path breaks, use fallback.
        if "init_weights" in str(e) or "weights" in str(e):
            emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
        else:
            raise

    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Vendi Score (official package) for image folders (with torchvision fix)")
    ap.add_argument("--images", type=str, required=True, help="目录：包含图片或若干子目录")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' 或 'cpu'；用于 Inception 嵌入")
    ap.add_argument("--per-subdir", action="store_true", help="每个一级子目录单独评测")
    ap.add_argument("--save", type=str, default=None, help="保存 JSON 报告的路径")
    args = ap.parse_args()

    root = Path(args.images)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    report: Dict[str, object] = {"root": str(root), "device": args.device, "results": {}}

    if args.per_subdir:
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        for d in subdirs:
            paths = list_images(d)
            imgs = load_pils(paths)
            if not imgs:
                continue
            scores = compute_vendi_for_images(imgs, device=args.device)
            report["results"][d.name] = {"num_images": len(imgs), **scores}
    else:
        paths = list_images(root)
        imgs = load_pils(paths)
        if not imgs:
            raise SystemExit("No images found.")
        scores = compute_vendi_for_images(imgs, device=args.device)
        report["results"]["ALL"] = {"num_images": len(imgs), **scores}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.save:
        outp = Path(args.save)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
