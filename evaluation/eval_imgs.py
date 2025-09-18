#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Evaluation Tool
- For each generated folder, compute:
  * Vendi Score (pixel & inception embedding)
  * FID (against a shared real/reference folder)
  * KID (Kernel Inception Distance; unbiased MMD^2 with cubic polynomial kernel)
  * CLIP Score (using local OpenAI CLIP JIT; no model fallback)
  * 1 - MS-SSIM (diversity)
  * BRISQUE (quality)
- Reuses: one InceptionV3 (for FID/KID) and cached features of the real set across all folders.

Dependencies:
  torch, torchvision, PIL, numpy, tqdm, skimage, opencv-python
  vendi_score (image_utils, vendi)
  piq (FID, BRISQUELoss); optional but recommended
  open_clip_torch (optional; only for tokenize if available)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# Optional deps
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

try:
    import open_clip  # only used for tokenize if available
except ImportError:
    open_clip = None

# For MS-SSIM diversity
from skimage.metrics import structural_similarity as ssim
import cv2

# Vendi score
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
            print(f"Warning: Could not load image {p}, skipping.")
    return imgs

def _save_per_folder_json(
    gen_dir: Path,
    metrics: Dict[str, object],
    real_dir: Path,
    device: torch.device,
    eval_dir_name: str = "eval"
):
    """
    Save per-folder JSON to:  <base>/<eval_dir_name>/<name>.json
    - base = the directory before the first 'imgs' in gen_dir
    - name = the immediate child right after 'imgs'
    """
    parts = gen_dir.parts
    if "imgs" in parts:
        idx = parts.index("imgs")
        base = Path(*parts[:idx]) if idx > 0 else Path(".")
        eval_dir = base / eval_dir_name
        fname_stem = parts[idx + 1] if len(parts) > idx + 1 else gen_dir.name
    else:
        eval_dir = gen_dir.parent / eval_dir_name
        fname_stem = gen_dir.name

    payload = {
        "real_root": str(real_dir),
        "device": str(device),
        "results": {gen_dir.name: metrics},
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / f"{fname_stem}.json"
    out_file.write_text(json.dumps(_nan_to_none(payload), indent=2, ensure_ascii=False))
    print(f"Saved per-folder JSON: {out_file}")

# -------------------------------
# Vendi: embedding fallback (newer torchvision)
# -------------------------------
def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    import torchvision
    from torchvision.models import Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    model = torchvision.models.inception_v3(weights=weights, aux_logits=True, transform_input=True)
    model.fc = nn.Identity()
    model.eval().to(device)

    feats: List[torch.Tensor] = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            batch = [tfm(img) for img in imgs[i : i + bs]]
            x = torch.stack(batch, dim=0).to(device)
            f = model(x)
            if isinstance(f, (tuple, list)):
                f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
    X = torch.cat(feats, dim=0).numpy()
    return float(vendi.score_dual(X, normalize=False))

def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    pix_vs = float(image_utils.pixel_vendi_score(imgs))
    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device))
    except Exception as e:
        if "init_weights" in str(e) or "weights" in str(e):
            print("Vendi_score default failed, using torchvision compatibility fallback...")
            emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
        else:
            raise
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# -------------------------------
# Inception features (for FID & KID)
# -------------------------------
def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        print("Warning: piq not installed; FID/KID will be skipped.")
        return None
    return InceptionV3().to(device)

def _extract_inception_features(
    image_paths: List[Path],
    extractor: nn.Module,
    device: torch.device,
    num_workers: int = 4,
    batch_size: int = 32,
) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(299),
        T.ToTensor(),
    ])

    class ImagePathDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            path = self.files[i]
            img = Image.open(path).convert('RGB')
            return transform(img)

    if not image_paths:
        return torch.empty(0, 2048)

    dataset = ImagePathDataset(image_paths)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=("cuda" in device.type)
    )

    all_features = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=f"Extracting features: {Path(image_paths[0]).parent.name}"):
            feats = extractor(images.to(device))
            if isinstance(feats, list):
                feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0) if all_features else torch.empty(0, 2048)

def compute_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> Optional[float]:
    if piq is None:
        return None
    if real_features.numel() == 0 or fake_features.numel() == 0:
        return None
    fid_metric = piq.FID()
    value = fid_metric(real_features, fake_features)
    return float(value)

# -------------------------------
# === KID (Kernel Inception Distance) ===
# -------------------------------
def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Polynomial kernel k(x,y) = (x^T y / d + 1)^3  where d = feature_dim.
    x: [m, d], y: [n, d] -> returns [m, n]
    """
    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** 3

def compute_kid_from_features(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subset_size: int = 1000,
    n_subsets: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[float, float]]:
    """
    Unbiased KID estimate (mean ± std over subsets) using polynomial kernel of degree 3.
    Returns (kid_mean, kid_std) or None if inputs are invalid.
    """
    if real_features is None or fake_features is None: return None
    if real_features.numel() == 0 or fake_features.numel() == 0: return None
    if real_features.shape[1] != fake_features.shape[1]:
        print("KID: feature dim mismatch.")
        return None

    m_full, n_full = real_features.shape[0], fake_features.shape[0]
    if m_full < 2 or n_full < 2:
        return None

    s = min(subset_size, m_full, n_full)
    if s < 2:
        return None

    if rng is None:
        rng = np.random.default_rng(0)

    X = real_features.to(torch.float64)
    Y = fake_features.to(torch.float64)
    stats = []

    for _ in range(max(1, n_subsets)):
        ridx = rng.choice(m_full, size=s, replace=False)
        fidx = rng.choice(n_full, size=s, replace=False)
        xs = X[ridx]  # [s,d]
        ys = Y[fidx]  # [s,d]

        k_xx = _poly_kernel(xs, xs)
        k_yy = _poly_kernel(ys, ys)
        k_xy = _poly_kernel(xs, ys)

        # Unbiased MMD^2 estimator
        sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (s * (s - 1))
        sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (s * (s - 1))
        sum_xy = k_xy.mean()

        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        stats.append(float(mmd2.item()))

    stats = np.array(stats, dtype=np.float64)
    kid_mean = float(stats.mean())
    kid_std  = float(stats.std(ddof=1)) if stats.size > 1 else 0.0
    return kid_mean, kid_std

# -------------------------------
# CLIP score (use same OpenAI JIT as pipeline; no model fallback)
# -------------------------------
def load_clip_from_jit(jit_path: Path):
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    return model

def calculate_clip_score(
    image_folder: Path,
    model: Any,
    preprocess: T.Compose,                 # kept for signature compatibility (unused)
    device: torch.device,
    batch_size: int,
    tokenizer=None,                        # if None, try open_clip.tokenize then clip.tokenize
    is_openclip: bool = False,             # kept for signature compatibility (unused)
    clip_image_size: int = 224,            # align with pipeline cfg.clip_image_size
) -> Optional[float]:
    """
    Compute CLIPScore using the SAME image preprocessing as the pipeline (CLIPWrapper).
    """
    prompt = image_folder.name.replace('_', ' ').replace('-', ' ')

    if tokenizer is None:
        tok = None
        try:
            import open_clip
            tok = open_clip.tokenize
        except Exception:
            try:
                import clip as openai_clip
                tok = openai_clip.tokenize
            except Exception:
                tok = None
        tokenizer = tok

    image_paths = list(image_folder.rglob("*.jpg")) + list(image_folder.rglob("*.png"))
    if not image_paths:
        return None
    if tokenizer is None:
        print("No tokenizer available (open_clip or clip). Skipping CLIP Score.")
        return None

    with torch.inference_mode():
        text_tokens = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    to_tensor = T.ToTensor()
    all_scores: List[float] = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"CLIP Score: {image_folder.name}"):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(to_tensor(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not imgs:
                continue

            x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)  # [B,3,H,W]
            if x.shape[-2:] != (clip_image_size, clip_image_size):
                x = F.interpolate(
                    x, size=(clip_image_size, clip_image_size),
                    mode="bilinear", align_corners=False, antialias=True
                )
            x = (x - mean) / std

            image_features = model.encode_image(x)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
            sim = (100.0 * (image_features @ text_features.T)).squeeze(-1).detach().cpu().tolist()
            all_scores.extend(sim)

    return float(np.mean(all_scores)) if all_scores else None

# -------------------------------
# 1 - MS-SSIM (diversity)
# -------------------------------
def calculate_ms_ssim_diversity(image_folder: Path, max_pairs: int = 100) -> Optional[float]:
    paths = list(image_folder.rglob("*.jpg")) + list(image_folder.rglob("*.png"))
    if len(paths) < 2:
        return None

    rng = np.random.default_rng(0)
    if len(paths) > int(np.sqrt(max_pairs * 2)) + 1 and max_pairs > 0:
        paths = rng.choice(paths, size=int(np.sqrt(max_pairs * 2)) + 1, replace=False)

    pairs = [(paths[i], paths[j]) for i in range(len(paths)) for j in range(i+1, len(paths))]
    if len(pairs) > max_pairs and max_pairs > 0:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    scores = []
    for p1, p2 in tqdm(pairs, desc=f"MS-SSIM: {image_folder.name}"):
        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        if img1 is None or img2 is None:
            continue
        h1, w1, _ = img1.shape
        img2 = cv2.resize(img2, (w1, h1))

        min_dim = min(h1, w1)
        win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)
        if win_size < 3:
            continue
        score = ssim(img1, img2, data_range=255, channel_axis=-1, win_size=win_size)
        scores.append(score)

    if not scores:
        return None
    return float(1.0 - np.mean(scores))

# -------------------------------
# BRISQUE
# -------------------------------
def calculate_brisque_quality(image_folder: Path, device: torch.device) -> Optional[float]:
    if piq is None:
        print("piq not installed; BRISQUE will be skipped.")
        return None

    image_paths = list(image_folder.rglob("*.jpg")) + list(image_folder.rglob("*.png"))
    if not image_paths:
        return None

    transform = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none')

    scores = []
    for p in tqdm(image_paths, desc=f"BRISQUE: {image_folder.name}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            s = metric(x)
        scores.append(float(s.item()))
    if not scores:
        return None
    return float(np.mean(scores))

# -------------------------------
# Utilities
# -------------------------------
def _nan_to_none(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified evaluator for multiple generated folders against a shared real/reference set.")
    ap.add_argument("--gen", nargs="+", required=True, help="One or more generated image folders.")
    ap.add_argument("--real", required=True, help="Reference (real) image folder for FID/KID.")
    ap.add_argument("--out", required=True, help="Output JSON path for results.")

    ap.add_argument("--device", type=str, default="cuda:0", help='Device, e.g., "cpu", "cuda", "cuda:0"')
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"), help="Local OpenAI CLIP JIT path.")
    ap.add_argument("--clip-image-size", type=int, default=224, help="Match pipeline CLIP image size (e.g., 224 or 336).")
    ap.add_argument("--max-pairs", type=int, default=100, help="Max pairs for MS-SSIM diversity (controls cost).")
    ap.add_argument("--eval_dir_name", type=str, default="eval")

    # === KID ===
    ap.add_argument("--kid-subset-size", type=int, default=64, help="Subset size per KID estimate (min(#real,#fake,#this)).")
    ap.add_argument("--kid-subsets", type=int, default=20, help="Number of random subsets for KID (0 to disable).")

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    gen_dirs = [Path(p) for p in args.gen]
    real_dir = Path(args.real)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    for d in gen_dirs:
        if not d.exists() or not d.is_dir():
            raise SystemExit(f"Generated folder not found or not dir: {d}")
    if not real_dir.exists() or not real_dir.is_dir():
        raise SystemExit(f"Real folder not found or not dir: {real_dir}")

    # --- Build shared tools/models ---
    # Inception for FID/KID
    inception = _build_inception_extractor(device)

    # Precompute real features once (for FID/KID)
    real_features = None
    if inception is not None:
        real_paths = list_images(real_dir)
        if real_paths:
            real_features = _extract_inception_features(
                real_paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size
            )
        else:
            print("No real images found; FID/KID will be skipped.")
            inception = None

    # CLIP: use the SAME OpenAI JIT as pipeline; no model fallback
    clip_model = None
    jit_path = Path(args.clip_jit) if args.clip_jit else None
    if jit_path and jit_path.exists():
        try:
            clip_model = load_clip_from_jit(jit_path).to(device)
            print(f"Loaded CLIP JIT from {jit_path}")
        except Exception as e:
            print(f"Failed to load CLIP JIT: {e}")
    else:
        print(f"CLIP JIT not found at {jit_path}; CLIP Score will be skipped.")

    # --- Evaluate each generated folder ---
    results: Dict[str, Dict[str, Optional[float]]] = {}
    summary = {
        "real_root": str(real_dir),
        "device": str(device),
        "results": results,
    }

    for gen_dir in gen_dirs:
        print(f"\n=== Evaluating: {gen_dir.name} ===")
        paths = list_images(gen_dir)
        if not paths:
            print(f"No images in {gen_dir}, skipping.")
            continue

        # Vendi
        imgs = load_pils(paths)
        if not imgs:
            print(f"Could not load any images from {gen_dir}, skipping.")
            continue
        vendi_scores = compute_vendi_for_images(imgs, device=str(device))

        # FID/KID features for this folder
        fid_value = None
        kid_mean, kid_std = None, None
        fake_features = None
        if inception is not None and real_features is not None and real_features.numel() > 0:
            fake_features = _extract_inception_features(
                paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size
            )
            # FID
            fid_value = compute_fid_from_features(real_features, fake_features)
            # KID (only if enabled)
            if args.kid_subsets > 0 and fake_features is not None and fake_features.numel() > 0:
                rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
                kid_ret = compute_kid_from_features(
                    real_features, fake_features,
                    subset_size=args.kid_subset_size,
                    n_subsets=args.kid_subsets,
                    rng=rng
                )
                if kid_ret is not None:
                    kid_mean, kid_std = kid_ret

        # CLIP Score (only if JIT available)
        clip_score = None
        if clip_model is not None:
            clip_score = calculate_clip_score(
                gen_dir, clip_model, None, device, args.batch_size,
                clip_image_size=args.clip_image_size
            )

        # 1 - MS-SSIM (diversity)
        msssim_div = calculate_ms_ssim_diversity(gen_dir, max_pairs=args.max_pairs)

        # BRISQUE
        brisque_quality = calculate_brisque_quality(gen_dir, device=device)

        # Collect
        results[gen_dir.name] = {
            "num_images": len(imgs),
            "vendi_pixel": vendi_scores.get("vendi_pixel"),
            "vendi_inception": vendi_scores.get("vendi_inception"),
            "fid": fid_value,
            # === KID ===
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "clip_score": clip_score,
            "one_minus_ms_ssim": msssim_div,
            "brisque": brisque_quality,
        }

        _save_per_folder_json(
            gen_dir=gen_dir,
            metrics=results[gen_dir.name],
            real_dir=real_dir,
            device=device,
            eval_dir_name=args.eval_dir_name
        )

    # Save JSON (sanitize NaN/Inf)
    sanitized = _nan_to_none(summary)
    out_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False))
    print(f"\nSaved results to: {out_path}")

if __name__ == "__main__":
    main()