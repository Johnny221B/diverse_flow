#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vendi-Score and FID evaluator (Final Verified Version 3).
- Computes Vendi Score using the official package.
- Optionally computes FID using the piq library with the correct API usage.
- Both metrics reuse the same torchvision Inception v3 model from cache.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    print("Warning: piq library not found. FID calculation will not be available.")
    print("Please run: pip install piq")
    piq = None

from vendi_score import image_utils, vendi

# -------------------------------
# IO helpers
# -------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def list_images(dir_path: Path) -> List[Path]:
    """Finds all image files in a directory and its subdirectories."""
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def load_pils(paths: List[Path]) -> List[Image.Image]:
    """Loads a list of image paths into PIL images, skipping corrupted ones."""
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            print(f"Warning: Could not load image {p}, skipping.")
            pass
    return imgs

# -------------------------------
# Compat fallback for modern torchvision
# -------------------------------
def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    """Fallback for Vendi Score calculation on newer torchvision versions."""
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

# -------------------------------
# Core Vendi evaluator
# -------------------------------
def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    """Computes both pixel-based and embedding-based Vendi Scores."""
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
# Final, Final Corrected FID Calculation Function
# -------------------------------
def compute_fid_score(real_images_path: str, fake_images_path: str, device: str = "cuda") -> float:
    """Computes FID score using the correct, two-step piq library API."""
    if piq is None:
        print("FID calculation skipped because piq is not installed.")
        return float('nan')

    print(f"\nCalculating FID score...")
    print(f"  Real images path: {real_images_path}")
    print(f"  Fake images path: {fake_images_path}")

    # 1. Build the InceptionV3 feature extractor.
    feature_extractor = InceptionV3().to(torch.device(device))
   
    # Helper function to extract features for a folder of images
    def extract_features(image_paths, extractor, device):
        transform = T.Compose([
            T.Resize(299),
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

        dataset = ImagePathDataset(image_paths)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
       
        all_features = []
        with torch.inference_mode():
            for images in tqdm(loader, desc=f"Extracting features from {Path(image_paths[0]).parent.name}"):
                features = extractor(images.to(device))
                if isinstance(features, list):
                    features = features[0]
               
                # --- START: THE FIX ---
                # The extractor returns (N, C, H, W) features. We need to flatten the spatial dims.
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                # --- END: THE FIX ---

                all_features.append(features.cpu())
       
        return torch.cat(all_features, dim=0)

    # 2. Extract features for both real and fake images
    real_paths = list_images(Path(real_images_path))
    fake_paths = list_images(Path(fake_images_path))
   
    real_features = extract_features(real_paths, feature_extractor, device)
    fake_features = extract_features(fake_paths, feature_extractor, device)

    # 3. Instantiate the FID metric and compute the score from the features
    fid_metric = piq.FID()
    fid_value = fid_metric(real_features, fake_features)
   
    return float(fid_value)

# -------------------------------
# CLI (Main Function)
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vendi Score and FID evaluator for image folders")
    ap.add_argument("--images", type=str, required=True, help="Directory containing generated images")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'; used for Inception embeddings")
    ap.add_argument("--per-subdir", action="store_true", help="Evaluate each first-level subdirectory separately (FID does not support this mode)")
    ap.add_argument("--save", type=str, default=None, help="Path to save JSON report")
   
    ap.add_argument("--compute-fid", action="store_true", help="Also compute FID score")
    ap.add_argument("--real-images", type=str, default=None, help="Directory with real images (required for FID)")
    args = ap.parse_args()

    if args.compute_fid and not args.real_images:
        raise SystemExit("Error: --real-images path is required when using --compute-fid.")
    if args.compute_fid and args.per_subdir:
        print("Warning: FID calculation does not support --per-subdir mode and will be skipped.")

    root = Path(args.images)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    report: Dict[str, object] = {"root": str(root), "device": args.device, "results": {}}

    if args.per_subdir:
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        for d in subdirs:
            print(f"\n--- Processing subdir: {d.name} ---")
            paths = list_images(d)
            if not paths: continue
            imgs = load_pils(paths)
            if not imgs: continue
            scores = compute_vendi_for_images(imgs, device=args.device)
            report["results"][d.name] = {"num_images": len(imgs), **scores}
    else:
        print(f"\n--- Processing all images in: {root} ---")
        paths = list_images(root)
        if not paths: raise SystemExit("No images found.")
        imgs = load_pils(paths)
        if not imgs: raise SystemExit("Could not load any valid images.")
       
        vendi_scores = compute_vendi_for_images(imgs, device=args.device)
        report["results"]["ALL"] = {"num_images": len(imgs), **vendi_scores}

        if args.compute_fid:
            fid_score = compute_fid_score(args.real_images, args.images, device=args.device)
            report["results"]["ALL"]["fid"] = fid_score

    print("\n--- FINAL REPORT ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.save:
        outp = Path(args.save)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nReport saved to: {args.save}")

if __name__ == "__main__":
    main()