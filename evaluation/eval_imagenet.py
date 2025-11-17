#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flat-folder evaluator for diversity/quality metrics + FID + CLIP.

Folder layout example:

root/
  ├── cads/         # 400 images
  ├── dpp/          # 400 images
  ├── pg/           # 400 images
  ├── standardFM/   # 400 images
  └── eval/         # this script will create this folder & CSV

Usage example:

  python eval_flat_methods_fid_clip.py \
      --root /path/to/imagenet_400 \
      --methods cads dpp pg standardFM \
      --device cuda:0 \
      --max-pairs 200 \
      --real-dir /path/to/real_400 \
      --fid-mode piq \
      --clip-jit ~/.cache/clip/ViT-B-32.pt \
      --clip-image-size 224
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# ---------- optional deps ----------
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

try:
    from pytorch_msssim import ms_ssim as ms_ssim_torch
except ImportError:
    ms_ssim_torch = None

try:
    import cleanfid
except ImportError:
    cleanfid = None

try:
    from vendi_score import image_utils, vendi
except ImportError:
    image_utils = None
    vendi = None

try:
    import open_clip
except ImportError:
    open_clip = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 和你原来一样的 pattern，方便从文件名里抽 prompt
META_RE = re.compile(
    r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$"
)


# ------------------ small utils ------------------

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])


def mean_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Mean and (1-alpha) CI using normal approximation (mean ± 1.96 * std / sqrt(n)).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    z = 1.96
    half = z * std / np.sqrt(arr.size)
    return mean, mean - half, mean + half


def parse_prompt_from_stem(stem: str) -> str:
    """
    从文件名（不含扩展名）里恢复 prompt：
    - 优先按 <prompt>_seedXXXX_gY_sZ 解析
    - 否则用整段名字，把 '_' 和 '-' 换成空格
    """
    m = META_RE.match(stem)
    if m:
        prompt = m.groupdict()["prompt"]
    else:
        prompt = stem
    prompt = prompt.replace("_", " ").replace("-", " ")
    return prompt


# ------------------ Vendi ------------------

def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    import torchvision
    from torchvision.models import Inception_V3_Weights

    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    model = torchvision.models.inception_v3(
        weights=weights, aux_logits=True, transform_input=True
    )
    model.fc = nn.Identity()
    model.eval().to(device)

    feats = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            x = torch.stack([tfm(img) for img in imgs[i:i + bs]], dim=0).to(device)
            f = model(x)
            if isinstance(f, (tuple, list)):
                f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
    X = torch.cat(feats, dim=0).numpy()
    return float(vendi.score_dual(X, normalize=False))


def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    if image_utils is None or vendi is None:
        print("vendi_score not installed; skip Vendi.")
        return {"vendi_pixel": None, "vendi_inception": None}

    pix_vs = float(image_utils.pixel_vendi_score(imgs))
    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device))
    except Exception as e:
        if "init_weights" in str(e) or "weights" in str(e):
            print("Vendi default failed, use torchvision fallback...")
            emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
        else:
            raise
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}


# ------------------ Inception features for FID ------------------

def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        print("Warning: piq not installed; FID(piq) will be skipped.")
        return None
    return InceptionV3().to(device)


def _make_fid_transform():
    return T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(299),
        T.ToTensor(),
    ])


def _extract_inception_features(paths: List[Path], extractor: nn.Module,
                                device: torch.device,
                                num_workers=4, batch_size=32) -> torch.Tensor:
    transform = _make_fid_transform()

    class D(torch.utils.data.Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            img = Image.open(self.files[i]).convert("RGB")
            return transform(img)

    if not paths:
        return torch.empty(0, 2048)

    loader = torch.utils.data.DataLoader(
        D(paths),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=("cuda" in device.type),
    )
    feats_all = []
    with torch.inference_mode():
        for imgs in tqdm(loader, desc=f"Extracting Inception feats ({Path(paths[0]).parent.name})"):
            feats = extractor(imgs.to(device))
            if isinstance(feats, list):
                feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)
            feats_all.append(feats.cpu())
    return torch.cat(feats_all, dim=0) if feats_all else torch.empty(0, 2048)


def compute_fid_from_features(real_features: torch.Tensor,
                              fake_features: torch.Tensor) -> Optional[float]:
    if piq is None or real_features.numel() == 0 or fake_features.numel() == 0:
        return None
    return float(piq.FID()(real_features, fake_features))


def compute_fid_cleanfid(real_dir: Path, gen_dir: Path, mode="clean") -> Optional[float]:
    if cleanfid is None:
        print("cleanfid not installed; skip clean-FID.")
        return None
    try:
        return float(cleanfid.compute_fid(str(real_dir), str(gen_dir),
                                          mode=mode, model_name="inception_v3"))
    except Exception as e:
        print(f"clean-fid failed on {gen_dir}: {e}")
        return None


# ------------------ CLIP Score ------------------

def load_clip_from_jit(jit_path: Path, image_size: int = 224):
    """
    Load OpenAI CLIP JIT and return (model, preprocess).
    """
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    preprocess = T.Compose([
        T.Resize((image_size, image_size),
                 interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    return model, preprocess


def calculate_clip_scores_flat(
    image_paths: List[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> List[float]:
    """
    对 flat 文件夹的所有图片算 text-image CLIP score：
      - prompt 从文件名 stem 解析 (parse_prompt_from_stem)
      - 每张图有自己的 prompt
    返回：所有图片的 score 列表
    """
    if not image_paths:
        return []

    # tokenizer: open_clip 优先，其次 openai-clip
    tok = None
    try:
        import open_clip as _oc
        tok = _oc.tokenize
    except Exception:
        try:
            import clip as openai_clip
            tok = openai_clip.tokenize
        except Exception:
            tok = None
    if tok is None:
        print("No tokenizer available; skip CLIP Score.")
        return []

    # 按 prompt 分组，避免重复 encode_text
    prompt_to_paths: Dict[str, List[Path]] = {}
    for p in image_paths:
        prompt = parse_prompt_from_stem(p.stem)
        prompt_to_paths.setdefault(prompt, []).append(p)

    scores: List[float] = []

    with torch.inference_mode():
        for prompt, paths in tqdm(prompt_to_paths.items(),
                                  desc=f"CLIP Score (prompts={len(prompt_to_paths)})"):
            t = tok([prompt]).to(device)
            tfeat = model.encode_text(t)
            tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-12)

            # 这一个 prompt 下的一批图
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                ims = []
                for p in batch_paths:
                    try:
                        ims.append(preprocess(Image.open(p).convert("RGB")))
                    except Exception:
                        continue
                if not ims:
                    continue
                x = torch.stack(ims).to(device=device, dtype=torch.float32)
                if not hasattr(model, "encode_image"):
                    print("JIT model has no encode_image; skip CLIP.")
                    return []
                feat = model.encode_image(x)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
                sim = (100.0 * feat @ tfeat.T).squeeze(-1).detach().cpu().tolist()
                scores.extend([float(s) for s in sim])

    return scores


# ------------------ MS-SSIM ------------------

def calculate_ms_ssim_scores(folder: Path, max_pairs: int,
                             device: torch.device) -> List[float]:
    """
    Return a list of (1 - MS-SSIM) scores for random image pairs.
    """
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if len(paths) < 2:
        return []

    rng = np.random.default_rng(0)

    target_n = len(paths)
    if max_pairs > 0:
        target_n = max(2, int(np.sqrt(max_pairs * 2)) + 1)
        target_n = min(target_n, len(paths))
    if len(paths) > target_n:
        paths = list(rng.choice(paths, size=target_n, replace=False))

    pairs = [(paths[i], paths[j])
             for i in range(len(paths)) for j in range(i + 1, len(paths))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    tfm = T.Compose([T.ToTensor()])
    scores = []

    for p1, p2 in tqdm(pairs, desc=f"MS-SSIM pairs: {folder.name}"):
        try:
            im1 = Image.open(p1).convert("RGB")
            im2 = Image.open(p2).convert("RGB")
        except Exception:
            continue
        im2 = im2.resize(im1.size, Image.BICUBIC)
        x1 = tfm(im1).unsqueeze(0).to(device)
        x2 = tfm(im2).unsqueeze(0).to(device)

        with torch.inference_mode():
            if piq is not None:
                s = piq.multi_scale_ssim(
                    x1, x2, data_range=1.0, reduction="none"
                )
                score = float(s.item())
            elif ms_ssim_torch is not None:
                score = float(ms_ssim_torch(x1, x2, data_range=1.0).item())
            else:
                print("No MS-SSIM backend; skip.")
                return []
        scores.append(1.0 - score)

    return scores


# ------------------ BRISQUE ------------------

def calculate_brisque_scores(folder: Path, device: torch.device) -> List[float]:
    """
    Return a list of BRISQUE scores (one per image).
    """
    if piq is None:
        print("piq not installed; skip BRISQUE.")
        return []

    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not paths:
        return []

    tfm = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none').to(device)

    scores = []
    for p in tqdm(paths, desc=f"BRISQUE: {folder.name}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        x = tfm(img).unsqueeze(0).to(device)

        # 跳过零方差图像，避免 BRISQUE 内部 assert
        with torch.no_grad():
            var = x.view(x.size(0), -1).var(dim=1)
        if torch.isclose(var, torch.zeros_like(var)).all():
            print(f"[BRISQUE] skip zero-variance image: {p}")
            continue

        with torch.inference_mode():
            s = metric(x)
        scores.append(float(s.item()))

    return scores


# ------------------ evaluate one method ------------------

def evaluate_flat_method(
    method_dir: Path,
    method_name: str,
    device: torch.device,
    max_pairs: int,
    inception,
    real_features: Optional[torch.Tensor],
    real_dir: Optional[Path],
    fid_mode: str,
    match_real_count: bool,
    clip_model,
    clip_preproc,
    clip_batch_size: int,
) -> Dict[str, Any]:

    if not method_dir.exists() or not method_dir.is_dir():
        raise FileNotFoundError(f"Method dir not found: {method_dir}")

    img_paths = list_images(method_dir)
    if not img_paths:
        raise RuntimeError(f"No images in {method_dir}")

    print(f"\n=== Evaluating method: {method_name} ({len(img_paths)} images) ===")

    # -------- Vendi --------
    imgs = []
    for p in tqdm(img_paths, desc=f"Loading images for Vendi: {method_name}"):
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not imgs:
        raise RuntimeError(f"Could not load any images in {method_dir}")
    vendi_scores = compute_vendi_for_images(imgs, device=str(device))

    # -------- MS-SSIM (1 - MS-SSIM) --------
    ms_scores = calculate_ms_ssim_scores(method_dir, max_pairs=max_pairs, device=device)
    ms_mean, ms_ci_low, ms_ci_high = mean_ci(ms_scores) if ms_scores else (None, None, None)

    # -------- BRISQUE --------
    brisque_scores = calculate_brisque_scores(method_dir, device=device)
    br_mean, br_ci_low, br_ci_high = mean_ci(brisque_scores) if brisque_scores else (None, None, None)

    # -------- FID (piq) / clean-FID --------
    fid_value = None
    fid_clean = None

    fake_feats = torch.empty(0)
    if inception is not None:
        fake_feats = _extract_inception_features(
            img_paths, inception, device,
            num_workers=4, batch_size=32
        )

    if fid_mode in ("piq", "both") and inception is not None and \
       real_features is not None and real_features.numel() > 0 and \
       fake_feats.numel() > 0:
        real_feats_used = real_features
        if match_real_count and real_features.shape[0] >= fake_feats.shape[0]:
            rng = np.random.default_rng(abs(hash(method_name)) % (2**32))
            idx = rng.choice(real_features.shape[0], size=fake_feats.shape[0], replace=False)
            real_feats_used = real_features[idx]
        fid_value = compute_fid_from_features(real_feats_used, fake_feats)

    if fid_mode in ("clean", "both") and real_dir is not None:
        fid_clean = compute_fid_cleanfid(real_dir, method_dir, mode="clean")

    # -------- CLIP Score --------
    clip_mean = clip_ci_low = clip_ci_high = None
    if clip_model is not None and clip_preproc is not None:
        clip_scores = calculate_clip_scores_flat(
            img_paths, clip_model, clip_preproc, device, clip_batch_size
        )
        if clip_scores:
            clip_mean, clip_ci_low, clip_ci_high = mean_ci(clip_scores)

    result = {
        "method": method_name,
        "num_images": len(imgs),
        "vendi_pixel": vendi_scores.get("vendi_pixel"),
        "vendi_inception": vendi_scores.get("vendi_inception"),
        "one_minus_ms_ssim_mean": ms_mean,
        "one_minus_ms_ssim_ci_low": ms_ci_low,
        "one_minus_ms_ssim_ci_high": ms_ci_high,
        "brisque_mean": br_mean,
        "brisque_ci_low": br_ci_low,
        "brisque_ci_high": br_ci_high,
        "fid": fid_value,
        "fid_clean": fid_clean,
        "clip_mean": clip_mean,
        "clip_ci_low": clip_ci_low,
        "clip_ci_high": clip_ci_high,
    }

    print("Result:", result)
    return result


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Flat-folder eval: Vendi + MS-SSIM + BRISQUE + FID + CLIP.")
    ap.add_argument("--root", required=True,
                    help="Root dir containing method subfolders (cads, dpp, pg, standardFM, ...).")
    ap.add_argument("--methods", nargs="+",
                    default=["cads", "dpp", "pg", "standardFM"],
                    help="Method folder names under root.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-pairs", type=int, default=200,
                    help="Max random pairs for MS-SSIM.")

    # FID 相关
    ap.add_argument("--real-dir", type=str, default="",
                    help="Flat folder of real images for FID. If empty, FID is skipped.")
    ap.add_argument("--fid-mode", choices=["piq", "clean", "both", "none"], default="piq")
    ap.add_argument("--match-real-count", action="store_true",
                    help="Subsample real features to match #fake when computing FID(piq).")

    # CLIP 相关
    ap.add_argument("--clip-jit", type=str,
                    default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--clip-image-size", type=int, default=224)
    ap.add_argument("--clip-batch-size", type=int, default=64)

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root dir not found: {root}")

    # ---------- FID: prepare real features ----------
    real_dir = Path(args.real_dir) if args.real_dir else None
    inception = None
    real_features = None
    if args.fid_mode in ("piq", "both") and real_dir is not None:
        inception = _build_inception_extractor(device)
        if inception is not None:
            real_paths = list_images(real_dir)
            if real_paths:
                real_features = _extract_inception_features(
                    real_paths, inception, device,
                    num_workers=4, batch_size=32
                )
            else:
                print("No real images; FID(piq) will be skipped.")
                inception = None

    # ---------- CLIP: load JIT ----------
    clip_model = None
    clip_preproc = None
    jit_path = Path(args.clip_jit) if args.clip_jit else None
    if jit_path and jit_path.exists():
        try:
            clip_model, clip_preproc = load_clip_from_jit(
                jit_path, image_size=args.clip_image_size
            )
            clip_model = clip_model.to(device)
            print(f"Loaded CLIP JIT from {jit_path} (image_size={args.clip_image_size})")
        except Exception as e:
            print(f"Failed to load CLIP JIT: {e}")
    else:
        print(f"CLIP JIT not found at {jit_path}; CLIP Score will be skipped.")

    # ---------- run for each method ----------
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "flat_eval_with_fid_clip.csv"

    columns = [
        "method", "num_images",
        "vendi_pixel", "vendi_inception",
        "one_minus_ms_ssim_mean", "one_minus_ms_ssim_ci_low", "one_minus_ms_ssim_ci_high",
        "brisque_mean", "brisque_ci_low", "brisque_ci_high",
        "fid", "fid_clean",
        "clip_mean", "clip_ci_low", "clip_ci_high",
    ]

    all_results: List[Dict[str, Any]] = []

    for method in args.methods:
        m_dir = root / method
        if not m_dir.exists():
            print(f"[skip] method folder not found: {m_dir}")
            continue
        try:
            res = evaluate_flat_method(
                m_dir, method, device=device, max_pairs=args.max_pairs,
                inception=inception, real_features=real_features,
                real_dir=real_dir,
                fid_mode=args.fid_mode, match_real_count=args.match_real_count,
                clip_model=clip_model, clip_preproc=clip_preproc,
                clip_batch_size=args.clip_batch_size,
            )
            all_results.append(res)
        except Exception as e:
            print(f"[error] evaluating {method}: {e}")

    if not all_results:
        print("No methods evaluated; nothing to write.")
        return

    print(f"\nWriting CSV -> {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    print("Done.")


if __name__ == "__main__":
    main()
