# We'll create a new evaluator script that:
# - Accepts multiple methods (e.g., "our,pg,cads,dpp") and an outputs root
# - Discovers per-method, per-step seed folders like outputs/{method}_NFE/imgs/seed{seed}_steps{steps}/
# - Computes metrics for each seed folder (reusing your metric functions)
# - Aggregates across seeds for the same (method, steps) to mean/std
# - Writes a CSV to outputs/results/aggregate_metrics.csv
# - Optionally also writes per-seed JSONs like your original (kept behavior)
#
# I’ll copy your metric implementations and adapt the CLI & grouping/aggregation logic.

from pathlib import Path
import os, json, csv, argparse, re, sys, math, time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from skimage.metrics import structural_similarity as ssim
import cv2

# Vendi score
try:
    from vendi_score import image_utils, vendi
except Exception:
    image_utils = None
    vendi = None


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

def _nan_to_none(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj

# ---------------- Vendi ----------------
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
    # If vendi is unavailable, fallback to simple entropy-like proxy
    if vendi is None:
        # cosine sim matrix entropy proxy
        S = X @ X.T
        S = (S - S.min()) / (S.max() - S.min() + 1e-8)
        # entropy of average row
        p = (S.mean(axis=1) + 1e-8)
        p = p / p.sum()
        return float(-(p * np.log(p + 1e-8)).sum())
    return float(vendi.score_dual(X, normalize=False))

def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, Optional[float]]:
    pix_vs = None
    if image_utils is not None:
        try:
            pix_vs = float(image_utils.pixel_vendi_score(imgs))
        except Exception:
            pix_vs = None
    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device)) if image_utils is not None else _embedding_vendi_score_fallback(imgs, device=device)
    except Exception as e:
        emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# ---------------- Inception features ----------------
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

# ---------------- KID ----------------
def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** 3

def compute_kid_from_features(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    subset_size: int = 1000,
    n_subsets: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[float, float]]:
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
        xs = X[ridx]
        ys = Y[fidx]

        k_xx = _poly_kernel(xs, xs)
        k_yy = _poly_kernel(ys, ys)
        k_xy = _poly_kernel(xs, ys)

        sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (s * (s - 1))
        sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (s * (s - 1))
        sum_xy = k_xy.mean()

        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        stats.append(float(mmd2.item()))

    stats = np.array(stats, dtype=np.float64)
    kid_mean = float(stats.mean())
    kid_std  = float(stats.std(ddof=1)) if stats.size > 1 else 0.0
    return kid_mean, kid_std

# ---------------- CLIP score ----------------
def load_clip_from_jit(jit_path: Path):
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    return model

def calculate_clip_score(
    image_folder: Path,
    model: Any,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
    tokenizer=None,
    is_openclip: bool = False,
    clip_image_size: int = 224,
) -> Optional[float]:
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

            x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
            if x.shape[-2:] != (clip_image_size, clip_image_size):
                x = F.interpolate(x, size=(clip_image_size, clip_image_size), mode="bilinear", align_corners=False, antialias=True)
            x = (x - mean) / std

            image_features = model.encode_image(x)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
            sim = (100.0 * (image_features @ text_features.T)).squeeze(-1).detach().cpu().tolist()
            all_scores.extend(sim)

    return float(np.mean(all_scores)) if all_scores else None

# ---------------- MS-SSIM ----------------
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

# ---------------- BRISQUE ----------------
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

# ---------------- Evaluation core ----------------
def eval_seed_folder(gen_dir: Path, device: torch.device, inception, real_features, args) -> Dict[str, Optional[float]]:
    paths = list_images(gen_dir)
    if not paths:
        return {}

    imgs = load_pils(paths)
    if not imgs:
        return {}

    vendi_scores = compute_vendi_for_images(imgs, device=str(device))

    # FID/KID
    fid_value = None
    kid_mean, kid_std = None, None
    if inception is not None and real_features is not None and real_features.numel() > 0:
        fake_features = _extract_inception_features(
            paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size
        )
        fid_value = compute_fid_from_features(real_features, fake_features)
        if args.kid_subsets > 0 and fake_features is not None and fake_features.numel() > 0:
            rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
            kid_ret = compute_kid_from_features(real_features, fake_features, subset_size=args.kid_subset_size, n_subsets=args.kid_subsets, rng=rng)
            if kid_ret is not None:
                kid_mean, kid_std = kid_ret

    # CLIP
    clip_score = None
    if args.clip_model is not None:
        clip_score = calculate_clip_score(gen_dir, args.clip_model, None, device, args.batch_size, clip_image_size=args.clip_image_size)

    # Diversity & quality
    msssim_div = calculate_ms_ssim_diversity(gen_dir, max_pairs=args.max_pairs)
    brisque_quality = calculate_brisque_quality(gen_dir, device=device)

    return {
        "num_images": len(imgs),
        "vendi_pixel": vendi_scores.get("vendi_pixel"),
        "vendi_inception": vendi_scores.get("vendi_inception"),
        "fid": fid_value,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "clip_score": clip_score,
        "one_minus_ms_ssim": msssim_div,
        "brisque": brisque_quality,
    }

def aggregate_mean_std(per_seed_metrics: List[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    # compute mean/std for each numeric metric across seeds
    keys = set().union(*[m.keys() for m in per_seed_metrics]) if per_seed_metrics else set()
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if m.get(k) is not None and isinstance(m.get(k), (int, float))]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0, "count": len(vals)}
        else:
            agg[k] = {"mean": None, "std": None, "count": 0}
    return agg

def parse_seed_steps(folder_name: str) -> Optional[Tuple[int, int]]:
    # expect names like seed{seed}_steps{steps}
    m = re.match(r"seed(\\d+)_steps(\\d+)", folder_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def discover_seed_step_folders(method_root_imgs: Path) -> Dict[int, List[Path]]:
    # returns mapping: steps -> list of seed folder Paths
    mapping: Dict[int, List[Path]] = {}
    for sub in sorted(method_root_imgs.iterdir() if method_root_imgs.exists() else []):
        if not sub.is_dir():
            continue
        parsed = parse_seed_steps(sub.name)
        if not parsed:
            continue
        seed, steps = parsed
        mapping.setdefault(steps, []).append(sub)
    return mapping

def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics across seeds per method & steps.")
    ap.add_argument("--methods", type=str, required=True, help="Comma-separated method names, e.g. 'our,pg,cads,dpp'")
    ap.add_argument("--outputs-root", type=str, default="outputs", help="Root outputs folder that contains {method}_NFE/")
    ap.add_argument("--real", type=str, required=True, help="Reference real image folder for FID/KID")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--clip-image-size", type=int, default=224)
    ap.add_argument("--max-pairs", type=int, default=100)
    ap.add_argument("--kid-subset-size", type=int, default=64)
    ap.add_argument("--kid-subsets", type=int, default=20)
    ap.add_argument("--save-per-seed-json", action="store_true", help="Also save per-seed JSON under each method's eval/")
    args = ap.parse_args(args=None)

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    outputs_root = Path(args.outputs_root)
    results_dir = outputs_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.real)

    # Build shared models
    inception = _build_inception_extractor(device)
    real_features = None
    if inception is not None:
        real_paths = list_images(real_dir)
        if real_paths:
            real_features = _extract_inception_features(real_paths, inception, device, num_workers=args.num_workers, batch_size=args.batch_size)
        else:
            print("No real images found; FID/KID disabled.")
            inception = None

    # CLIP JIT
    args.clip_model = None
    jit_path = Path(args.clip_jit) if args.clip_jit else None
    if jit_path and jit_path.exists():
        try:
            args.clip_model = load_clip_from_jit(jit_path).to(device)
            print(f"Loaded CLIP JIT from {jit_path}")
        except Exception as e:
            print(f"Failed to load CLIP JIT: {e}")

    # Iterate methods
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []  # for CSV aggregation

    for method in methods:
        method_root = outputs_root / f"{method}_NFE"
        imgs_root = method_root / "imgs"
        eval_root = method_root / "eval"
        imgs_root.mkdir(parents=True, exist_ok=True)
        eval_root.mkdir(parents=True, exist_ok=True)

        step_to_seeds = discover_seed_step_folders(imgs_root)
        if not step_to_seeds:
            print(f"[WARN] No seed-step folders found for method={method} under {imgs_root}")
            continue

        for steps, seed_dirs in sorted(step_to_seeds.items()):
            per_seed_metrics = []
            for seed_dir in seed_dirs:
                m = eval_seed_folder(seed_dir, device, inception, real_features, args)
                if not m:
                    continue
                per_seed_metrics.append(m)

                if args.save_per_seed_json:
                    # Save per-seed JSON under method eval/
                    payload = {
                        "real_root": str(real_dir),
                        "device": str(device),
                        "results": {seed_dir.name: _nan_to_none(m)},
                        "method": method,
                        "steps": int(steps),
                    }
                    out_file = eval_root / f"{seed_dir.name}.json"
                    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            if not per_seed_metrics:
                continue

            agg = aggregate_mean_std(per_seed_metrics)

            # Build a flat CSV row
            flat = {
                "method": method,
                "steps": int(steps),
                "seeds": len(seed_dirs),
                "num_images_mean": agg.get("num_images", {}).get("mean"),
                "num_images_std": agg.get("num_images", {}).get("std"),
                "vendi_pixel_mean": agg.get("vendi_pixel", {}).get("mean"),
                "vendi_pixel_std": agg.get("vendi_pixel", {}).get("std"),
                "vendi_inception_mean": agg.get("vendi_inception", {}).get("mean"),
                "vendi_inception_std": agg.get("vendi_inception", {}).get("std"),
                "fid_mean": agg.get("fid", {}).get("mean"),
                "fid_std": agg.get("fid", {}).get("std"),
                "kid_mean_mean": agg.get("kid_mean", {}).get("mean"),
                "kid_mean_std": agg.get("kid_mean", {}).get("std"),
                "kid_std_mean": agg.get("kid_std", {}).get("mean"),
                "kid_std_std": agg.get("kid_std", {}).get("std"),
                "clip_score_mean": agg.get("clip_score", {}).get("mean"),
                "clip_score_std": agg.get("clip_score", {}).get("std"),
                "one_minus_ms_ssim_mean": agg.get("one_minus_ms_ssim", {}).get("mean"),
                "one_minus_ms_ssim_std": agg.get("one_minus_ms_ssim", {}).get("std"),
                "brisque_mean": agg.get("brisque", {}).get("mean"),
                "brisque_std": agg.get("brisque", {}).get("std"),
            }
            rows.append(flat)

    # Write CSV
    csv_path = results_dir / "aggregate_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["method","steps","seeds"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v)))) else v) for k,v in r.items()})
    print(f"[OK] Saved aggregate CSV: {csv_path}")

if __name__ == "__main__":
    main()