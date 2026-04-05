import argparse, os, re, json, csv, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# 尝试导入评估库
try:
    import piq
except ImportError:
    piq = None

from vendi_score import vendi

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# 适配 T2I-CompBench 的 slug 命名或带参数命名
META_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")

# ─────────────── KID helpers ───────────────────────────────────────────────
_INC_MODEL_CACHE: Dict[str, nn.Module] = {}

def _get_inception(device: str) -> Tuple[nn.Module, Any]:
    from torchvision.models import inception_v3, Inception_V3_Weights
    if device not in _INC_MODEL_CACHE:
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = nn.Identity()
        model.eval().to(device)
        _INC_MODEL_CACHE[device] = (model, weights.transforms())
    return _INC_MODEL_CACHE[device]

def extract_inception_features(img_paths: List[Path], device: str,
                               batch_size: int = 32) -> torch.Tensor:
    """Extract InceptionV3 pool features for a list of image paths."""
    model, tfm = _get_inception(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch = []
            for p in img_paths[i : i + batch_size]:
                try:
                    batch.append(tfm(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not batch:
                continue
            x = torch.stack(batch).to(device)
            f = model(x)
            if isinstance(f, (list, tuple)):
                f = f[0]
            feats.append(f.cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0, 2048)

def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Polynomial kernel k(x,y) = (x^T y / d + 1)^3."""
    d = x.shape[1]
    return (x.to(torch.float64) @ y.to(torch.float64).T / float(d) + 1.0) ** 3

def compute_kid(ref_feats: torch.Tensor, gen_feats: torch.Tensor,
                n_subsets: int = 100, subset_size: int = 1000,
                rng: Optional[np.random.Generator] = None) -> Optional[Tuple[float, float]]:
    """
    Global KID: ref_feats = COCO val features (M~5000),
                gen_feats = all generated images of one method+concept (N = #prompts × G).

    Uses subset sampling (standard practice for large N/M) to get a stable estimate.
    Returns (kid_mean, kid_std) ×1e3, or None.
    """
    if ref_feats.numel() == 0 or gen_feats.numel() == 0:
        return None
    M, N = ref_feats.shape[0], gen_feats.shape[0]
    s = min(subset_size, M, N)
    if s < 2:
        return None
    if rng is None:
        rng = np.random.default_rng(42)
    stats = []
    for _ in range(n_subsets):
        xs = ref_feats[rng.choice(M, s, replace=False)]
        ys = gen_feats[rng.choice(N, s, replace=False)]
        k_xx = _poly_kernel(xs, xs)
        k_yy = _poly_kernel(ys, ys)
        k_xy = _poly_kernel(xs, ys)
        m_, n_ = k_xx.shape[0], k_yy.shape[0]
        mmd2 = (
            (k_xx.sum() - k_xx.diag().sum()) / (m_ * (m_ - 1))
            + (k_yy.sum() - k_yy.diag().sum()) / (n_ * (n_ - 1))
            - 2.0 * k_xy.mean()
        )
        stats.append(float(mmd2.item()))
    a = np.array(stats)
    return float(a.mean() * 1e3), float(a.std(ddof=1) * 1e3)


def build_coco_features(coco_dir: str, device: str,
                        max_images: int = 5000) -> torch.Tensor:
    """
    Extract InceptionV3 features for COCO val images (used as KID reference).
    Cached to a .pt file next to the COCO dir for speed.
    """
    coco_path = Path(coco_dir)
    cache_file = coco_path.parent / f"coco_inception_feats_{max_images}.pt"

    if cache_file.exists():
        print(f"[KID] Loading cached COCO features from {cache_file}")
        return torch.load(cache_file, map_location="cpu")

    print(f"[KID] Extracting COCO inception features (max {max_images} images)…")
    img_paths = sorted(
        p for p in coco_path.iterdir()
        if p.suffix.lower() in IMG_EXTS
    )[:max_images]
    feats = extract_inception_features(img_paths, device, batch_size=64)
    torch.save(feats, cache_file)
    print(f"[KID] Saved {feats.shape[0]} COCO features to {cache_file}")
    return feats
# ──────────────────────────────────────────────────────────────────────────

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_meta_from_name(name: str) -> Dict[str, Any]:
    m = META_RE.match(name)
    if not m:
        return {"prompt": name.replace("_", " "), "seed": None, "guidance": None, "steps": None}
    d = m.groupdict()
    d["prompt"]   = d["prompt"].replace("_", " ")
    return d

# --------- 彻底修复 Vendi Score 报错 ----------
def get_vendi_features(imgs: List[Image.Image], device: str) -> np.ndarray:
    """手动提取 InceptionV3 特征，避开 vendi_score 库内部的 torchvision 兼容性 Bug"""
    model, preprocess = _get_inception(device)
    feats = []
    with torch.no_grad():
        for img in imgs:
            x = preprocess(img).unsqueeze(0).to(device)
            f = model(x)
            if isinstance(f, (list, tuple)): f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_vendi_scores_safe(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    """计算 Vendi Score (无参考)"""
    try:
        # Pixel Vendi 比较稳，可以直接用
        from vendi_score import image_utils
        pix_vs = float(image_utils.pixel_vendi_score(imgs))
        
        # 提取特征并计算 Inception Vendi
        X = get_vendi_features(imgs, device)
        emb_vs = float(vendi.score_dual(X)) # 核心矩阵算法
        
        return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}
    except Exception as e:
        print(f"  [Vendi Error] {e}")
        return {"vendi_pixel": 1.0, "vendi_inception": 1.0}

def compute_msssim_diversity(paths: List[Path], device: torch.device) -> float:
    if len(paths) < 2 or piq is None: return 0.0
    rng = np.random.default_rng(42)
    max_pairs = 30 # T2I任务对子不用太多
    indices = rng.choice(len(paths), size=len(paths), replace=False)
    pairs = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            pairs.append((paths[indices[i]], paths[indices[j]]))
            if len(pairs) >= max_pairs: break
        if len(pairs) >= max_pairs: break

    tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])
    scores = []
    for p1, p2 in pairs:
        try:
            x1 = tfm(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
            x2 = tfm(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
            # 使用 piq 的 ms_ssim
            s = piq.multi_scale_ssim(x1, x2, data_range=1.0)
            scores.append(s.item())
        except: continue
    return float(1.0 - np.mean(scores)) if scores else 0.0

# --------- CLIP 对齐度 ----------
def compute_clip_alignment(folder: Path, prompt: str, model, preprocess, device):
    image_paths = list_images(folder)
    if not image_paths or model is None: return 0.0
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_input = tokenizer([prompt]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = []
            for p in image_paths:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                img_feat = model.encode_image(img)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims.append((img_feat @ text_features.T).item())
        return float(np.mean(sims)) * 100.0
    except: return 0.0

# ─────────────── path helper ──────────────────────────────────────────────
def _find_method_dir(outputs_root: Path, method: str, concept: str) -> Optional[Path]:
    for candidate in [
        outputs_root / f"{method}_{concept}",
        outputs_root / f"{method}_t2i_{concept}",
        outputs_root / f"baseline_{method}_{concept}",
    ]:
        if (candidate / "imgs").exists():
            return candidate
    return None


# ─────────────── per-prompt metrics (vendi / clip / ms-ssim) ──────────────
def evaluate_method_concept(outputs_root: Path, method: str, concept: str, args):
    target_dir = _find_method_dir(outputs_root, method, concept)
    if target_dir is None:
        print(f"[Skip] Could not find folder for method={method}, concept={concept}")
        return

    gen_root = target_dir / "imgs"
    eval_dir = target_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    csv_path = eval_dir / "metrics_per_prompt.csv"

    # CLIP model (loaded once per call)
    clip_model, clip_preprocess = None, None
    try:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(args.device).eval()
    except Exception:
        pass

    fields = ["prompt_folder", "prompt_text",
              "vendi_inception", "vendi_pixel",
              "clip_score", "one_minus_ms_ssim"]

    prompt_dirs = sorted([d for d in gen_root.iterdir() if d.is_dir()])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p_dir in tqdm(prompt_dirs, desc=f"Eval {target_dir.name}"):
            img_paths = list_images(p_dir)
            if not img_paths:
                continue
            meta = parse_meta_from_name(p_dir.name)
            imgs_pil = [Image.open(p).convert("RGB") for p in img_paths]

            v_scores  = compute_vendi_scores_safe(imgs_pil, device=args.device)
            clip_align = compute_clip_alignment(p_dir, meta["prompt"],
                                                clip_model, clip_preprocess, args.device)
            msssim_div = compute_msssim_diversity(img_paths, torch.device(args.device))

            writer.writerow({
                "prompt_folder":     p_dir.name,
                "prompt_text":       meta["prompt"],
                "vendi_inception":   round(v_scores["vendi_inception"], 4),
                "vendi_pixel":       round(v_scores["vendi_pixel"], 4),
                "clip_score":        round(clip_align, 4),
                "one_minus_ms_ssim": round(msssim_div, 4),
            })
            del imgs_pil
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[per-prompt] Saved → {csv_path}")


# ─────────────── global KID (all images of method+concept vs COCO) ─────────
def compute_global_kid_for_method(outputs_root: Path, method: str, concept: str,
                                  coco_feats: torch.Tensor, device: str,
                                  n_subsets: int = 100, subset_size: int = 1000
                                  ) -> Optional[Tuple[float, float]]:
    """
    Collect ALL generated images of (method, concept) across every prompt,
    extract InceptionV3 features, then compute KID vs COCO val.

    Returns (kid_mean×1e3, kid_std×1e3) or None.
    """
    target_dir = _find_method_dir(outputs_root, method, concept)
    if target_dir is None:
        print(f"[KID] method={method} concept={concept} not found, skip.")
        return None

    gen_root = target_dir / "imgs"
    all_paths: List[Path] = []
    for p_dir in sorted(gen_root.iterdir()):
        if p_dir.is_dir():
            all_paths.extend(list_images(p_dir))

    if not all_paths:
        print(f"[KID] No images found under {gen_root}")
        return None

    print(f"[KID] {method}/{concept}: {len(all_paths)} images total, extracting features…")
    gen_feats = extract_inception_features(all_paths, device, batch_size=64)

    result = compute_kid(coco_feats, gen_feats, n_subsets=n_subsets,
                         subset_size=subset_size)
    if result:
        print(f"[KID] {method}/{concept}: KID = {result[0]:.4f} ± {result[1]:.4f} (×1e3)")
    return result


# ─────────────── main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=str, default="./outputs")
    parser.add_argument("--methods",  nargs="+", required=True)
    parser.add_argument("--concepts", nargs="+", required=True)
    parser.add_argument("--device",   type=str,  default="cuda:0")
    # KID
    parser.add_argument("--coco-dir", type=str,
                        default="/data2/toby/OSCAR/datasets/coco2017/val2017",
                        help="Path to COCO val2017 directory. Set to '' to skip KID.")
    parser.add_argument("--kid-subsets",     type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.outputs_root)

    # ── Step 1: per-prompt metrics ─────────────────────────────────────────
    for method in args.methods:
        for concept in args.concepts:
            evaluate_method_concept(root, method, concept, args)

    # ── Step 2: global KID vs COCO ─────────────────────────────────────────
    if not args.coco_dir:
        print("[KID] --coco-dir not set, skipping KID.")
        return

    coco_feats = build_coco_features(args.coco_dir, args.device)

    kid_rows = []
    for method in args.methods:
        for concept in args.concepts:
            result = compute_global_kid_for_method(
                root, method, concept, coco_feats, args.device,
                n_subsets=args.kid_subsets,
                subset_size=args.kid_subset_size,
            )
            kid_rows.append({
                "method":    method,
                "concept":   concept,
                "kid_mean":  round(result[0], 4) if result else None,
                "kid_std":   round(result[1], 4) if result else None,
            })

    # Save global KID summary
    kid_csv = root / "global_kid.csv"
    with open(kid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "concept", "kid_mean", "kid_std"])
        writer.writeheader()
        writer.writerows(kid_rows)
    print(f"\n[KID] Global KID summary → {kid_csv}")

    # Pretty-print table
    print(f"\n{'Method':<12} {'Concept':<12} {'KID (×1e-3)':<14} {'±std'}")
    print("-" * 50)
    for r in kid_rows:
        mean_s = f"{r['kid_mean']:.4f}" if r["kid_mean"] is not None else "N/A"
        std_s  = f"{r['kid_std']:.4f}"  if r["kid_std"]  is not None else "N/A"
        print(f"{r['method']:<12} {r['concept']:<12} {mean_s:<14} {std_s}")


if __name__ == "__main__":
    main()