# -*- coding: utf-8 -*-
"""
Batch Aggregated Evaluator across Concepts and Methods.
Calculates Mean and 95% Confidence Intervals per Guidance Level.

Usage example:
  python eval_aggregated.py \
    --outputs-root /data2/toby/OSCAR/outputs \
    --concepts bowl truck dog \
    --methods cads dpp pg \
    --real-root /data2/toby/OSCAR/real_cls_crops \
    --save-dir ./results_aggregated \
    --fid-mode both --match-real-count \
    --clip-jit ~/.cache/clip/ViT-B-32.pt
"""

import argparse, os, re, sys, csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd  # Added pandas for easy aggregation
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# --- Imports for Metrics (Same as before) ---
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    import cleanfid
except ImportError:
    cleanfid = None

try:
    from pytorch_msssim import ms_ssim as ms_ssim_torch
except ImportError:
    ms_ssim_torch = None

from vendi_score import image_utils, vendi

# ========================== Utils & Metrics Logic ==========================
# (Most helper functions are kept identical to your script to ensure consistency)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# Adjusted Regex to be robust
META_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def list_child_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def parse_meta_from_name(name: str) -> Dict[str, Any]:
    m = META_RE.match(name)
    if not m:
        # Fallback parsing if regex fails
        return {"prompt": name, "seed": -1, "guidance": -1.0, "steps": -1}
    d = m.groupdict()
    d["prompt"]   = d["prompt"].replace("_", " ")
    d["seed"]     = int(d["seed"])
    d["guidance"] = float(d["guidance"])
    d["steps"]    = int(d["steps"])
    return d

# --------- Vendi ----------
def _embedding_vendi_score_fallback(imgs: List[Image.Image], device: str = "cuda") -> float:
    import torchvision
    from torchvision.models import Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    tfm = weights.transforms()
    model = torchvision.models.inception_v3(weights=weights, aux_logits=True, transform_input=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    feats = []
    bs = 32
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            x = torch.stack([tfm(img) for img in imgs[i:i+bs]], dim=0).to(device)
            f = model(x)
            if isinstance(f, (tuple, list)): f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu())
    X = torch.cat(feats, dim=0).numpy()
    return float(vendi.score_dual(X, normalize=False))

def compute_vendi_for_images(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    if not imgs: return {}
    pix_vs = float(image_utils.pixel_vendi_score(imgs))
    try:
        emb_vs = float(image_utils.embedding_vendi_score(imgs, device=device))
    except Exception as e:
        emb_vs = _embedding_vendi_score_fallback(imgs, device=device)
    return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}

# --------- Inception features ----------
def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        return None
    return InceptionV3().to(device)

def _make_fid_transform():
    return T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(299),
        T.ToTensor(),
    ])

def _extract_inception_features(paths: List[Path], extractor: nn.Module, device: torch.device,
                                num_workers=4, batch_size=32) -> torch.Tensor:
    transform = _make_fid_transform()
    class D(torch.utils.data.Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            img = Image.open(self.files[i]).convert("RGB")
            return transform(img)
    if not paths: return torch.empty(0, 2048)
    loader = torch.utils.data.DataLoader(D(paths), batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=("cuda" in device.type))
    feats_all = []
    with torch.inference_mode():
        # Reduced logging
        for imgs in loader:
            feats = extractor(imgs.to(device))
            if isinstance(feats, list): feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).squeeze(-1).squeeze(-1)
            feats_all.append(feats.cpu())
    return torch.cat(feats_all, dim=0) if feats_all else torch.empty(0, 2048)

# --------- FID / KID / CLEAN-FID ----------
def compute_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> Optional[float]:
    if piq is None or real_features.numel()==0 or fake_features.numel()==0: return None
    return float(piq.FID()(real_features, fake_features))

def compute_fid_cleanfid(real_dir: Path, gen_dir: Path, mode="clean") -> Optional[float]:
    if cleanfid is None: return None
    try:
        return float(cleanfid.compute_fid(str(real_dir), str(gen_dir), mode=mode, model_name="inception_v3", verbose=False))
    except Exception:
        return None

def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** 3

def compute_kid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor,
                              subset_size: int = 1000, n_subsets: int = 10,
                              rng: Optional[np.random.Generator] = None) -> Optional[Tuple[float, float]]:
    if real_features is None or fake_features is None: return None
    if real_features.numel()==0 or fake_features.numel()==0: return None
    if real_features.shape[1] != fake_features.shape[1]: return None

    m_full, n_full = real_features.shape[0], fake_features.shape[0]
    if m_full < 2 or n_full < 2: return None
    subset = min(subset_size, m_full, n_full)
    if subset < 2: return None
    if rng is None: rng = np.random.default_rng(0)

    X = real_features.to(torch.float64)
    Y = fake_features.to(torch.float64)
    stats = []
    for _ in range(max(1, n_subsets)):
        ridx = rng.choice(m_full, size=subset, replace=False)
        fidx = rng.choice(n_full, size=subset, replace=False)
        xs, ys = X[ridx], Y[fidx]
        k_xx = _poly_kernel(xs, xs)
        k_yy = _poly_kernel(ys, ys)
        k_xy = _poly_kernel(xs, ys)
        m, n = k_xx.shape[0], k_yy.shape[0]
        sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
        sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
        sum_xy = k_xy.mean()
        mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
        stats.append(float(mmd2.item()))
    stats = np.array(stats, dtype=np.float64)
    return float(stats.mean()), float(stats.std(ddof=1)) if stats.size > 1 else 0.0

# --------- CLIP Cosine ----------
def load_clip_from_jit(jit_path: Path, image_size: int = 224):
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    preprocess = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess

def calculate_clip_cosine(folder: Path, model, preprocess, device, batch_size,
                          text: Optional[str] = None, tokenizer=None):
    image_paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not image_paths: return None

    if text is None:
        meta = parse_meta_from_name(folder.name)
        text = meta.get("prompt") or folder.name.replace("_"," ")

    if tokenizer is None:
        try:
            import open_clip as _oc; tokenizer = _oc.tokenize
        except:
            try:
                import clip as openai_clip; tokenizer = openai_clip.tokenize
            except:
                return None

    with torch.inference_mode():
        t = tokenizer([text]).to(device)
        tfeat = model.encode_text(t)
        tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-12)
        sims = []
        for i in range(0, len(image_paths), batch_size):
            ims = []
            for p in image_paths[i:i+batch_size]:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                except: continue
            if not ims: continue
            x = torch.stack(ims).to(device=device, dtype=torch.float32)
            feat = model.encode_image(x)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
            sim = (100.0 * feat @ tfeat.T).squeeze(-1).detach().cpu().numpy()
            sims.extend(sim.tolist())
    return float(np.mean(sims)) if sims else None

# --------- MS-SSIM & BRISQUE ----------
def calculate_ms_ssim_diversity(folder: Path, max_pairs: int, device: torch.device) -> Optional[float]:
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if len(paths) < 2: return None
    rng = np.random.default_rng(0)
    target_n = max(2, int(np.sqrt(max_pairs * 2)) + 1) if max_pairs > 0 else len(paths)
    if len(paths) > target_n:
        paths = rng.choice(paths, size=target_n, replace=False)
    pairs = [(paths[i], paths[j]) for i in range(len(paths)) for j in range(i+1, len(paths))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    tfm = T.Compose([T.ToTensor()])
    scores = []
    with torch.inference_mode():
        for p1, p2 in pairs:
            try:
                im1 = Image.open(p1).convert("RGB")
                im2 = Image.open(p2).convert("RGB")
            except: continue
            im2 = im2.resize(im1.size, Image.BICUBIC)
            x1 = tfm(im1).unsqueeze(0).to(device)
            x2 = tfm(im2).unsqueeze(0).to(device)
            if piq: s = piq.multi_scale_ssim(x1, x2, data_range=1.0, reduction="none"); score = float(s.item())
            elif ms_ssim_torch: score = float(ms_ssim_torch(x1, x2, data_range=1.0).item())
            else: return None
            scores.append(score)
    return float(1.0 - np.mean(scores)) if scores else None

def calculate_brisque_quality(folder: Path, device: torch.device) -> Optional[float]:
    if piq is None: return None
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not paths: return None
    tfm = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none').to(device)
    scores = []
    for p in paths:
        try: img = Image.open(p).convert("RGB")
        except: continue
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            var = x.view(x.size(0), -1).var(dim=1)
        if torch.isclose(var, torch.zeros_like(var)).all(): continue
        with torch.inference_mode():
            scores.append(float(metric(x).item()))
    return float(np.mean(scores)) if scores else None

# ========================== Aggregation Main Logic ==========================

def evaluate_single_folder(gen_dir: Path, meta: Dict, 
                           device, fid_mode, match_real_count, batch_size, num_workers, max_pairs,
                           clip_model, clip_preproc, inception, real_features, real_dir,
                           kid_subset_size, kid_subsets):
    """Computes all metrics for ONE folder (one seed/guidance/step set)."""
    
    result = {}
    paths = list_images(gen_dir)
    if not paths: return None

    # Load images
    try: imgs = [Image.open(p).convert("RGB") for p in paths]
    except: return None

    # 1. Vendi
    v_scores = compute_vendi_for_images(imgs, device=str(device))
    result.update(v_scores)

    # 2. Feature Extraction (Fake)
    fake_feats = torch.empty(0)
    if inception is not None:
        fake_feats = _extract_inception_features(paths, inception, device, num_workers=num_workers, batch_size=batch_size)

    # 3. FID (piq)
    if fid_mode in ("piq","both") and (inception is not None) and (real_features is not None) and (real_features.numel()>0):
        real_feats_used = real_features
        # Match count logic
        if match_real_count and fake_feats.numel() > 0 and real_features.shape[0] >= fake_feats.shape[0]:
            rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
            idx = rng.choice(real_features.shape[0], size=fake_feats.shape[0], replace=False)
            real_feats_used = real_features[idx]
        
        if fake_feats.numel() > 0:
            result["fid"] = compute_fid_from_features(real_feats_used, fake_feats)
        else:
            result["fid"] = None
    else:
        result["fid"] = None

    # 4. FID (clean-fid)
    if fid_mode in ("clean","both"):
        result["fid_clean"] = compute_fid_cleanfid(real_dir, gen_dir, mode="clean")
    else:
        result["fid_clean"] = None

    # 5. KID
    if (kid_subsets > 0) and (inception is not None) and (real_features is not None) and (fake_feats.numel() > 0):
        rng_kid = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
        kid_ret = compute_kid_from_features(real_features, fake_feats, subset_size=kid_subset_size, n_subsets=kid_subsets, rng=rng_kid)
        if kid_ret:
            result["kid_mean"], result["kid_std"] = kid_ret
        else:
            result["kid_mean"], result["kid_std"] = None, None
    else:
        result["kid_mean"], result["kid_std"] = None, None

    # 6. CLIP
    if (clip_model is not None) and (clip_preproc is not None):
        result["clip_score"] = calculate_clip_cosine(gen_dir, clip_model, clip_preproc, device, batch_size, text=meta.get("prompt"))
    else:
        result["clip_score"] = None

    # 7. MS-SSIM
    result["one_minus_ms_ssim"] = calculate_ms_ssim_diversity(gen_dir, max_pairs=max_pairs, device=device)

    # 8. BRISQUE
    result["brisque"] = calculate_brisque_quality(gen_dir, device=device)

    return result

def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics across multiple concepts and methods.")
    ap.add_argument("--outputs-root", required=True, help="Root dir containing {method}_{concept}/imgs")
    ap.add_argument("--real-root", required=True, help="Root containing real images (<real-root>/<concept>)")
    
    # Changed: Inputs are lists now
    ap.add_argument("--concepts", nargs="+", required=True, help="List of concepts, e.g. bowl truck dog")
    ap.add_argument("--methods", nargs="+", required=True, help="List of methods, e.g. cads dpp pg")
    
    ap.add_argument("--save-dir", required=True, help="Where to save the raw and aggregated CSVs")
    
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=100)
    ap.add_argument("--fid-mode", choices=["piq","clean","both","none"], default="both")
    ap.add_argument("--match-real-count", action="store_true")
    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--clip-image-size", type=int, default=224)
    ap.add_argument("--kid-subset-size", type=int, default=1000)
    ap.add_argument("--kid-subsets", type=int, default=10)

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    outputs_root = Path(args.outputs_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup shared Models (Inception, CLIP)
    need_inception = (args.fid_mode in ("piq","both")) or (args.kid_subsets > 0)
    inception = _build_inception_extractor(device) if need_inception else None

    clip_model, clip_preproc = None, None
    if args.clip_jit and Path(args.clip_jit).exists():
        try:
            clip_model, clip_preproc = load_clip_from_jit(Path(args.clip_jit), args.clip_image_size)
            clip_model = clip_model.to(device)
        except Exception as e:
            print(f"Error loading CLIP: {e}")

    # Data collection list
    all_results = []

    # 2. Iterate Logic: Concept -> Load Real Features -> Iterate Methods
    # This order optimizes speed by loading Real features once per concept.
    
    for concept in args.concepts:
        print(f"\n>>> Processing Concept: {concept} <<<")
        
        real_dir = Path(args.real_root) / concept
        if not real_dir.exists():
            # fallback
            real_dir = Path(args.real_root) / concept.replace(" ", "_")
        
        if not real_dir.exists():
            print(f"  [Warning] Real directory for {concept} not found at {real_dir}. FID/KID will fail.")
            real_features = None
        else:
            # Extract Real Features Once for this concept
            real_paths = list_images(real_dir)
            if inception and real_paths:
                print(f"  Extracting real features for {concept} ({len(real_paths)} imgs)...")
                real_features = _extract_inception_features(real_paths, inception, device, 
                                                            num_workers=args.num_workers, batch_size=args.batch_size)
            else:
                real_features = None

        # Now iterate methods for this concept
        for method in args.methods:
            method_concept_dir = outputs_root / f"{method}_{concept}" / "imgs"
            if not method_concept_dir.exists():
                print(f"  [Skip] Method dir not found: {method_concept_dir}")
                continue
            
            # Find all K-set folders (seed/guidance pairs)
            gen_dirs = list_child_dirs(method_concept_dir)
            
            for gen_dir in tqdm(gen_dirs, desc=f"  {method}/{concept}"):
                # Parse meta (Guidance, Seed)
                meta = parse_meta_from_name(gen_dir.name)
                if meta["guidance"] is None: continue # Skip if parsing failed
                
                # Compute Metrics
                metrics = evaluate_single_folder(
                    gen_dir, meta, device, 
                    args.fid_mode, args.match_real_count, args.batch_size, args.num_workers, args.max_pairs,
                    clip_model, clip_preproc, inception, real_features, real_dir,
                    args.kid_subset_size, args.kid_subsets
                )
                
                if metrics:
                    # Flatten into a dict for DataFrame
                    row = {
                        "method": method,
                        "concept": concept,
                        "guidance": meta["guidance"],
                        "seed": meta["seed"],
                        "steps": meta["steps"],
                        "folder_name": gen_dir.name
                    }
                    row.update(metrics)
                    all_results.append(row)

    # 3. Save Raw Data (All individual folder results)
    if not all_results:
        print("No results collected.")
        return

    df_raw = pd.DataFrame(all_results)
    raw_csv_path = save_dir / "all_concepts_raw_metrics.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\nSaved raw metrics to: {raw_csv_path}")

    # 4. Aggregation (Calculate Mean + 95% CI)
    # We group by Method and Guidance. The 'Concept' and 'Seed' are aggregated over.
    
    # Metrics to aggregate
    metric_cols = [
        "fid", "fid_clean", "kid_mean", "clip_score", 
        "vendi_pixel", "vendi_inception", "one_minus_ms_ssim", "brisque"
    ]
    valid_cols = [c for c in metric_cols if c in df_raw.columns]

    # GroupBy object
    grouped = df_raw.groupby(["method", "guidance"])[valid_cols]
    
    # Calculate Mean
    df_mean = grouped.mean()
    
    # Calculate 95% CI: 1.96 * (std / sqrt(n))
    # sem() computes std / sqrt(n)
    df_sem = grouped.sem()
    df_ci = df_sem * 1.96
    
    # Rename columns for output
    df_mean_renamed = df_mean.rename(columns={c: c+"_mean" for c in valid_cols})
    df_ci_renamed = df_ci.rename(columns={c: c+"_95ci" for c in valid_cols})
    
    # Concatenate
    df_final = pd.concat([df_mean_renamed, df_ci_renamed], axis=1)
    
    # Reorder columns nicely
    final_cols = []
    for c in valid_cols:
        final_cols.append(f"{c}_mean")
        final_cols.append(f"{c}_95ci")
    
    df_final = df_final[final_cols].reset_index()
    
    agg_csv_path = save_dir / "aggregated_stats_by_cfg.csv"
    df_final.to_csv(agg_csv_path, index=False)
    
    print("-" * 40)
    print(f"Aggregation Complete!")
    print(f"Aggregated Statistics (Mean + 95% CI) saved to:\n  {agg_csv_path}")
    print("-" * 40)
    print(df_final.head().to_string())

if __name__ == "__main__":
    main()