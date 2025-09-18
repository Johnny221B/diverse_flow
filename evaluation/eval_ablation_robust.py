#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate metrics for two new storage layouts:

1) Ablation study
   outputs/ablation/<method>/{imgs,eval}/<prompt>_seedXXXX_gY.s_sZ/...
   -> Write one CSV per method at: outputs/ablation/<method>/eval/ablation_<method>.csv

2) Robust study
   outputs/robust_study/<group>/{imgs,eval}/<prefix>_<prompt>_seedXXXX_g3.0_s30/...
   where:
     group ∈ {lambda, alpha, noise_gate}
     prefix ∈ {lambda0.60, alpha0.50, ng0.50-0.95}
   -> Write one CSV per value (per group) at:
      outputs/robust_study/<group>/eval/<group>_<value>.csv

Metrics: vendi(pixel+embedding), FID (piq / cleanfid), CLIP cosine (OpenAI CLIP JIT),
         1 - MS-SSIM (diversity), BRISQUE (quality).

Dependencies are optional and gracefully skipped if missing.
"""

import argparse, os, re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
from collections import defaultdict
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Optional deps
try:
    import piq
    from piq.feature_extractors import InceptionV3
except ImportError:
    piq = None
    InceptionV3 = None

# only for tokenization fallback (model not used)
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

# ------------------------- Common utils -------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# base pattern used by both layouts (right part of run folder name)
META_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")

# robust prefix: lambda<val> | alpha<val> | ng<a>-<b>
ROBUST_PREFIX_RE = re.compile(
    r"^(?:(lambda(?P<lam>[0-9.]+))|(alpha(?P<alp>[0-9.]+))|(ng(?P<nga>[0-9.]+)-(?P<ngb>[0-9.]+)))_(?P<rest>.+)$"
)

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def list_child_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def parse_meta_from_name(name: str) -> Dict[str, Any]:
    """
    Parse "<prompt>_seedS_gG_sT" into dict fields.
    """
    m = META_RE.match(name)
    if not m:
        return {"prompt": name.replace("_", " "), "seed": None, "guidance": None, "steps": None}
    d = m.groupdict()
    d["prompt"]   = d["prompt"].replace("_", " ")
    d["seed"]     = int(d["seed"])
    d["guidance"] = float(d["guidance"])
    d["steps"]    = int(d["steps"])
    return d

def parse_meta_robust_folder(name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse robust run folder:
        "<prefix>_<prompt>_seedS_gG_sT"
    where prefix is one of:
        lambda<val> | alpha<val> | ng<a>-<b>
    Returns: (prefix_info, base_meta)
      prefix_info keys: group ∈ {lambda, alpha, noise_gate}, value(str),
                        value_num (float) or (float,float) for ng
    """
    m = ROBUST_PREFIX_RE.match(name)
    if not m:
        # Fallback: no prefix; treat as unknown group
        return {"group": "unknown", "value": "NA", "value_num": None}, parse_meta_from_name(name)
    gd = m.groupdict()
    rest = gd["rest"]
    if gd.get("lam"):
        val = gd["lam"]
        pinfo = {"group": "lambda", "value": f"{float(val):.2f}", "value_num": float(val)}
    elif gd.get("alp"):
        val = gd["alp"]
        pinfo = {"group": "alpha", "value": f"{float(val):.2f}", "value_num": float(val)}
    else:
        a, b = float(gd["nga"]), float(gd["ngb"])
        pinfo = {"group": "noise_gate", "value": f"{a:.2f}-{b:.2f}", "value_num": (a, b)}
    return pinfo, parse_meta_from_name(rest)

# ------------------------- Metrics -------------------------

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

# --------- FID ----------
def _build_inception_extractor(device: torch.device):
    if piq is None or InceptionV3 is None:
        print("Warning: piq not installed; FID(piq) will be skipped.")
        return None
    return InceptionV3().to(device)

def _make_fid_transform():
    return T.Compose([
        T.Resize(299, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
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
        for imgs in tqdm(loader, desc=f"Extracting features: {Path(paths[0]).parent.name}"):
            feats = extractor(imgs.to(device))
            if isinstance(feats, list): feats = feats[0]
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).squeeze(-1).squeeze(-1)
            feats_all.append(feats.cpu())
    return torch.cat(feats_all, dim=0) if feats_all else torch.empty(0, 2048)

def compute_fid_from_features(real_features: torch.Tensor, fake_features: torch.Tensor) -> Optional[float]:
    if piq is None or real_features.numel()==0 or fake_features.numel()==0: return None
    return float(piq.FID()(real_features, fake_features))

def compute_fid_cleanfid(real_dir: Path, gen_dir: Path, mode="clean") -> Optional[float]:
    if cleanfid is None:
        print("cleanfid not installed; skip clean-FID.")
        return None
    try:
        return float(cleanfid.compute_fid(str(real_dir), str(gen_dir), mode=mode, model_name="inception_v3"))
    except Exception as e:
        print(f"clean-fid failed on {gen_dir}: {e}")
        return None

# --------- CLIP cosine (JIT only) ----------
def load_clip_from_jit(jit_path: Path, image_size: int = 224):
    model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    preprocess = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess

def calculate_clip_cosine(folder: Path, model, preprocess, device, batch_size,
                          text: Optional[str] = None, tokenizer=None):
    image_paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not image_paths:
        return None

    prompt = text
    if prompt is None:
        meta = parse_meta_from_name(folder.name)
        prompt = meta.get("prompt") or folder.name.replace("_"," ").replace("-"," ")

    if tokenizer is None:
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
        tokenizer = tok

    if tokenizer is None:
        print("No tokenizer available (open_clip or clip). Skip CLIP cosine.")
        return None

    with torch.inference_mode():
        t = tokenizer([prompt]).to(device)
        tfeat = model.encode_text(t)
        tfeat = tfeat / (tfeat.norm(dim=-1, keepdim=True) + 1e-12)

        sims = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"CLIP cosine: {folder.name}"):
            ims = []
            for p in image_paths[i:i+batch_size]:
                try:
                    ims.append(preprocess(Image.open(p).convert("RGB")))
                except Exception:
                    continue
            if not ims:
                continue
            x = torch.stack(ims).to(device=device, dtype=torch.float32)
            if not hasattr(model, "encode_image"):
                print("JIT model has no encode_image; skip.")
                return None
            feat = model.encode_image(x)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
            sim = (100.0 * feat @ tfeat.T).squeeze(-1).detach().cpu().numpy()
            sims.extend(sim.tolist())

    return float(np.mean(sims)) if sims else None

# --------- MS-SSIM ----------
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
    for p1, p2 in tqdm(pairs, desc=f"MS-SSIM: {folder.name}"):
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
                s = piq.multi_scale_ssim(x1, x2, data_range=1.0, reduction="none")
                score = float(s.item())
            elif ms_ssim_torch is not None:
                score = float(ms_ssim_torch(x1, x2, data_range=1.0).item())
            else:
                print("Neither piq nor pytorch_msssim installed; skip MS-SSIM.")
                return None
        scores.append(score)
    if not scores: return None
    return float(1.0 - np.mean(scores))

# --------- BRISQUE ----------
def calculate_brisque_quality(folder: Path, device: torch.device) -> Optional[float]:
    if piq is None:
        print("piq not installed; skip BRISQUE.")
        return None
    paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png"))
    if not paths: return None
    tfm = T.Compose([T.ToTensor()])
    metric = piq.BRISQUELoss(data_range=1.0, reduction='none')
    scores = []
    for p in tqdm(paths, desc=f"BRISQUE: {folder.name}"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            s = metric(x)
        scores.append(float(s.item()))
    return float(np.mean(scores)) if scores else None

# ------------------------- Evaluation core -------------------------

COMMON_COLUMNS = [
    "folder","prompt","seed","guidance","steps","num_images",
    "vendi_pixel","vendi_inception","fid","fid_clean",
    "clip_score","one_minus_ms_ssim","brisque"
]

def eval_one_dir(gen_dir: Path, device: torch.device, fid_mode: str, real_dir: Optional[Path],
                 match_real_count: bool, batch_size: int, num_workers: int, max_pairs: int,
                 clip_model, clip_preproc, inception, real_features) -> Optional[Dict[str, Any]]:
    paths = list_images(gen_dir)
    if not paths:
        print(f"[skip-empty] {gen_dir}")
        return None

    meta = parse_meta_from_name(gen_dir.name)

    # Vendi
    try:
        imgs = [Image.open(p).convert("RGB") for p in paths]
    except Exception:
        imgs = []
    if not imgs:
        print(f"[skip-broken] cannot load images in {gen_dir}")
        return None
    vendi_scores = compute_vendi_for_images(imgs, device=str(device))

    # FID (piq)
    fid_value = None
    if fid_mode in ("piq","both") and (inception is not None) and (real_features is not None) and (real_features.numel()>0):
        fake_feats = _extract_inception_features(paths, inception, device, num_workers=num_workers, batch_size=batch_size)
        real_feats_used = real_features
        if match_real_count and real_features.shape[0] >= fake_feats.shape[0] and fake_feats.shape[0] > 0:
            rng = np.random.default_rng(abs(hash(gen_dir.name)) % (2**32))
            idx = rng.choice(real_features.shape[0], size=fake_feats.shape[0], replace=False)
            real_feats_used = real_features[idx]
        fid_value = compute_fid_from_features(real_feats_used, fake_feats)

    # FID (clean-fid)
    fid_clean = None
    if fid_mode in ("clean","both") and real_dir is not None:
        fid_clean = compute_fid_cleanfid(real_dir, gen_dir, mode="clean")

    # CLIP cosine (JIT only)
    clip_cos = None
    if (clip_model is not None) and (clip_preproc is not None):
        clip_text = meta.get("prompt")
        clip_cos = calculate_clip_cosine(
            gen_dir, clip_model, clip_preproc, device, batch_size,
            text=clip_text, tokenizer=None
        )

    # MS-SSIM diversity
    msssim_div = calculate_ms_ssim_diversity(gen_dir, max_pairs=max_pairs, device=device)

    # BRISQUE
    brisque_quality = calculate_brisque_quality(gen_dir, device=device)

    return {
        "folder": gen_dir.name,
        "prompt": meta.get("prompt"),
        "seed": meta.get("seed"),
        "guidance": meta.get("guidance"),
        "steps": meta.get("steps"),
        "num_images": len(imgs),
        "vendi_pixel": vendi_scores.get("vendi_pixel"),
        "vendi_inception": vendi_scores.get("vendi_inception"),
        "fid": fid_value,
        "fid_clean": fid_clean,
        "clip_score": clip_cos,
        "one_minus_ms_ssim": msssim_div,
        "brisque": brisque_quality,
    }

# ------------------------- Ablation evaluation -------------------------

def evaluate_ablation(outputs_root: Path, methods: List[str], device: torch.device,
                      fid_mode: str, real_dir: Optional[Path], match_real_count: bool,
                      batch_size: int, num_workers: int, max_pairs: int,
                      clip_model, clip_preproc, inception, real_features):
    ablation_root = outputs_root / "ablation"
    if not ablation_root.exists():
        print(f"[warn] ablation root not found: {ablation_root}")
        return

    for method in methods:
        mroot = ablation_root / method
        imgs_root = mroot / "imgs"
        eval_dir  = mroot / "eval"
        if not imgs_root.exists():
            print(f"[skip] {imgs_root} not found.")
            continue
        eval_dir.mkdir(parents=True, exist_ok=True)

        gen_dirs = [d for d in list_child_dirs(imgs_root) if list_images(d)]
        if not gen_dirs:
            print(f"[skip] No image subfolders under: {imgs_root}")
            continue

        csv_path = eval_dir / f"ablation_{method}.csv"
        print(f"[ablation:{method}] writing -> {csv_path}")

        columns = ["method"] + COMMON_COLUMNS
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for d in gen_dirs:
                row = eval_one_dir(
                    d, device, fid_mode, real_dir, match_real_count,
                    batch_size, num_workers, max_pairs,
                    clip_model, clip_preproc, inception, real_features
                )
                if row is None: continue
                row["method"] = method
                w.writerow(row)

        print(f"[ablation:{method}] done -> {csv_path}")

# ------------------------- Robust evaluation -------------------------

def safe_value_for_filename(val: str) -> str:
    # keep hyphen; replace multiple dots to single safe representation
    return val.replace('.', '_')

def evaluate_robust(outputs_root: Path, groups: List[str], device: torch.device,
                    fid_mode: str, real_dir: Optional[Path], match_real_count: bool,
                    batch_size: int, num_workers: int, max_pairs: int,
                    clip_model, clip_preproc, inception, real_features):
    robust_root = outputs_root / "robust_study"
    if not robust_root.exists():
        print(f"[warn] robust_study root not found: {robust_root}")
        return

    for group in groups:
        groot = robust_root / group
        imgs_root = groot / "imgs"
        eval_dir  = groot / "eval"
        if not imgs_root.exists():
            print(f"[skip] {imgs_root} not found.")
            continue
        eval_dir.mkdir(parents=True, exist_ok=True)

        gen_dirs = [d for d in list_child_dirs(imgs_root) if list_images(d)]
        if not gen_dirs:
            print(f"[skip] No image subfolders under: {imgs_root}")
            continue

        # 将 run 目录按“取值”分桶
        buckets: DefaultDict[str, List[Path]] = defaultdict(list)
        for d in gen_dirs:
            pinfo, base = parse_meta_robust_folder(d.name)
            if pinfo["group"] != group:
                # 若目录层级与前缀不符，仍按该 group 处理，但记录到 value=NA
                buckets["NA"].append(d)
            else:
                buckets[pinfo["value"]].append(d)

        # 每个取值单独写一个 CSV
        for val, dirs in sorted(buckets.items(), key=lambda kv: kv[0]):
            val_tag = safe_value_for_filename(val)
            csv_path = eval_dir / f"{group}_{val_tag}.csv"
            print(f"[robust:{group}] value={val} -> {csv_path}")

            # 列包含 group 与 value
            columns = ["group","value"] + COMMON_COLUMNS
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader()
                for d in dirs:
                    row = eval_one_dir(
                        d, device, fid_mode, real_dir, match_real_count,
                        batch_size, num_workers, max_pairs,
                        clip_model, clip_preproc, inception, real_features
                    )
                    if row is None: continue
                    row["group"] = group
                    row["value"] = val
                    w.writerow(row)

            print(f"[robust:{group}] value={val} done -> {csv_path}")

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate metrics for ablation/robust study layouts.")
    ap.add_argument("--study", choices=["ablation","robust","both"], default="both",
                    help="Which study layout to evaluate.")
    ap.add_argument("--outputs-root", type=str, default="./outputs",
                    help="Root outputs dir that contains 'ablation' and/or 'robust_study'.")
    ap.add_argument("--real-dir", type=str, default=None,
                    help="Path to real images dir for FID. If omitted, FID(clean) is skipped; "
                         "FID(piq) also requires --fid-mode piq/both and Inception features.")
    # Filters
    ap.add_argument("--methods", nargs="+", default=["wo_OP","wo_LR","ourmethod"],
                    help="For ablation: which methods to include (default: detect common three).")
    ap.add_argument("--groups", nargs="+", default=["lambda","alpha","noise_gate"],
                    help="For robust: which groups to include.")
    # Device & dataloading
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=100)
    # FID
    ap.add_argument("--fid-mode", choices=["piq","clean","both","none"], default="both")
    ap.add_argument("--match-real-count", action="store_true")
    # CLIP
    ap.add_argument("--clip-jit", type=str, default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    ap.add_argument("--clip-image-size", type=int, default=224)

    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    outputs_root = Path(args.outputs_root).resolve()
    real_dir = Path(args.real_dir).resolve() if args.real_dir else None
    if (args.fid_mode in ("clean","both")) and real_dir is None:
        print("[note] --real-dir not provided: clean-FID will be skipped.")
    if (args.fid_mode in ("piq","both")) and (piq is None):
        print("[note] piq not installed: FID(piq) will be skipped.")

    # Build shared tools/models (reused across evaluations)
    inception = None
    real_features = None
    if args.fid_mode in ("piq","both"):
        inception = _build_inception_extractor(device)
        if inception is not None and real_dir is not None and real_dir.exists():
            real_paths = list_images(real_dir)
            if real_paths:
                real_features = _extract_inception_features(real_paths, inception, device,
                                                            num_workers=args.num_workers, batch_size=args.batch_size)
            else:
                print("[note] No real images found; FID(piq) will be skipped.")
                inception = None

    # CLIP setup: OpenAI JIT only
    clip_model = None
    clip_preproc = None
    jit_path = Path(args.clip_jit) if args.clip_jit else None
    if jit_path and jit_path.exists():
        try:
            clip_model, clip_preproc = load_clip_from_jit(jit_path, image_size=args.clip_image_size)
            clip_model = clip_model.to(device)
            print(f"Loaded CLIP JIT from {jit_path} (image_size={args.clip_image_size})")
        except Exception as e:
            print(f"Failed to load CLIP JIT: {e}")
    else:
        print(f"[note] CLIP JIT not found at {jit_path}; CLIP cosine will be skipped.")

    # Run
    if args.study in ("ablation","both"):
        evaluate_ablation(
            outputs_root=outputs_root,
            methods=args.methods,
            device=device,
            fid_mode=args.fid_mode if args.fid_mode!="none" else "none",
            real_dir=real_dir if args.fid_mode in ("clean","both") else None,
            match_real_count=args.match_real_count,
            batch_size=args.batch_size, num_workers=args.num_workers, max_pairs=args.max_pairs,
            clip_model=clip_model, clip_preproc=clip_preproc,
            inception=inception, real_features=real_features
        )

    if args.study in ("robust","both"):
        evaluate_robust(
            outputs_root=outputs_root,
            groups=args.groups,
            device=device,
            fid_mode=args.fid_mode if args.fid_mode!="none" else "none",
            real_dir=real_dir if args.fid_mode in ("clean","both") else None,
            match_real_count=args.match_real_count,
            batch_size=args.batch_size, num_workers=args.num_workers, max_pairs=args.max_pairs,
            clip_model=clip_model, clip_preproc=clip_preproc,
            inception=inception, real_features=real_features
        )

if __name__ == "__main__":
    main()
