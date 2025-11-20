#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 1 (PRD & AUC-PRD) evaluator (Adaptive n_bins version).
Fixes the "jagged curve" issue for small datasets (e.g., 32 images).
"""

import argparse, json, re, gc, sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== CLIP =====
import torch
import torch.nn.functional as F
import clip
from sklearn.cluster import KMeans

# ----------------- utils -----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
def is_image(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS
def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and is_image(p)]) if d.is_dir() else []

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --------- K-set Parser ---------
KSET_RE = re.compile(
    r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$"
)

def scan_ksets_all(imgs_root: Path, print_found: bool=False) -> List[Dict]:
    out = []
    if not imgs_root.exists():
        return out
    for d in sorted(imgs_root.iterdir()):
        if not d.is_dir(): continue
        m = KSET_RE.match(d.name)
        if not m: continue
        imgs = list_images(d)
        if not imgs: continue
        out.append({
            "dir": d,
            "prompt": m.group("prompt"),
            "seed": int(m.group("seed")),
            "guidance": float(m.group("guidance")),
            "steps": int(m.group("steps")),
            "images": imgs
        })
    if print_found:
        print(f"[DEBUG] Found {len(out)} K-sets under {imgs_root}")
    return out

# --------- CLIP Featurizer ---------
class CLIPFeaturizer:
    def __init__(self, model_name="ViT-B/32", device=None, batch_size=64,
                 download_root: str=None, jit: bool=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.preprocess = clip.load(
                model_name, device=self.device, jit=jit, download_root=download_root
            )
        except Exception as e:
            print(f"[ERROR] clip.load failed: {e}", file=sys.stderr)
            raise
        self.model.eval()
        self.bs = batch_size
        print(f"[INFO] CLIP = {model_name} | device={self.device}")

    @torch.no_grad()
    def encode_paths(self, paths: List[Path]) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), self.bs):
            batch = []
            for p in paths[i:i+self.bs]:
                try:
                    im = Image.open(p).convert("RGB")
                    batch.append(self.preprocess(im))
                except Exception as e:
                    print(f"[WARN] Error reading {p}: {e}")
            if not batch: continue
            x = torch.stack(batch).to(self.device)
            f = self.model.encode_image(x)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        free_mem()
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)

# --------- PRD Core (Adaptive n_bins) ---------
def prd_from_hist(p: np.ndarray, q: np.ndarray, lambdas: np.ndarray):
    p = np.clip(p, 1e-12, None); p /= p.sum()
    q = np.clip(q, 1e-12, None); q /= q.sum()
    prec, reca = [], []
    for lam in lambdas:
        prec.append(np.minimum(lam * p, q).sum())
        reca.append(np.minimum(p, q / lam).sum())
    return np.asarray(prec), np.asarray(reca)

def compute_prd_curve(real_feats: np.ndarray, gen_feats: np.ndarray,
                      target_n_bins: int = 50, # 用户期望的 bin 数量
                      n_lambdas: int = 151, random_state: int = 0):
    
    if real_feats.size == 0 or gen_feats.size == 0:
        lambdas = np.exp(np.linspace(np.log(1e-3), np.log(1e3), n_lambdas))
        return np.zeros_like(lambdas), np.zeros_like(lambdas)
    
    # [FIX] 自适应调整 n_bins
    # 规则：n_bins 不能超过总样本数的 1/2，否则聚类会变成“点对点”匹配
    n_samples = min(len(real_feats), len(gen_feats))
    
    # 如果样本量极小（比如32），sqrt(32) ≈ 5.6 -> 使用 5 或 6
    # 如果样本量大（比如10000），使用 target_n_bins (50 或 100)
    adaptive_bins = int(np.sqrt(n_samples))
    
    # 取两者中的较小值，并确保至少为 2
    final_n_bins = max(2, min(target_n_bins, adaptive_bins))
    
    if final_n_bins != target_n_bins:
        # 只在第一次或变化时打印警告，避免刷屏 (这里简单打印)
        pass 
        # print(f"[AUTO-FIX] Sample size={n_samples} is too small for n_bins={target_n_bins}. Adjusted to {final_n_bins}.")

    X = np.concatenate([real_feats, gen_feats], axis=0)
    
    km = KMeans(n_clusters=final_n_bins, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X)
    
    r_labels = labels[:len(real_feats)]
    g_labels = labels[len(real_feats):]
    
    p = np.bincount(r_labels, minlength=final_n_bins).astype(np.float64)
    q = np.bincount(g_labels, minlength=final_n_bins).astype(np.float64)
    
    lambdas = np.exp(np.linspace(np.log(1e-3), np.log(1e3), n_lambdas))
    P, R = prd_from_hist(p, q, lambdas)
    return P, R

def auc_prd(P: np.ndarray, R: np.ndarray) -> float:
    if P.size == 0 or R.size == 0: return 0.0
    order = np.argsort(R)
    x = R[order]
    y = P[order]
    return float(np.trapz(y, x))

def recall_at_precision(P: np.ndarray, R: np.ndarray, targets=(0.60,0.70,0.76)) -> Dict[float,float]:
    if P.size == 0 or R.size == 0:
        return {float(t): 0.0 for t in targets}
    order = np.argsort(P)
    x = P[order]; y = R[order]
    out = {}
    for t in targets:
        t = float(np.clip(t, 0.0, 1.0))
        if t <= x[0]: out[t] = float(y[0]); continue
        if t >= x[-1]: out[t] = float(y[-1]); continue
        j = np.searchsorted(x, t)
        x0,x1 = x[j-1],x[j]; y0,y1 = y[j-1],y[j]
        out[t] = float(y0 + (y1 - y0) * ((t - x0) / max(x1 - x0, 1e-12)))
    return out

# --------- Main Logic ---------
def run_for_method(args, clipper):
    imgs_root = args.imgs_root
    out_dir = args.out_dir
    
    print(f"Scan: {imgs_root}")
    real_imgs = list_images(args.real_dir)
    if not real_imgs:
        print(f"[ERROR] No real images in {args.real_dir}")
        return
    real_feats = clipper.encode_paths(real_imgs)
    
    ksets = scan_ksets_all(imgs_root, print_found=args.print_found)
    if not ksets:
        print(f"[WARN] No K-sets found.")
        return

    by_g_p = defaultdict(lambda: defaultdict(list))

    for ks in tqdm(ksets, desc=f"[{args.concept}] Eval"):
        if ks["steps"] != args.steps: continue
        
        gen_feats = clipper.encode_paths(ks["images"])
        
        # [CRITICAL] Here we pass the desired 50, but function will auto-lower it
        P, R = compute_prd_curve(real_feats, gen_feats, target_n_bins=50) 
        
        auc = auc_prd(P, R)
        iso = recall_at_precision(P, R)
        by_g_p[ks["guidance"]][ks["prompt"]].append((P,R,auc,iso))

    # Aggregation
    for g, mp in by_g_p.items():
        prompt_stats = []
        for p_str, curves in mp.items():
            if not curves: continue
            # Just average metrics for simplicity in aggregation
            auc_mu = float(np.mean([c[2] for c in curves]))
            iso_mu = {k: float(np.mean([c[3][k] for c in curves])) for k in (0.60,0.70,0.76)}
            
            # For P, R curves, we need to interpolate to average them properly?
            # Actually, standard PRD approach: average the histograms OR average the curves.
            # Here we just take the curve of the FIRST seed as reference for Recall grid (simplified)
            # Better: Linear interpolate to a fixed grid.
            
            # Simple averaging of raw curves (works if binning is stable-ish, but curves vary)
            # Let's just output the data for the first seed or average if lengths match
            # To handle different x-axis, we strictly should interp.
            # For this script, we assume standard behavior.
            
            Ps = np.stack([c[0] for c in curves], 0) # shape [n_seeds, 151]
            Rs = np.stack([c[1] for c in curves], 0) # shape [n_seeds, 151]
            # PRD usually computed on fixed lambdas, so P and R arrays correspond 1-to-1 across seeds?
            # Yes, lambdas are fixed in compute_prd_curve. 
            # So R and P vectors are aligned by lambda index, NOT by value.
            # Averaging P and R at each lambda index is valid.
            
            P_mu = Ps.mean(0)
            P_sd = Ps.std(0)
            R_ref = Rs.mean(0) # Average Recall as well
            
            prompt_stats.append((R_ref, P_mu, P_sd, auc_mu, iso_mu))

        if not prompt_stats: continue

        # Aggregate across prompts
        R_ref = prompt_stats[0][0] # Use first prompt's R grid approx
        P_mu = np.stack([t[1] for t in prompt_stats], 0).mean(0)
        P_sd = np.stack([t[2] for t in prompt_stats], 0).mean(0)
        auc_mu = float(np.mean([t[3] for t in prompt_stats]))
        r60 = float(np.mean([t[4][0.60] for t in prompt_stats]))
        r70 = float(np.mean([t[4][0.70] for t in prompt_stats]))
        r76 = float(np.mean([t[4][0.76] for t in prompt_stats]))

        rows = []
        for rr, pp_mu, pp_sd in zip(R_ref, P_mu, P_sd):
            rows.append({
                "method": args.method_name, # Use a simpler key
                "concept": args.concept,
                "guidance": g,
                "recall": float(rr),
                "precision_mu": float(pp_mu),
                "precision_sd": float(pp_sd),
                "auc_prd_mu": auc_mu,
                "rec_at_p0.60_mu": r60,
                "rec_at_p0.70_mu": r70,
                "rec_at_p0.76_mu": r76,
            })
        
        df = pd.DataFrame(rows).sort_values(by="recall")
        out_csv = out_dir / f"exp1_{g}_{args.concept}.csv"
        if args.append_csv and out_csv.exists():
            df.to_csv(out_csv, mode="a", header=False, index=False)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv} (bins used: auto)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--methods", required=True) # "dpp,pg"
    ap.add_argument("--outputs_root", type=Path, required=True)
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--append_csv", type=int, default=0) # Default 0 (overwrite) is safer for debugging
    ap.add_argument("--print_found", type=int, default=0)
    ap.add_argument("--clip_model_name", default="ViT-B/32")
    ap.add_argument("--clip_download_root", default="~/.cache/clip")
    
    args = ap.parse_args()
    
    clip_root = str(Path(args.clip_download_root).expanduser())
    clipper = CLIPFeaturizer(args.clip_model_name, args.device, download_root=clip_root)
    
    meths = [m.strip() for m in args.methods.split(",") if m.strip()]
    
    for m in meths:
        # Setup paths
        real_dir = args.dataset_root / args.concept
        if not real_dir.exists(): real_dir = args.dataset_root / args.concept.replace(" ","_")
        
        imgs_root = args.outputs_root / f"{m}_{args.concept}" / "imgs"
        out_dir = args.outputs_root / f"{m}_{args.concept}" / "eval"
        
        # Inject args
        args.real_dir = real_dir
        args.imgs_root = imgs_root
        args.out_dir = out_dir
        args.method_name = m
        
        run_for_method(args, clipper)

if __name__ == "__main__":
    main()