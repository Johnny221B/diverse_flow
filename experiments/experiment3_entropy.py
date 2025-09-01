#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export normalized cluster entropy (H/log k) for each {method, guidance} under one concept.

Inputs
------
- REAL_DIR: real reference images folder for ONE concept (e.g., .../real_cls_crops/truck/)
- OUTPUTS_ROOT: contains {method}_{concept}/imgs/{prompt}_seedXXXX_g{guidance}_s{steps}/
               (also supports old format: {prompt}_{seed}_{guidance}_{steps})
- PROMPTS_JSON: multi-class JSON; we only load prompts for the given --concept
- METHODS: comma-separated list
- GUIDANCES: e.g., "3.0,5.0,7.5"
- k (clusters): default 10
- steps: default 30

Aggregation
-----------
For each guidance:
  per-Kset entropy -> (seed-mean per prompt) -> macro-mean over prompts
Output one CSV containing rows for all {method, guidance} of the concept:
  columns: method, concept, guidance, k, entropy_mean, entropy_std
"""

import argparse, json, re, gc, difflib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from collections import Counter

# ===== CLIP features =====
import torch
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git
from sklearn.cluster import KMeans

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
def is_image(p: Path) -> bool: return p.suffix.lower() in IMG_EXTS
def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and is_image(p)]) if d.is_dir() else []

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---- normalization helpers ----
def _normalize_prompt(s: str, lower=True) -> str:
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() if lower else s
def _normalize_label(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def load_prompts_for_concept(path: Path, concept: str, lower=True) -> List[str]:
    """Load ONLY the prompts for the given concept from a multi-class JSON."""
    obj = json.load(open(path, "r"))
    key = _normalize_label(concept)

    def _norm(lst):
        seen, out = set(), []
        for p in lst:
            q = _normalize_prompt(str(p), lower=lower)
            if q not in seen:
                out.append(q); seen.add(q)
        return out

    if isinstance(obj, list):
        return _norm(obj)

    if isinstance(obj, dict) and "class_prompts" in obj and isinstance(obj["class_prompts"], dict):
        for k, v in obj["class_prompts"].items():
            if _normalize_label(k) == key:
                return _norm(v)

    if isinstance(obj, dict):
        if "prompts" in obj and isinstance(obj["prompts"], list):
            return _norm(obj["prompts"])
        for k, v in obj.items():
            if isinstance(v, list) and _normalize_label(k) == key:
                return _norm(v)

    # Fallback (not recommended): flatten all lists
    flat = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list): flat += v
    elif isinstance(obj, list):
        flat = obj
    return _norm(flat)

# ---- parse K-set dirnames (new & old formats) ----
NEW_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$")
OLD_RE = re.compile(r"^(?P<prompt>.+)_(?P<seed>\d+?)_(?P<guidance>[-+]?\d*\.?\d+?)_(?P<steps>\d+?)$")

def parse_kset_dirname(name: str):
    for pat in (NEW_RE, OLD_RE):
        m = pat.match(name)
        if m:
            return {
                "prompt_raw": m.group("prompt"),
                "seed": int(m.group("seed")),
                "guidance": float(m.group("guidance")),
                "steps": int(m.group("steps")),
            }
    return None

# ---- prefix + fuzzy matching of prompts ----
def match_prompt(folder_prompt_norm: str, allowed_prompts_norm: List[str], fuzzy_ratio: float) -> Tuple[bool, str]:
    # prefix first
    for a in allowed_prompts_norm:
        if folder_prompt_norm.startswith(a):
            return True, a
    # fuzzy
    best = None; best_score = 0.0
    for a in allowed_prompts_norm:
        score = difflib.SequenceMatcher(None, folder_prompt_norm, a).ratio()
        if score > best_score:
            best, best_score = a, score
    if best is not None and best_score >= fuzzy_ratio:
        return True, best
    return False, ""

def scan_ksets(gen_imgs_root: Path, allowed_prompts: List[str], target_guidances: List[float],
               steps: int, fuzzy_ratio: float, lower=True, debug=False) -> List[Dict]:
    allowed_norm = [_normalize_prompt(p, lower=lower) for p in allowed_prompts]
    gd_set = set([float(g) for g in target_guidances])

    out, unmatched = [], []
    for d in sorted(gen_imgs_root.iterdir()):
        if not d.is_dir(): continue
        info = parse_kset_dirname(d.name)
        if not info:
            unmatched.append((d.name, "regex_mismatch"))
            continue

        folder_prompt_norm = _normalize_prompt(info["prompt_raw"], lower=lower)
        ok, canon_prompt = match_prompt(folder_prompt_norm, allowed_norm, fuzzy_ratio)
        if not ok:
            unmatched.append((d.name, f"prompt_not_matched:{folder_prompt_norm}"))
            continue

        if info["guidance"] not in gd_set:  # guidance 过滤
            continue
        if info["steps"] != steps:          # steps 过滤
            continue

        imgs = list_images(d)
        if not imgs:
            unmatched.append((d.name, "no_images"))
            continue

        out.append({
            "dir": d,
            "folder": d.name,                 # 新增：文件夹名
            "prompt_raw": info["prompt_raw"], # 新增：原始 prompt 片段
            "prompt": canon_prompt,           # canonical prompt
            "seed": info["seed"],
            "guidance": info["guidance"],
            "steps": info["steps"],
            "images": imgs,
            "n_images": len(imgs),            # 新增：图片数
        })

    if debug:
        print(f"[DEBUG] scan_ksets: matched={len(out)}, unmatched={len(unmatched)}")
        for name, why in unmatched[:20]:
            print(f"  - UNMATCHED: {name}  | {why}")
        if len(unmatched) > 20:
            print(f"  ... ({len(unmatched)-20} more)")
    return out

    if debug:
        print(f"[DEBUG] scan_ksets: matched={len(out)}, unmatched={len(unmatched)}")
        for name, why in unmatched[:20]:
            print(f"  - UNMATCHED: {name}  | {why}")
        if len(unmatched) > 20:
            print(f"  ... ({len(unmatched)-20} more)")
    return out

# ---- CLIP featurizer & kmeans ----
class CLIPFeaturizer:
    def __init__(self, model_name="ViT-L/14@336px", device=None, batch_size=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.bs = batch_size
    @torch.no_grad()
    def encode_paths(self, paths: List[Path]) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), self.bs):
            batch = []
            for p in paths[i:i+self.bs]:
                im = Image.open(p).convert("RGB")
                batch.append(self.preprocess(im))
            x = torch.stack(batch).to(self.device)
            f = self.model.encode_image(x)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
        free_mem()
        return np.concatenate(feats, axis=0)

def kmeans_centers(real_feats: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(real_feats)
    C = km.cluster_centers_
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C

# ---- entropy on one K-set ----
def entropy_for_kset(gen_feats: np.ndarray, centers: np.ndarray) -> float:
    sims = gen_feats @ centers.T
    nearest = np.argmax(sims, axis=1)  # same as argmin of (1 - sims)
    k = centers.shape[0]
    counts = np.bincount(nearest, minlength=k).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    eps = 1e-12
    H = -(p * np.log(p + eps)).sum()
    H_norm = H / (np.log(k) + eps)
    return float(H_norm)

# ---- aggregate: seed-mean per prompt -> macro over prompts ----
def aggregate_entropy(by_g_p_seed: Dict[float, Dict[str, List[float]]]) -> Dict[float, Dict[str, float]]:
    out = {}
    for g, mp in by_g_p_seed.items():
        per_prompt_mean = []
        for p, lst in mp.items():
            per_prompt_mean.append(np.mean(lst))
        ent_mu = float(np.mean(per_prompt_mean))
        ent_sd = float(np.std (per_prompt_mean))
        out[g] = {"entropy_mean": ent_mu, "entropy_std": ent_sd}
    return out

# ---- main driver for one concept ----
def run_for_concept(real_dir: Path, outputs_root: Path, concept: str,
                    methods: List[str], prompts: List[str],
                    guidances: List[float], steps: int,
                    k: int, batch_size: int, device: str,
                    fuzzy_ratio: float, out_dir: Path, debug: bool,
                    show_matched:bool, dump_matched: Path):

    out_dir.mkdir(parents=True, exist_ok=True)
    feat = CLIPFeaturizer(device=device, batch_size=batch_size)

    # real feats & kmeans
    real_imgs = list_images(real_dir)
    assert len(real_imgs) > 0, f"No real images in {real_dir}"
    real_feats = feat.encode_paths(real_imgs)
    centers = kmeans_centers(real_feats, k=k, seed=0)

    rows = []  # to CSV

    for method in methods:
        gen_root = outputs_root / f"{method}_{concept}" / "imgs"
        if not gen_root.exists():
            print(f"[WARN] missing: {gen_root} — skip method {method}")
            continue

        ksets = scan_ksets(gen_root, prompts, guidances, steps,
                           fuzzy_ratio=fuzzy_ratio, lower=True, debug=debug)
        if not ksets:
            print(f"[WARN] no K-sets under {gen_root} for given prompts/guidances")
            continue
        
        if ksets:
            # 统计
            c_guid = Counter([ks["guidance"] for ks in ksets])
            c_prompt = Counter([ks["prompt"]   for ks in ksets])
            total_imgs = sum(ks["n_images"] for ks in ksets)

            print(f"[INFO] {concept}|{method}: matched K-sets = {len(ksets)} "
                f"(images={total_imgs}, by_guidance={dict(sorted(c_guid.items()))})")

            if show_matched:
                print("[MATCHED] folder | prompt_raw -> prompt | seed | g | s | #imgs")
                for ks in sorted(ksets, key=lambda x: (x["guidance"], x["prompt"], x["seed"], x["folder"])):
                    print(f"  - {ks['folder']} | {ks['prompt_raw']} -> {ks['prompt']} | "
                        f"{ks['seed']} | {ks['guidance']} | {ks['steps']} | {ks['n_images']}")

            if dump_matched is not None:
                dump_dir = Path(dump_matched)
                dump_dir.mkdir(parents=True, exist_ok=True)
                import pandas as _pd
                _dfm = _pd.DataFrame([{
                    "method": method,
                    "concept": concept,
                    "folder": ks["folder"],
                    "prompt_raw": ks["prompt_raw"],
                    "prompt": ks["prompt"],
                    "seed": ks["seed"],
                    "guidance": ks["guidance"],
                    "steps": ks["steps"],
                    "n_images": ks["n_images"],
                    "abs_dir": str(ks["dir"])
                } for ks in ksets]).sort_values(["guidance","prompt","seed","folder"])
                out_csv_list = dump_dir / f"matched_{method}_{concept}.csv"
                _dfm.to_csv(out_csv_list, index=False)
                print(f"[SAVE] matched list -> {out_csv_list}")

        print(f"[INFO] {concept}|{method}: matched K-sets = {len(ksets)}")
        # g -> prompt -> list of entropy over seeds
        by_g_p_seed: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for ks in tqdm(ksets, desc=f"[{concept}|{method}] K-sets"):
            gen_feats = feat.encode_paths(ks["images"])
            ent = entropy_for_kset(gen_feats, centers)
            by_g_p_seed[ks["guidance"]][ks["prompt"]].append(ent)

        stats = aggregate_entropy(by_g_p_seed)
        for g, st in stats.items():
            rows.append({
                "method": method,
                "concept": concept,
                "guidance": float(g),
                "k": int(k),
                "entropy_mean": float(st["entropy_mean"]),
                "entropy_std":  float(st["entropy_std"]),
            })

    if not rows:
        print("[WARN] nothing to save; no rows collected.")
        return

    df = pd.DataFrame(rows).sort_values(by=["method","guidance"]).reset_index(drop=True)
    out_csv = out_dir / f"exp3_entropy_{concept}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv} (rows={len(df)})")

# ---- CLI ----
def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=Path, required=True, help="real reference folder for ONE concept")
    ap.add_argument("--outputs_root", type=Path, required=True, help="root containing {method}_{concept}/imgs/")
    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--methods", type=str, required=True, help="comma-separated methods")
    ap.add_argument("--prompts_json", type=Path, required=True, help="multi-class prompts JSON")
    ap.add_argument("--guidances", type=str, required=True, help="comma-separated guidances, e.g. '3.0,5.0,7.5'")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fuzzy_ratio", type=float, default=0.85)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--show_matched", type=int, default=1)
    ap.add_argument("--dump_matched", type=Path, default=None)
    args = ap.parse_args()

    prompts = load_prompts_for_concept(args.prompts_json, args.concept, lower=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    guidances = parse_floats(args.guidances)

    run_for_concept(
        real_dir=args.real_dir,
        outputs_root=args.outputs_root,
        concept=args.concept,
        methods=methods,
        prompts=prompts,
        guidances=guidances,
        steps=args.steps,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
        fuzzy_ratio=float(args.fuzzy_ratio),
        out_dir=args.out_dir,
        debug=bool(args.debug),
        show_matched=bool(args.show_matched),
        dump_matched=args.dump_matched
    )

if __name__ == "__main__":
    main()
