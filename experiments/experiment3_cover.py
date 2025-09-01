#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E3 Figure 1: Coverage–τ curves (k=10, τ in {0.3,0.45,0.6,0.75,0.9})

目录假设：
  REAL_DIR/                        # 单个 concept 的真实图像文件夹（裁剪或整图均可）
  OUTPUTS_ROOT/{method}_{concept}/imgs/
      {prompt}_seed1111_g3.0_s30/  # ← 兼容这种新格式（也兼容老格式 {prompt}_{seed}_{guidance}_{steps}）
      ...

Prompt 匹配：
  - 先把 JSON 和目录里的 prompt 都“规范化”（空格→下划线，合并多下划线，去首尾下划线，小写）
  - 用“前缀匹配 + 模糊匹配（difflib ratio ≥ 阈值）”，能稳住轻微拼写错误（如 a_trcuk）

输出（按 guidance 一份）：
  out_dir/exp3_{guidance}_{concept}.csv
  列：method,concept,guidance,k,tau,coverage_mean,coverage_std
"""

import argparse, json, re, gc, difflib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ===== CLIP 特征 =====
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

# -------- 规范化 --------
def _normalize_prompt(s: str, lower=True) -> str:
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() if lower else s

def _normalize_label(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def load_prompts_for_concept(path: Path, concept: str, lower=True) -> List[str]:
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

    # 兜底 flatten（不推荐）
    flat = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list): flat += v
    elif isinstance(obj, list):
        flat = obj
    return _norm(flat)

# -------- 解析目录名：支持两种格式 --------
# 新：{prompt}_seed{seed}_g{guidance}_s{steps}
NEW_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$")
# 旧：{prompt}_{seed}_{guidance}_{steps}
OLD_RE = re.compile(r"^(?P<prompt>.+)_(?P<seed>\d+?)_(?P<guidance>[-+]?\d*\.?\d+?)_(?P<steps>\d+?)$")

def parse_kset_dirname(name: str):
    for pat, tag in ((NEW_RE, "new"), (OLD_RE, "old")):
        m = pat.match(name)
        if m:
            return {
                "prompt_raw": m.group("prompt"),
                "seed": int(m.group("seed")),
                "guidance": float(m.group("guidance")),
                "steps": int(m.group("steps")),
                "fmt": tag
            }
    return None

# -------- 前缀 + 模糊匹配 --------
def match_prompt(folder_prompt_norm: str, allowed_prompts_norm: List[str], fuzzy_ratio: float) -> Tuple[bool, str]:
    """
    返回 (是否匹配成功, 匹配到的 canonical prompt)
    规则：
      1) 前缀匹配：folder_prompt 以 allowed 开头
      2) 模糊匹配：difflib.SequenceMatcher ratio >= fuzzy_ratio
    """
    # 1) 前缀
    for a in allowed_prompts_norm:
        if folder_prompt_norm.startswith(a):
            return True, a
    # 2) 模糊：取最相近的一个看阈值
    best = None; best_score = 0.0
    for a in allowed_prompts_norm:
        score = difflib.SequenceMatcher(None, folder_prompt_norm, a).ratio()
        if score > best_score:
            best, best_score = a, score
    if best is not None and best_score >= fuzzy_ratio:
        return True, best
    return False, ""

def scan_ksets(gen_imgs_root: Path,
               allowed_prompts: List[str],
               target_guidances: List[float],
               steps: int,
               fuzzy_ratio: float,
               lower=True,
               debug=False) -> List[Dict]:
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

        if info["guidance"] not in gd_set:
            continue
        if info["steps"] != steps:
            continue

        imgs = list_images(d)
        if not imgs:
            unmatched.append((d.name, "no_images"))
            continue

        out.append({
            "dir": d,
            "prompt": canon_prompt,          # 用“匹配到的规范 prompt”做聚合键
            "seed": info["seed"],
            "guidance": info["guidance"],
            "steps": info["steps"],
            "images": imgs
        })

    if debug:
        print(f"[DEBUG] scan_ksets: matched={len(out)}, unmatched={len(unmatched)}")
        for name, why in unmatched[:20]:
            print(f"  - UNMATCHED: {name}  | {why}")
        if len(unmatched) > 20:
            print(f"  ... ({len(unmatched)-20} more)")
    return out

# -------- CLIP & kmeans & coverage --------
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

def coverage_for_kset(gen_feats: np.ndarray, centers: np.ndarray, taus: List[float]) -> np.ndarray:
    sims = gen_feats @ centers.T
    dists = 1.0 - sims
    nearest = np.argmin(dists, axis=1)
    min_d = dists[np.arange(len(gen_feats)), nearest]
    k = centers.shape[0]
    cov = []
    for t in taus:
        covered = np.zeros(k, dtype=bool)
        mask = min_d <= float(t)
        if np.any(mask):
            covered[np.unique(nearest[mask])] = True
        cov.append(covered.mean())
    return np.asarray(cov, dtype=np.float64)

def aggregate(by_g_p_seed: Dict[float, Dict[str, List[np.ndarray]]], taus: List[float]):
    out = {}
    taus = np.asarray(taus, dtype=np.float64)
    for g, mp in by_g_p_seed.items():
        per_prompt_cov = []
        for p, lst in mp.items():
            cov_stack = np.stack(lst, 0)  # [n_seed, len(taus)]
            cov_mu = cov_stack.mean(0)
            per_prompt_cov.append(cov_mu)
        cov_mu = np.stack(per_prompt_cov, 0).mean(0)
        cov_sd = np.stack(per_prompt_cov, 0).std(0)
        out[g] = {"taus": taus, "cov_mu": cov_mu, "cov_sd": cov_sd}
    return out

# -------- 主流程 --------
def run_for_concept(real_dir: Path, outputs_root: Path, concept: str,
                    methods: List[str], prompts: List[str],
                    guidances: List[float], steps: int,
                    k: int, taus: List[float],
                    batch_size: int, device: str,
                    fuzzy_ratio: float,
                    out_dir: Path, debug: bool):

    out_dir.mkdir(parents=True, exist_ok=True)
    feat = CLIPFeaturizer(device=device, batch_size=batch_size)

    real_imgs = list_images(real_dir)
    assert len(real_imgs) > 0, f"No real images in {real_dir}"
    real_feats = feat.encode_paths(real_imgs)
    centers = kmeans_centers(real_feats, k=k, seed=0)

    per_guidance_rows: Dict[float, List[dict]] = defaultdict(list)

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

        # g -> prompt -> [coverage_vec] over seeds
        by_g_p_seed: Dict[float, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

        for ks in tqdm(ksets, desc=f"[{concept}|{method}] K-sets"):
            gen_feats = feat.encode_paths(ks["images"])
            cov_vec = coverage_for_kset(gen_feats, centers, taus)
            by_g_p_seed[ks["guidance"]][ks["prompt"]].append(cov_vec)

        stats = aggregate(by_g_p_seed, taus)

        for g, st in stats.items():
            for t, mu, sd in zip(st["taus"], st["cov_mu"], st["cov_sd"]):
                per_guidance_rows[g].append({
                    "method": method,
                    "concept": concept,
                    "guidance": float(g),
                    "k": int(k),
                    "tau": float(t),
                    "coverage_mean": float(mu),
                    "coverage_std": float(sd),
                })

    # 保存每个 guidance 的 CSV
    for g, rows in per_guidance_rows.items():
        df = pd.DataFrame(rows).sort_values(by=["method", "tau"]).reset_index(drop=True)
        out_csv = out_dir / f"exp3_{g}_{concept}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}  (rows={len(df)})")

# -------- CLI --------
def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=Path, required=True, help="真实参考（单 concept）图像文件夹")
    ap.add_argument("--outputs_root", type=Path, required=True, help="生成根目录，含 {method}_{concept}/imgs/")
    ap.add_argument("--concept", type=str, required=True)
    ap.add_argument("--methods", type=str, required=True, help="逗号分隔的方法名")
    ap.add_argument("--prompts_json", type=Path, required=True, help="多类 JSON；只会取当前 concept 的 5 条")
    ap.add_argument("--guidances", type=str, required=True, help="逗号分隔 guidance，如 '5.0' 或 '3.0,5.0,7.5'")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--taus", type=str, default="0.3,0.45,0.6,0.75,0.9")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fuzzy_ratio", type=float, default=0.85, help="模糊匹配阈值，0~1，越大越严格")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--debug", type=int, default=0, help="1=打印未匹配样例")
    args = ap.parse_args()

    prompts = load_prompts_for_concept(args.prompts_json, args.concept, lower=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    guidances = parse_floats(args.guidances)
    taus = parse_floats(args.taus)

    run_for_concept(
        real_dir=args.real_dir,
        outputs_root=args.outputs_root,
        concept=args.concept,
        methods=methods,
        prompts=prompts,
        guidances=guidances,
        steps=args.steps,
        k=args.k,
        taus=taus,
        batch_size=args.batch_size,
        device=args.device,
        fuzzy_ratio=float(args.fuzzy_ratio),
        out_dir=args.out_dir,
        debug=bool(args.debug),
    )

if __name__ == "__main__":
    main()