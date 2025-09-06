#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E3 Figure 1: Coverage–τ curves (k=10, τ in {0.3,0.45,0.6,0.75,0.9})

本版改动
-------
- CHANGED: 不再根据 prompts_json 过滤；直接处理 imgs 下所有 K-set 目录。
- CHANGED: 支持直接传入 {outputs_root}/{method}_{concept}/ 乃至 {method}_{concept}/imgs。
- CHANGED: 若未显式提供 --out_dir，则默认写入 {method_concept_root}/eval/。
- 兼容新旧两种 K-set 命名（见正则）。
"""

import argparse, re, gc
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

# -------- 解析目录名：支持两种格式 --------
# 新：{prompt}_seed{seed}_g{guidance}_s{steps}
NEW_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>\d+)_g(?P<guidance>[-+]?\d*\.?\d+)_s(?P<steps>\d+)$")
# 旧：{prompt}_{seed}_{guidance}_{steps}
OLD_RE = re.compile(r"^(?P<prompt>.+)_(?P<seed>\d+?)_(?P<guidance>[-+]?\d*\.?\d+?)_(?P<steps>\d+?)$")

def parse_kset_dirname(name: str):
    for pat in (NEW_RE, OLD_RE):
        m = pat.match(name)
        if m:
            return {
                "prompt": m.group("prompt"),
                "seed": int(m.group("seed")),
                "guidance": float(m.group("guidance")),
                "steps": int(m.group("steps")),
            }
    return None

# -------- CHANGED: 扫描 imgs 下所有 K-set，不做 prompt 过滤 --------
def scan_ksets_all(imgs_root: Path,
                   target_guidances: List[float],
                   steps: int,
                   debug: bool=False) -> List[Dict]:
    out, unmatched = [], []
    gd_set = set(float(g) for g in target_guidances) if target_guidances else None
    if not imgs_root.exists():
        return out
    for d in sorted(imgs_root.iterdir()):
        if not d.is_dir(): 
            continue
        info = parse_kset_dirname(d.name)
        if not info:
            unmatched.append((d.name, "regex_mismatch"))
            continue
        if gd_set is not None and info["guidance"] not in gd_set:
            continue
        if steps is not None and info["steps"] != steps:
            continue
        imgs = list_images(d)
        if not imgs:
            unmatched.append((d.name, "no_images"))
            continue
        out.append({
            "dir": d,
            "prompt": info["prompt"],
            "seed": info["seed"],
            "guidance": info["guidance"],
            "steps": info["steps"],
            "images": imgs
        })
    if debug:
        print(f"[DEBUG] scan_ksets_all: matched={len(out)}, unmatched={len(unmatched)}")
        for name, why in unmatched[:20]:
            print(f"  - UNMATCHED: {name} | {why}")
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
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)

def kmeans_centers(real_feats: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(real_feats)
    C = km.cluster_centers_
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C

def coverage_for_kset(gen_feats: np.ndarray, centers: np.ndarray, taus: List[float]) -> np.ndarray:
    if gen_feats.size == 0:
        return np.zeros(len(taus), dtype=np.float64)
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

# -------- 小工具：推断 imgs/ 与 out_dir --------
def resolve_imgs_root(method_concept_root: Path) -> Path:
    """优先找 {root}/imgs；若传入的就是 imgs 目录也可用。否则报错。"""
    if (method_concept_root / "imgs").is_dir():
        return method_concept_root / "imgs"
    if method_concept_root.name == "imgs" and method_concept_root.is_dir():
        return method_concept_root
    raise FileNotFoundError(f"Cannot locate imgs/ under {method_concept_root}")

def default_out_dir(method_concept_root: Path) -> Path:
    return method_concept_root / "eval"

def infer_method_name(method_concept_root: Path, concept: str) -> str:
    """从 {method}_{concept} 目录名推断 method；失败则用目录名回退。"""
    name = method_concept_root.name
    suffix = f"_{concept}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name  # 回退

# -------- 主流程（支持两种输入方式） --------
def run(
    real_dir: Path,
    # 方式 A：直接传入若干个 {method}_{concept} 根目录
    method_concept_roots: List[Path],
    # 方式 B：给 outputs_root + methods + concept（自动拼路径）
    outputs_root: Path,
    methods: List[str],
    concept: str,
    guidances: List[float],
    steps: int,
    k: int,
    taus: List[float],
    batch_size: int,
    device: str,
    out_dir: Path,    # 若为 None，则各自默认到 {method}_{concept}/eval
    debug: bool,
):
    feat = CLIPFeaturizer(device=device, batch_size=batch_size)

    real_imgs = list_images(real_dir)
    assert len(real_imgs) > 0, f"No real images in {real_dir}"
    real_feats = feat.encode_paths(real_imgs)
    centers = kmeans_centers(real_feats, k=k, seed=0)

    # 统一整理成 [(method_name, method_concept_root, imgs_root, out_dir_for_this_method)]
    tasks = []

    # A：显式给定若干 {method}_{concept} 根目录
    for mc_root in method_concept_roots or []:
        mc_root = mc_root.resolve()
        imgs_root = resolve_imgs_root(mc_root)
        method = infer_method_name(mc_root, concept)
        odir = out_dir if out_dir is not None else default_out_dir(mc_root)  # CHANGED
        odir.mkdir(parents=True, exist_ok=True)
        tasks.append((method, mc_root, imgs_root, odir))

    # B：给 outputs_root + methods + concept
    if outputs_root is not None and methods:
        for m in methods:
            mc_root = (outputs_root / f"{m}_{concept}").resolve()
            imgs_root = resolve_imgs_root(mc_root)
            odir = out_dir if out_dir is not None else default_out_dir(mc_root)  # CHANGED
            odir.mkdir(parents=True, exist_ok=True)
            tasks.append((m, mc_root, imgs_root, odir))

    if not tasks:
        raise ValueError("No method/concept roots to process. Provide --mc_roots or --outputs_root+--methods+--concept.")

    # 逐任务计算
    for method, mc_root, imgs_root, odir in tasks:
        print(f"\n[RUN] concept={concept} | method={method}\n"
              f"      real_dir   = {real_dir}\n"
              f"      imgs_root  = {imgs_root}\n"
              f"      out_dir    = {odir}\n")
        ksets = scan_ksets_all(imgs_root, guidances, steps, debug=debug)
        if not ksets:
            print(f"[WARN] No K-sets under {imgs_root} that match guidances/steps.")
            continue

        # g -> prompt -> [coverage_vec] over seeds
        by_g_p_seed: Dict[float, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

        for ks in tqdm(ksets, desc=f"[{concept}|{method}] K-sets"):
            gen_feats = feat.encode_paths(ks["images"])
            cov_vec = coverage_for_kset(gen_feats, centers, taus)
            by_g_p_seed[ks["guidance"]][ks["prompt"]].append(cov_vec)

        stats = aggregate(by_g_p_seed, taus)

        # 按 guidance 各写一个 CSV（放该 method 的 out_dir 下）
        for g, st in stats.items():
            rows = []
            for t, mu, sd in zip(st["taus"], st["cov_mu"], st["cov_sd"]):
                rows.append({
                    "method": method,
                    "concept": concept,
                    "guidance": float(g),
                    "k": int(k),
                    "tau": float(t),
                    "coverage_mean": float(mu),
                    "coverage_std": float(sd),
                })
            df = pd.DataFrame(rows).sort_values(by=["method", "tau"]).reset_index(drop=True)
            out_csv = odir / f"exp3_{g}_{concept}.csv"
            df.to_csv(out_csv, index=False)
            print(f"[SAVE] {out_csv}  (rows={len(df)})")

# -------- CLI --------
def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=Path, required=True, help="真实参考（单 concept）图像文件夹")

    ap.add_argument("--mc_roots", type=str, default="", 
                    help="逗号分隔的 {outputs_root}/{method}_{concept} 路径（或其 imgs 目录）")

    ap.add_argument("--outputs_root", type=Path, default=None, help="生成根目录，含 {method}_{concept}/")
    ap.add_argument("--concept", type=str, default=None)
    ap.add_argument("--methods", type=str, default="", help="逗号分隔的方法名")

    ap.add_argument("--guidances", type=str, required=True, help="逗号分隔 guidance，如 '5.0' 或 '3.0,5.0,7.5'")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--taus", type=str, default="0.05,0.20,0.35,0.50,0.65,0.80,0.95")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=Path, default=None, 
                    help="default is {method}_{concept}/eval/")
    ap.add_argument("--debug", type=int, default=0, help="1=打印未匹配样例")

    args = ap.parse_args()

    mc_roots = []
    if args.mc_roots.strip():
        mc_roots = [Path(x.strip()) for x in args.mc_roots.split(",") if x.strip()]
        if args.concept is None:
            name = mc_roots[0].name if mc_roots else ""
            if not args.concept:
                raise ValueError("Please provide --concept when using --mc_roots (用于推断 method 名与输出命名)。")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    guidances = parse_floats(args.guidances)
    taus = parse_floats(args.taus)

    run(
        real_dir=args.real_dir,
        method_concept_roots=mc_roots,
        outputs_root=args.outputs_root,
        methods=methods,
        concept=args.concept,
        guidances=guidances,
        steps=args.steps,
        k=args.k,
        taus=taus,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        debug=bool(args.debug),
    )

if __name__ == "__main__":
    main()
