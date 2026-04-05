"""
eval_4img.py — Re-evaluate all methods by randomly sampling N images per prompt (default 4).
Saves results to {method_dir}/eval_{N}img/metrics_per_prompt.csv
Then produces a ranking of prompts where ourmethod has the largest advantage.

Usage:
    python eval_4img.py --n-imgs 4 --device cuda:0 \
        --methods ourmethod base cads dpp pg apg mix \
        --concepts t2i_color t2i_complex t2i_spatial
"""

import argparse, gc, csv, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

try:
    import piq
except ImportError:
    piq = None

from vendi_score import vendi

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ─────────────── InceptionV3 cache ────────────────────────────────────────────
_INC_CACHE: Dict[str, Tuple[nn.Module, any]] = {}

def _get_inception(device: str):
    if device not in _INC_CACHE:
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = nn.Identity()
        model.eval().to(device)
        _INC_CACHE[device] = (model, weights.transforms())
    return _INC_CACHE[device]

def get_vendi_features(imgs: List[Image.Image], device: str) -> np.ndarray:
    model, preprocess = _get_inception(device)
    feats = []
    with torch.no_grad():
        for img in imgs:
            x = preprocess(img).unsqueeze(0).to(device)
            f = model(x)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def compute_vendi_scores(imgs: List[Image.Image], device: str) -> Dict[str, float]:
    try:
        from vendi_score import image_utils
        pix_vs = float(image_utils.pixel_vendi_score(imgs))
        X = get_vendi_features(imgs, device)
        emb_vs = float(vendi.score_dual(X))
        return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}
    except Exception as e:
        print(f"  [Vendi Error] {e}")
        return {"vendi_pixel": 1.0, "vendi_inception": 1.0}

def compute_msssim_diversity(paths: List[Path], device: torch.device) -> float:
    if len(paths) < 2 or piq is None:
        return 0.0
    rng = np.random.default_rng(42)
    max_pairs = min(30, len(paths) * (len(paths) - 1) // 2)
    indices = list(range(len(paths)))
    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.append((paths[indices[i]], paths[indices[j]]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    tfm = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    scores = []
    for p1, p2 in pairs:
        try:
            x1 = tfm(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
            x2 = tfm(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
            s = piq.multi_scale_ssim(x1, x2, data_range=1.0)
            scores.append(s.item())
        except:
            continue
    return float(1.0 - np.mean(scores)) if scores else 0.0

def compute_clip_alignment(img_paths: List[Path], prompt: str, clip_model, clip_preprocess, device: str) -> float:
    if not img_paths or clip_model is None:
        return 0.0
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_input = tokenizer([prompt]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = []
            for p in img_paths:
                img = clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                img_feat = clip_model.encode_image(img)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims.append((img_feat @ text_features.T).item())
        return float(np.mean(sims)) * 100.0
    except:
        return 0.0

def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def _find_method_dir(outputs_root: Path, method: str, concept: str) -> Optional[Path]:
    for candidate in [
        outputs_root / f"{method}_{concept}",
        outputs_root / f"{method}_t2i_{concept}",
    ]:
        if (candidate / "imgs").exists():
            return candidate
    return None

# ─────────────── per-method evaluation ────────────────────────────────────────
def evaluate_method_concept(outputs_root: Path, method: str, concept: str,
                            n_imgs: int, seed: int, args) -> Optional[Path]:
    target_dir = _find_method_dir(outputs_root, method, concept)
    if target_dir is None:
        print(f"[Skip] {method}/{concept} not found")
        return None

    gen_root = target_dir / "imgs"
    eval_dir = target_dir / f"eval_{n_imgs}img"
    eval_dir.mkdir(exist_ok=True)
    csv_path = eval_dir / "metrics_per_prompt.csv"

    if csv_path.exists():
        print(f"[Skip] Already done: {csv_path}")
        return csv_path

    # Load CLIP once
    clip_model, clip_preprocess = None, None
    try:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(args.device).eval()
    except Exception:
        pass

    fields = ["prompt_folder", "prompt_text", "vendi_inception", "vendi_pixel",
              "clip_score", "one_minus_ms_ssim"]

    prompt_dirs = sorted([d for d in gen_root.iterdir() if d.is_dir()])
    rng = np.random.default_rng(seed)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p_dir in tqdm(prompt_dirs, desc=f"{method}/{concept}"):
            all_imgs = list_images(p_dir)
            if not all_imgs:
                continue

            # Sample n_imgs randomly (or all if fewer)
            n = min(n_imgs, len(all_imgs))
            idxs = rng.choice(len(all_imgs), size=n, replace=False)
            idxs.sort()
            sampled = [all_imgs[i] for i in idxs]

            prompt_text = p_dir.name.replace("_", " ")
            imgs_pil = [Image.open(p).convert("RGB") for p in sampled]

            v_scores = compute_vendi_scores(imgs_pil, args.device)
            clip_score = compute_clip_alignment(sampled, prompt_text, clip_model, clip_preprocess, args.device)
            msssim_div = compute_msssim_diversity(sampled, torch.device(args.device))

            writer.writerow({
                "prompt_folder":     p_dir.name,
                "prompt_text":       prompt_text,
                "vendi_inception":   round(v_scores["vendi_inception"], 4),
                "vendi_pixel":       round(v_scores["vendi_pixel"], 4),
                "clip_score":        round(clip_score, 4),
                "one_minus_ms_ssim": round(msssim_div, 4),
            })
            del imgs_pil
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Done] {csv_path}")
    return csv_path


# ─────────────── analysis: find best prompts for ourmethod ───────────────────
def analyze_best_prompts(outputs_root: Path, methods: List[str], concepts: List[str],
                         n_imgs: int, our_method: str = "ourmethod",
                         top_k: int = 20) -> pd.DataFrame:
    """
    For each prompt (across all concepts), compute ourmethod's average advantage
    over all baselines across all diversity metrics (vendi_inception, vendi_pixel,
    one_minus_ms_ssim). Rank prompts by this advantage.
    """
    baselines = [m for m in methods if m != our_method]
    metrics = ["vendi_inception", "vendi_pixel", "one_minus_ms_ssim"]

    all_rows = []
    for concept in concepts:
        # Load ourmethod data
        our_dir = _find_method_dir(outputs_root, our_method, concept)
        if our_dir is None:
            continue
        our_csv = our_dir / f"eval_{n_imgs}img" / "metrics_per_prompt.csv"
        if not our_csv.exists():
            print(f"[Warn] Missing: {our_csv}")
            continue
        our_df = pd.read_csv(our_csv)
        our_df["prompt_key"] = our_df["prompt_folder"].str.lower()

        # Load baseline data
        baseline_dfs = {}
        for bl in baselines:
            bl_dir = _find_method_dir(outputs_root, bl, concept)
            if bl_dir is None:
                continue
            bl_csv = bl_dir / f"eval_{n_imgs}img" / "metrics_per_prompt.csv"
            if not bl_csv.exists():
                continue
            df = pd.read_csv(bl_csv)
            df["prompt_key"] = df["prompt_folder"].str.lower()
            baseline_dfs[bl] = df.set_index("prompt_key")

        if not baseline_dfs:
            continue

        for _, row in our_df.iterrows():
            pk = row["prompt_key"]
            our_vals = {m: row[m] for m in metrics}

            # Compute advantage vs each baseline
            gaps = []
            per_bl = {}
            for bl, bl_df in baseline_dfs.items():
                if pk not in bl_df.index:
                    continue
                bl_row = bl_df.loc[pk]
                for m in metrics:
                    gap = our_vals[m] - bl_row[m]
                    gaps.append(gap)
                    per_bl[f"gap_{bl}_{m}"] = round(gap, 4)

            if not gaps:
                continue

            avg_gap = float(np.mean(gaps))
            result_row = {
                "concept":          concept,
                "prompt_folder":    row["prompt_folder"],
                "prompt_text":      row["prompt_text"],
                "avg_gap_vs_all":   round(avg_gap, 4),
                "our_vendi_inception":   round(our_vals["vendi_inception"], 4),
                "our_vendi_pixel":       round(our_vals["vendi_pixel"], 4),
                "our_one_minus_ms_ssim": round(our_vals["one_minus_ms_ssim"], 4),
                "our_clip_score":        round(row["clip_score"], 4),
            }
            result_row.update(per_bl)
            all_rows.append(result_row)

    if not all_rows:
        print("[Warn] No data to analyze.")
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows).sort_values("avg_gap_vs_all", ascending=False)
    return df_all


# ─────────────── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="./outputs")
    parser.add_argument("--methods", nargs="+",
                        default=["ourmethod", "base", "cads", "dpp", "pg", "apg", "mix"])
    parser.add_argument("--concepts", nargs="+",
                        default=["t2i_color", "t2i_complex", "t2i_spatial"])
    parser.add_argument("--n-imgs",  type=int, default=4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--top-k",   type=int, default=20,
                        help="Number of top prompts to show in ranking")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-evaluation, only do analysis")
    args = parser.parse_args()

    root = Path(args.outputs_root)

    # Step 1: evaluate
    if not args.skip_eval:
        for method in args.methods:
            for concept in args.concepts:
                evaluate_method_concept(root, method, concept,
                                        args.n_imgs, args.seed, args)

    # Step 2: analyze
    print("\n" + "=" * 60)
    print(f"Analyzing best prompts for ourmethod (n_imgs={args.n_imgs})...")
    df = analyze_best_prompts(root, args.methods, args.concepts,
                              args.n_imgs, top_k=args.top_k)
    if df.empty:
        return

    out_csv = root / f"best_prompts_4img_top{args.top_k}.csv"
    df.head(args.top_k).to_csv(out_csv, index=False)
    print(f"\nTop {args.top_k} prompts saved → {out_csv}")

    # Also save full ranking
    full_csv = root / f"best_prompts_4img_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"Full ranking saved → {full_csv}")

    # Print summary
    print(f"\n{'Rank':<5} {'Concept':<12} {'avg_gap':>8}  Prompt")
    print("-" * 80)
    for i, row in enumerate(df.head(args.top_k).itertuples(), 1):
        print(f"{i:<5} {row.concept:<12} {row.avg_gap_vs_all:>8.4f}  {row.prompt_text[:55]}")


if __name__ == "__main__":
    main()
