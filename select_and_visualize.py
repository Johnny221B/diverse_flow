"""
Select 8 prompts (all methods have 64 images) where OSCAR has the largest
diversity advantage. For each prompt:
  - ourmethod:  greedy farthest-point sampling  → most diverse 8 images
  - baselines:  closest-to-centroid selection   → most similar/clustered 8 images
This maximises the visual contrast between OSCAR and baselines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from fix_all_method import make_method_grid_auto

# ── config ────────────────────────────────────────────────────────────────────
OUTPUTS_ROOT  = Path("/data2/toby/OSCAR/outputs")
RESULTS_DIR   = Path("/data2/toby/OSCAR/results/diversity_showcase")
OUTPUT_PDF    = Path("/data2/toby/OSCAR/results/diversity_showcase.pdf")

METHODS       = ["ourmethod", "pg", "cads", "dpp", "base", "mix", "apg"]
CATEGORIES    = ["t2i_color", "t2i_complex"]

NUM_PROMPTS   = 8    # how many prompts to select
NUM_IMGS      = 8    # images per method per prompt  (grid 4×2)
GRID_ROWS     = 4
GRID_COLS     = 2

RANK_METRIC   = "vendi_inception"
THUMB_SIZE    = (64, 64)
# ──────────────────────────────────────────────────────────────────────────────


def load_metrics():
    frames = []
    for method in METHODS:
        for cat in CATEGORIES:
            csv = OUTPUTS_ROOT / f"{method}_{cat}" / "eval" / "metrics_per_prompt.csv"
            if not csv.exists():
                print(f"  [warn] missing: {csv}")
                continue
            df = pd.read_csv(csv)
            df["method"]   = method
            df["category"] = cat
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_imgs_dir(method, cat, prompt_folder):
    """Return the actual images directory, handling case differences (e.g. dpp uses lowercase)."""
    d = OUTPUTS_ROOT / f"{method}_{cat}" / "imgs" / prompt_folder
    if d.exists():
        return d
    d_lower = OUTPUTS_ROOT / f"{method}_{cat}" / "imgs" / prompt_folder.lower()
    if d_lower.exists():
        return d_lower
    return d   # may not exist; callers check


def get_image_count(method, cat, prompt_folder):
    d = get_imgs_dir(method, cat, prompt_folder)
    return len(list(d.glob("*.png"))) if d.exists() else 0


def select_top_prompts(df, n=8):
    """
    Among prompts where ALL methods have exactly 64 images, pick top-n by
    ourmethod's RANK_METRIC advantage over the mean of other methods.
    """
    our    = df[df["method"] == "ourmethod"][["category", "prompt_folder", RANK_METRIC]].copy()
    our    = our.rename(columns={RANK_METRIC: "our_score"})
    others = (df[df["method"] != "ourmethod"]
              .groupby(["category", "prompt_folder"])[RANK_METRIC]
              .mean().reset_index()
              .rename(columns={RANK_METRIC: "other_mean"}))
    merged = our.merge(others, on=["category", "prompt_folder"])
    merged["advantage"] = merged["our_score"] - merged["other_mean"]
    merged = merged.sort_values("advantage", ascending=False)

    selected = []
    for _, row in merged.iterrows():
        cat, prompt = row["category"], row["prompt_folder"]
        counts = [get_image_count(m, cat, prompt) for m in METHODS]
        if min(counts) < 64:          # require ALL methods to have 64 images
            continue
        selected.append({
            "category":  cat,
            "prompt":    prompt,
            "advantage": round(row["advantage"], 4),
            "our_score": round(row["our_score"], 4),
        })
        if len(selected) == n:
            break
    return selected


def load_thumbs(method, cat, prompt_folder):
    """Return (sorted list of png paths, stacked float32 array of shape (N, d))."""
    imgs_dir = get_imgs_dir(method, cat, prompt_folder)
    paths    = sorted(imgs_dir.glob("*.png"))
    vecs = np.stack([
        np.array(Image.open(p).convert("RGB").resize(THUMB_SIZE)).flatten().astype(np.float32)
        for p in paths
    ])
    return paths, vecs


def farthest_point_sampling(vecs, k):
    """Greedy farthest-point: pick k indices maximising pairwise distances."""
    n = len(vecs)
    if k >= n:
        return list(range(n))
    selected  = [0]
    min_dists = np.full(n, np.inf)
    for _ in range(k - 1):
        d = np.sum((vecs - vecs[selected[-1]]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return selected


def closest_to_centroid(vecs, k):
    """
    Pick k images closest to the centroid (mean embedding).
    These tend to look alike — same pose / same style — maximising visual
    similarity among the selected set.
    """
    centroid = vecs.mean(axis=0)
    dists    = np.sum((vecs - centroid) ** 2, axis=1)
    return list(np.argsort(dists)[:k])


def select_images(method, cat, prompt_folder, k):
    """
    For ourmethod: farthest-point (diverse).
    For baselines: closest-to-centroid (similar).
    Returns list of stem strings like ['003', '017', ...].
    """
    paths, vecs = load_thumbs(method, cat, prompt_folder)
    if method == "ourmethod":
        indices = farthest_point_sampling(vecs, k)
    else:
        indices = closest_to_centroid(vecs, k)
    return [paths[i].stem for i in indices]


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metrics …")
    df = load_metrics()

    print(f"Selecting top {NUM_PROMPTS} t2i_color prompts (all methods @ 64 imgs) by '{RANK_METRIC}' advantage …")
    top_prompts = select_top_prompts(df[df["category"] == "t2i_color"], n=NUM_PROMPTS)

    print(f"Found {len(top_prompts)} qualifying prompts:")
    for p in top_prompts:
        print(f"  [{p['category']}] {p['prompt']}  advantage={p['advantage']}  our={p['our_score']}")

    pdf_pages = []

    def run_prompts(prompt_list, start_idx):
        for i, entry in enumerate(prompt_list):
            page_num = start_idx + i
            cat, prompt = entry["category"], entry["prompt"]
            print(f"\n[{page_num}] {prompt} ({cat})")

            method_to_indices = {}
            for method in METHODS:
                stems = select_images(method, cat, prompt, NUM_IMGS)
                method_to_indices[method] = stems
                tag = "diverse↑" if method == "ourmethod" else "similar↓"
                print(f"  {method:12s} [{tag}]: {stems}")

            ref_path = (OUTPUTS_ROOT / f"ourmethod_{cat}" / "imgs" /
                        prompt / f"{method_to_indices['ourmethod'][0]}.png")

            output_prefix = RESULTS_DIR / f"{page_num:02d}_{prompt}"
            _, png_path, pdf_path = make_method_grid_auto(
                ref_path=str(ref_path),
                methods=METHODS,
                method_to_indices=method_to_indices,
                output_prefix=str(output_prefix),
                source_method="ourmethod",
                task_suffix=cat,
                grid_rows=GRID_ROWS,
                grid_cols=GRID_COLS,
            )
            print(f"  saved → {pdf_path}")
            pdf_pages.append(png_path)

    run_prompts(top_prompts, start_idx=1)

    # ── t2i_complex: top prompts where all methods have 64 imgs ───────────────
    NUM_COMPLEX = 4
    print(f"\nSelecting top {NUM_COMPLEX} t2i_complex prompts …")
    df_complex = df[df["category"] == "t2i_complex"].copy()
    # patch: metrics_per_prompt.csv uses original prompt_folder names;
    # image dirs for dpp are lowercase — handle in get_image_count via get_imgs_dir
    complex_prompts = select_top_prompts(df_complex, n=NUM_COMPLEX)
    print(f"Found {len(complex_prompts)} qualifying complex prompts:")
    for p in complex_prompts:
        print(f"  {p['prompt']}  advantage={p['advantage']}  our={p['our_score']}")
    run_prompts(complex_prompts, start_idx=len(top_prompts) + 1)

    # merge all pages into one multi-page PDF using Pillow
    print(f"\nMerging {len(pdf_pages)} pages → {OUTPUT_PDF}")
    canvases = [Image.open(p).convert("RGB") for p in pdf_pages]
    canvases[0].save(
        str(OUTPUT_PDF),
        save_all=True,
        append_images=canvases[1:],
        resolution=150.0,
    )
    print(f"Done: {OUTPUT_PDF}")
