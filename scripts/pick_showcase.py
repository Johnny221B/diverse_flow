#!/usr/bin/env python3
"""
From G=32 generated images, pick 8 for display:
  - For OUR method:   pick the 8 most DIVERSE images (greedy max-min CLIP distance)
  - For OTHER methods: pick the 8 most SIMILAR images (greedy min-max CLIP distance)

Usage:
    python scripts/pick_showcase.py \
        --ours   outputs/ourmethod_showcase \
        --others outputs/base_showcase outputs/cads_showcase outputs/dpp_showcase \
                 outputs/pg_showcase outputs/apg_showcase outputs/mix_showcase \
        --out    outputs/showcase_picked \
        --K 8
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_clip_model(device):
    """Load CLIP ViT-B/32 for feature extraction."""
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess
    except ImportError:
        # fallback: use openai clip from jit
        jit_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
        if os.path.exists(jit_path):
            model = torch.jit.load(jit_path, map_location=device).eval()
            preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
            return model, preprocess
        raise RuntimeError("Cannot load CLIP model. Install `pip install git+https://github.com/openai/CLIP.git` or ensure ~/.cache/clip/ViT-B-32.pt exists.")


@torch.no_grad()
def encode_images(image_paths, model, preprocess, device):
    """Encode a list of image paths into L2-normalized CLIP features."""
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(preprocess(img))
    batch = torch.stack(tensors).to(device)
    features = model.encode_image(batch)
    features = F.normalize(features.float(), dim=-1)
    return features


def greedy_max_diversity(sim_matrix, K):
    """Greedy selection: pick K indices that maximize pairwise distance.
    Start with the pair having lowest similarity, then iteratively add
    the image that has the largest minimum distance to the selected set."""
    N = sim_matrix.shape[0]
    dist = 1.0 - sim_matrix.cpu().numpy()

    # start with the pair having max distance
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [i, j]

    for _ in range(K - 2):
        best_idx, best_val = -1, -1.0
        for c in range(N):
            if c in selected:
                continue
            min_dist = min(dist[c, s] for s in selected)
            if min_dist > best_val:
                best_val = min_dist
                best_idx = c
        selected.append(best_idx)

    return selected


def greedy_max_similarity(sim_matrix, K):
    """Greedy selection: pick K indices that are most similar to each other.
    Start with the pair having highest similarity, then iteratively add
    the image with highest average similarity to the selected set."""
    N = sim_matrix.shape[0]
    sim = sim_matrix.cpu().numpy()
    np.fill_diagonal(sim, -1.0)  # exclude self

    i, j = np.unravel_index(np.argmax(sim), sim.shape)
    selected = [i, j]
    np.fill_diagonal(sim, 1.0)  # restore

    for _ in range(K - 2):
        best_idx, best_val = -1, -1.0
        for c in range(N):
            if c in selected:
                continue
            avg_sim = np.mean([sim[c, s] for s in selected])
            if avg_sim > best_val:
                best_val = avg_sim
                best_idx = c
        selected.append(best_idx)

    return selected


def process_method(method_dir, out_dir, model, preprocess, device, K, pick_diverse):
    """Process all prompts under a method directory."""
    imgs_dir = os.path.join(method_dir, "imgs")
    if not os.path.isdir(imgs_dir):
        print(f"  [SKIP] No imgs/ directory in {method_dir}")
        return

    method_name = os.path.basename(method_dir)
    method_out = os.path.join(out_dir, method_name)

    for prompt_slug in sorted(os.listdir(imgs_dir)):
        prompt_dir = os.path.join(imgs_dir, prompt_slug)
        if not os.path.isdir(prompt_dir):
            continue

        image_paths = sorted([
            os.path.join(prompt_dir, f)
            for f in os.listdir(prompt_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_paths) <= K:
            # nothing to pick, copy all
            pick_dir = os.path.join(method_out, prompt_slug)
            os.makedirs(pick_dir, exist_ok=True)
            for p in image_paths:
                shutil.copy2(p, os.path.join(pick_dir, os.path.basename(p)))
            continue

        # encode
        features = encode_images(image_paths, model, preprocess, device)
        sim = features @ features.T  # [N, N]

        # pick
        if pick_diverse:
            indices = greedy_max_diversity(sim, K)
            tag = "diverse"
        else:
            indices = greedy_max_similarity(sim, K)
            tag = "similar"

        # save picked images
        pick_dir = os.path.join(method_out, prompt_slug)
        os.makedirs(pick_dir, exist_ok=True)
        for rank, idx in enumerate(indices):
            src = image_paths[idx]
            ext = os.path.splitext(src)[1]
            dst = os.path.join(pick_dir, f"{rank:02d}_orig{idx:03d}{ext}")
            shutil.copy2(src, dst)

        print(f"  [{tag}] {prompt_slug}: picked {len(indices)} from {len(image_paths)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=str, required=True,
                    help='Path to our method output dir (e.g. outputs/ourmethod_showcase)')
    ap.add_argument('--others', type=str, nargs='+', required=True,
                    help='Paths to baseline output dirs')
    ap.add_argument('--out', type=str, default='outputs/showcase_picked',
                    help='Output directory for picked images')
    ap.add_argument('--K', type=int, default=8, help='Number of images to pick')
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Loading CLIP...")
    model, preprocess = load_clip_model(device)
    print("CLIP loaded.\n")

    # Our method: pick most diverse
    print(f"=== Our method (pick DIVERSE): {args.ours} ===")
    process_method(args.ours, args.out, model, preprocess, device, args.K, pick_diverse=True)

    # Baselines: pick most similar
    for other_dir in args.others:
        print(f"\n=== Baseline (pick SIMILAR): {other_dir} ===")
        process_method(other_dir, args.out, model, preprocess, device, args.K, pick_diverse=False)

    print(f"\nDone. Picked images saved to: {args.out}/")


if __name__ == "__main__":
    main()
