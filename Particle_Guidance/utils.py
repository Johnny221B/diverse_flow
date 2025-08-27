# ======================================
# FILE: Particle_Guidance/utils.py
# ======================================
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import random
from typing import List
import torch
from PIL import Image


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(text: str, maxlen: int = 120) -> str:
    """Make a filesystem-friendly slug from arbitrary text without regex."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    s = "_".join(text.strip().split())
    s = "".join(ch for ch in s if ch in allowed)
    # collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._-")
    if maxlen and len(s) > maxlen:
        s = s[:maxlen]
    return s or "untitled"


def save_images_grid(images: List[Image.Image], path: str, cols: int | None = None, padding: int = 8):
    if len(images) == 0:
        return
    w, h = images[0].size
    n = len(images)
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    grid = Image.new("RGB", (cols * w + (cols - 1) * padding, rows * h + (rows - 1) * padding), color=(255, 255, 255))
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(img, (x, y))
    grid.save(path)