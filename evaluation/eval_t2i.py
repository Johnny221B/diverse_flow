import argparse, os, re, json, csv, time, gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# 尝试导入评估库
try:
    import piq
except ImportError:
    piq = None

from vendi_score import vendi

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# 适配 T2I-CompBench 的 slug 命名或带参数命名
META_RE = re.compile(r"^(?P<prompt>.+)_seed(?P<seed>-?\d+)_g(?P<guidance>[0-9.]+)_s(?P<steps>\d+)$")

def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_meta_from_name(name: str) -> Dict[str, Any]:
    m = META_RE.match(name)
    if not m:
        return {"prompt": name.replace("_", " "), "seed": None, "guidance": None, "steps": None}
    d = m.groupdict()
    d["prompt"]   = d["prompt"].replace("_", " ")
    return d

# --------- 彻底修复 Vendi Score 报错 ----------
def get_vendi_features(imgs: List[Image.Image], device: str) -> np.ndarray:
    """手动提取 InceptionV3 特征，避开 vendi_score 库内部的 torchvision 兼容性 Bug"""
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = nn.Identity() # 去掉分类层
    model.eval().to(device)
    
    preprocess = weights.transforms()
    
    feats = []
    with torch.no_grad():
        for img in imgs:
            # Inception 要求输入大小 299
            x = preprocess(img).unsqueeze(0).to(device)
            f = model(x)
            if isinstance(f, (list, tuple)): f = f[0]
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    
    return np.concatenate(feats, axis=0)

def compute_vendi_scores_safe(imgs: List[Image.Image], device: str = "cuda") -> Dict[str, float]:
    """计算 Vendi Score (无参考)"""
    try:
        # Pixel Vendi 比较稳，可以直接用
        from vendi_score import image_utils
        pix_vs = float(image_utils.pixel_vendi_score(imgs))
        
        # 提取特征并计算 Inception Vendi
        X = get_vendi_features(imgs, device)
        emb_vs = float(vendi.score_dual(X)) # 核心矩阵算法
        
        return {"vendi_pixel": pix_vs, "vendi_inception": emb_vs}
    except Exception as e:
        print(f"  [Vendi Error] {e}")
        return {"vendi_pixel": 1.0, "vendi_inception": 1.0}

def compute_msssim_diversity(paths: List[Path], device: torch.device) -> float:
    if len(paths) < 2 or piq is None: return 0.0
    rng = np.random.default_rng(42)
    max_pairs = 30 # T2I任务对子不用太多
    indices = rng.choice(len(paths), size=len(paths), replace=False)
    pairs = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            pairs.append((paths[indices[i]], paths[indices[j]]))
            if len(pairs) >= max_pairs: break
        if len(pairs) >= max_pairs: break

    tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])
    scores = []
    for p1, p2 in pairs:
        try:
            x1 = tfm(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
            x2 = tfm(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
            # 使用 piq 的 ms_ssim
            s = piq.multi_scale_ssim(x1, x2, data_range=1.0)
            scores.append(s.item())
        except: continue
    return float(1.0 - np.mean(scores)) if scores else 0.0

# --------- CLIP 对齐度 ----------
def compute_clip_alignment(folder: Path, prompt: str, model, preprocess, device):
    image_paths = list_images(folder)
    if not image_paths or model is None: return 0.0
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_input = tokenizer([prompt]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = []
            for p in image_paths:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                img_feat = model.encode_image(img)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sims.append((img_feat @ text_features.T).item())
        return float(np.mean(sims)) * 100.0
    except: return 0.0

# --------- 主评价函数 ----------
def evaluate_method_concept(outputs_root: Path, method: str, concept: str, args):
    # --- 修复路径逻辑：尝试多种命名可能 ---
    possible_dirs = [
        outputs_root / f"{method}_{concept}",
        outputs_root / f"{method}_t2i_{concept}", # 适配你现在的 base_t2i_complex 等
        outputs_root / f"baseline_{method}_{concept}"
    ]
    
    target_dir = None
    for d in possible_dirs:
        if (d / "imgs").exists():
            target_dir = d
            break
            
    if target_dir is None:
        print(f"[Skip] Could not find folder for Method: {method}, Concept: {concept}")
        return

    gen_root = target_dir / "imgs"
    eval_dir = target_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    csv_path = eval_dir / "metrics_per_prompt.csv"

    # 初始化 CLIP
    clip_model, _, clip_preprocess = None, None, None
    try:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.to(args.device).eval()
    except: pass

    fields = ["prompt_folder", "prompt_text", "vendi_inception", "vendi_pixel", "clip_score", "one_minus_ms_ssim"]
    prompt_dirs = sorted([d for d in gen_root.iterdir() if d.is_dir()])
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p_dir in tqdm(prompt_dirs, desc=f"Eval {target_dir.name}"):
            img_paths = list_images(p_dir)
            if not img_paths: continue

            meta = parse_meta_from_name(p_dir.name)
            imgs_pil = [Image.open(p).convert("RGB") for p in img_paths]
            
            # 计算指标
            v_scores = compute_vendi_scores_safe(imgs_pil, device=args.device)
            clip_align = compute_clip_alignment(p_dir, meta["prompt"], clip_model, clip_preprocess, args.device)
            msssim_div = compute_msssim_diversity(img_paths, torch.device(args.device))

            writer.writerow({
                "prompt_folder": p_dir.name,
                "prompt_text": meta["prompt"],
                "vendi_inception": round(v_scores["vendi_inception"], 4),
                "vendi_pixel": round(v_scores["vendi_pixel"], 4),
                "clip_score": round(clip_align, 4),
                "one_minus_ms_ssim": round(msssim_div, 4)
            })
            del imgs_pil; gc.collect(); torch.cuda.empty_cache()

    print(f">> Saved results to {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=str, default="./outputs")
    # 这里传 base, ourmethod, apg 等前缀即可，脚本会自动寻找带 _t2i_ 的目录
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--concepts", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    for m in args.methods:
        for c in args.concepts:
            evaluate_method_concept(root, m, c, args)

if __name__ == "__main__":
    main()