import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
import argparse
import itertools
from skimage.metrics import structural_similarity as ssim
import imquality.brisque as brisque
import cv2
import piq

# ==============================================================================
# 0. 参数解析与设置
# ==============================================================================
parser = argparse.ArgumentParser(description="Calculate image quality metrics for given folders.")
parser.add_argument('folders', nargs='+', type=str, help='Path to one or more image folders.')
parser.add_argument('--clip-jit', type=str, default=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'),
                    help='Path to the local CLIP JIT model file.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for CLIP feature extraction.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cpu", "cuda", "cuda:0")') # <--- 修改1: 增加device参数

args = parser.parse_args()

# --- 设备设置 ---
if "cuda" in args.device and torch.cuda.is_available(): # <--- 修改2: 根据参数设置DEVICE
    DEVICE = args.device
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


# ==============================================================================
# 1. 加载本地 OpenAI CLIP JIT 模型 (使用你提供的代码)
# ==============================================================================
def load_openai_clip_from_jit(jit_path):
    print(f"Loading CLIP JIT model from: {jit_path}")
    if not os.path.exists(jit_path):
        raise FileNotFoundError(f"JIT path not found: {jit_path}. Please ensure you have the model file.")
        
    model = torch.jit.load(jit_path, map_location="cpu").eval()
    
    # OpenAI CLIP模型需要特定的图像预处理
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    return model, preprocess

clip_model, clip_preprocess = load_openai_clip_from_jit(args.clip_jit)
clip_model.to(DEVICE)
print(f"CLIP model and preprocessor loaded to {DEVICE}.")


# ==============================================================================
# 2. 指标计算函数 (这部分无需任何修改)
# ==============================================================================

def calculate_clip_score(image_folder, model, preprocess, device, batch_size):
    """计算文件夹中所有图片的平均CLIP Score。"""
    # ... (函数内部代码与之前完全相同)
    # 使用文件夹名作为文本提示 (prompt)
    prompt = Path(image_folder).name.replace('_', ' ').replace('-', ' ')
    print(f"\nCalculating CLIP Score for folder '{Path(image_folder).name}' with prompt: '{prompt}'")
    
    image_paths = list(Path(image_folder).rglob("*.jpg")) + list(Path(image_folder).rglob("*.png"))
    
    if not image_paths:
        print("No images found.")
        return 0.0

    all_scores = []
    model.eval()
    
    # 预先编码文本
    with torch.no_grad():
        text = open_clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Calculating CLIP Score"):
            batch_paths = image_paths[i:i + batch_size]
            images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            image_batch = torch.stack(images).to(device)
            
            image_features = model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度作为分数
            similarity = (100.0 * image_features @ text_features.T).cpu()
            all_scores.extend(similarity.numpy())
            
    return np.mean(all_scores)


def calculate_ms_ssim_diversity(image_folder, max_pairs=100):
    """计算文件夹内图片两两之间的平均 1 - MS-SSIM 来衡量多样性。(最终修复版)"""
    print(f"Calculating MS-SSIM diversity for folder '{Path(image_folder).name}'...")
    image_paths = list(Path(image_folder).rglob("*.jpg")) + list(Path(image_folder).rglob("*.png"))
   
    if len(image_paths) < 2:
        return 0.0

    if len(image_paths) > np.sqrt(max_pairs * 2) and max_pairs > 0:
        image_paths = np.random.choice(image_paths, size=int(np.sqrt(max_pairs * 2)) + 1, replace=False)
   
    all_ssim_scores = []
    pairs = list(itertools.combinations(image_paths, 2))
    if len(pairs) > max_pairs and max_pairs > 0:
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    for path1, path2 in tqdm(pairs, desc="Calculating MS-SSIM"):
        img1 = cv2.imread(str(path1))
        img2 = cv2.imread(str(path2))
       
        if img1 is None or img2 is None:
            print(f"\n[Warning] Skipping pair due to unreadable image: {path1} or {path2}")
            continue

        h1, w1, c1 = img1.shape
        # 确保图片大小一致
        img2 = cv2.resize(img2, (w1, h1))
       
        # --- START: 关键修改 ---
        # 确定一个安全的win_size
        min_dim = min(h1, w1)
        # win_size必须是奇数且小于等于最小边
        win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)

        if win_size < 3: # 如果win_size太小，无法计算
            print(f"\n[Warning] Skipping pair due to very small image size: {path1} or {path2}")
            continue

        # 使用 channel_axis=-1 来明确指出颜色通道是最后一个维度
        # 这取代了旧的 multichannel=True 参数
        score = ssim(img1, img2, data_range=255, channel_axis=-1, win_size=win_size)
        # --- END: 关键修改 ---

        all_ssim_scores.append(score)
       
    if not all_ssim_scores:
        return 0.0
       
    return 1 - np.mean(all_ssim_scores)


def calculate_brisque_quality(image_folder):
    """
    计算文件夹内所有图片的平均BRISQUE分数。
    (最终版: 使用piq库)
    """
    print(f"Calculating BRISQUE quality for folder '{Path(image_folder).name}'...")
   
    image_paths = list(Path(image_folder).rglob("*.jpg")) + list(Path(image_folder).rglob("*.png"))
   
    if not image_paths:
        return float('nan')

    # piq需要特定的图像预处理
    # 将PIL图片转换为 [0, 1] 范围的PyTorch Tensor
    brisque_transform = T.Compose([
        T.ToTensor() # 将图片转换为 [C, H, W] 格式，并归一化到 [0, 1]
    ])

    all_brisque_scores = []
    brisque_metric = piq.BRISQUELoss(data_range=1.0, reduction='none')

    for path in tqdm(image_paths, desc="Calculating BRISQUE"):
        # 使用PIL加载图片，确保是RGB格式
        img = Image.open(path).convert("RGB")
       
        # 预处理并增加一个batch维度
        img_tensor = brisque_transform(img).unsqueeze(0).to(DEVICE)
       
        # 使用piq计算分数
        with torch.no_grad():
            score = brisque_metric(img_tensor)
       
        all_brisque_scores.append(score.item())
       
    if not all_brisque_scores:
        return float('nan')
       
    return np.mean(all_brisque_scores)


# ==============================================================================
# 3. 主执行流程 (无需修改)
# ==============================================================================
def main():
    # ... (函数内部代码与之前完全相同)
    results = {}
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"Warning: Folder not found, skipping: {folder}")
            continue
        
        folder_name = Path(folder).name
        results[folder_name] = {}
        
        results[folder_name]['CLIP Score'] = calculate_clip_score(folder, clip_model, clip_preprocess, DEVICE, args.batch_size)
        results[folder_name]['1 - MS-SSIM (Diversity)'] = calculate_ms_ssim_diversity(folder)
        results[folder_name]['BRISQUE (Quality)'] = calculate_brisque_quality(folder)

    print("\n\n--- Final Results ---")
    header = f"{'Folder':<30} | {'CLIP Score (↑)':<20} | {'1 - MS-SSIM (↑)':<25} | {'BRISQUE (↓)':<20}"
    print(header)
    print("-" * len(header))
    
    for folder_name, metrics in results.items():
        cs = metrics.get('CLIP Score', float('nan'))
        msssim = metrics.get('1 - MS-SSIM (Diversity)', float('nan'))
        brisque = metrics.get('BRISQUE (Quality)', float('nan'))
        print(f"{folder_name:<30} | {cs:<20.4f} | {msssim:<25.4f}")
        print(f"{folder_name:<30} | {cs:<20.4f} | {msssim:<25.4f} | {brisque:<20.4f}")

if __name__ == '__main__':
    try:
        import open_clip
    except ImportError:
        print("Please install open_clip_torch: pip install open_clip_torch")
        exit()
    main()
    
# python calculate_metrics.py /mnt/data6t/yyz/flow_grpo/flow_base/outputs/dpp_a_photo_of_boxer /mnt/data6t/yyz/flow_grpo/flow_base/outputs/ourMethod_A_photo_of_boxer/imgs --device cuda:0