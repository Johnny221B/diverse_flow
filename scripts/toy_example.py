#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D Flow Comparison (静态两幅图，带箭头与“往外推”的过程感)

你将得到两张并排子图（保存到 outputs/<method>_toy/imgs/）：
- 左：Flow Matching
- 右：Our Method（更分散）

可视化要点：
- 不画 GMM 网格；用一个大圆圈代表目标分布的边界（点都在圆内运动）。
- 同一批初始点（数量一致），两幅图完全可比。
- 用箭头显示从起点到终点的“外推”方向；并用稀疏的中间轨迹点增强“运动感”。
- 背景采用柔和的红色径向底图（而不是纯白），再叠加大圆圈边界。

运行示例：
python visualize_push_arrows.py --K 600 --steps 80 --radius 3.0 --n-arrows 120 --seed 0 --method toy2d
"""

import os, math, argparse, random, time
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------- utils ---------
def _log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
def set_seed(seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def _angle_wrap(a):
    """wrap angle to [-pi, pi]"""
    return (a + np.pi) % (2*np.pi) - np.pi

def _to_numpy(t):
    return t.detach().cpu().numpy()

# --------- toy dynamics (无需训练，直接构造流) ---------
@torch.no_grad()
def simulate_flow_matching(z0: torch.Tensor, steps: int, R: float,
                           k_rad: float = 4.0,
                           k_attract: float = 2.5,
                           theta_star: float = 0.0,
                           noise_std: float = 0.02):
    """
    Flow Matching（基线）：
    - 径向外推：把半径推向 R_target（略小于边界）
    - 角向轻微吸引到固定角度 theta_star 对应的环上目标点 p*（导致“比较聚集”）
    - 少量各向同性噪声
    返回：所有时间的采样（用于画中间轨迹点）
    """
    device = z0.device
    x = z0.clone()
    T = steps
    traj = [x.clone()]
    R_tgt = 0.82 * R
    p_star = torch.tensor([R_tgt * math.cos(theta_star), R_tgt * math.sin(theta_star)], device=device)

    for i in range(T):
        r = torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-6)
        e_r = x / r
        v_rad = k_rad * (R_tgt - r) * e_r                          # 径向弹簧
        v_att = k_attract * (p_star.unsqueeze(0) - x)              # 朝单一扇区的吸引
        v = v_rad + v_att
        x = x + (1.0 / T) * v + noise_std * torch.randn_like(x)
        # 保证在圆内（轻微反弹）
        over = torch.norm(x, dim=1, keepdim=True) > R
        if over.any():
            x_over = x[over.squeeze(1)]
            r_over = torch.norm(x_over, dim=1, keepdim=True)
            x[over.squeeze(1)] = x_over * (R / r_over) * 0.995
        traj.append(x.clone())

    return torch.stack(traj, dim=0)  # [T+1, N, 2]

@torch.no_grad()
def simulate_our_method(z0: torch.Tensor, steps: int, R: float,
                        k_rad: float = 4.0,
                        repulse_sigma: float = 0.6,
                        repulse_scale: float = 2.0,
                        noise_std: float = 0.02):
    """
    Our Method（更分散）：
    - 同样的径向外推到 R_target
    - 叠加 pairwise 斥力（让点沿环向彼此分散）
    - 少量各向同性噪声
    """
    device = z0.device
    x = z0.clone()
    T = steps
    traj = [x.clone()]
    R_tgt = 0.82 * R

    for i in range(T):
        r = torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-6)
        e_r = x / r
        v_rad = k_rad * (R_tgt - r) * e_r

        # 斥力（高斯核）：对拥挤更敏感，促使沿环向均匀分布
        X = x.unsqueeze(1)                        # [N,1,2]
        Y = x.unsqueeze(0)                        # [1,N,2]
        diff = X - Y                              # [N,N,2]
        dist2 = (diff * diff).sum(dim=-1)         # [N,N]
        W = torch.exp(-0.5 * dist2 / (repulse_sigma ** 2))
        W = W - torch.diag(torch.diag(W))         # 去掉 self
        g_rep = -(W.unsqueeze(-1) * diff).sum(dim=1) / (repulse_sigma ** 2)  # -∇ϕ
        v_rep = repulse_scale * g_rep / (x.size(0) ** 0.5)  # 缩放一下以免太强

        v = v_rad + v_rep
        x = x + (1.0 / T) * v + noise_std * torch.randn_like(x)

        # 圆边界内
        over = torch.norm(x, dim=1, keepdim=True) > R
        if over.any():
            x_over = x[over.squeeze(1)]
            r_over = torch.norm(x_over, dim=1, keepdim=True)
            x[over.squeeze(1)] = x_over * (R / r_over) * 0.995

        traj.append(x.clone())

    return torch.stack(traj, dim=0)  # [T+1, N, 2]

# --------- 背景与绘图 ---------
def draw_radial_background(ax, R: float, res: int = 240):
    """柔和红色径向底图 + 圆圈轮廓"""
    xs = np.linspace(-R, R, res)
    ys = np.linspace(-R, R, res)
    X, Y = np.meshgrid(xs, ys)
    Rg = np.sqrt(X**2 + Y**2)
    # 径向背景：中心亮，边缘淡（更直观“往外推”）
    Z = np.exp(-0.5 * (Rg / (0.75 * R))**2)
    ax.imshow(Z, extent=[-R, R, -R, R], origin='lower', cmap='Reds', alpha=0.35, interpolation='bilinear')
    # 圆圈边界
    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(R*np.cos(theta), R*np.sin(theta), color='red', linewidth=2.0, alpha=0.9)

def format_axes(ax, title: str, R: float):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_xticks([]); ax.set_yticks([])  # 去掉网格与刻度
    ax.set_title(title)

def plot_process_panel(ax, traj: torch.Tensor, R: float,
                       arrow_idx: np.ndarray,
                       trail_keyframes: int = 4,
                       point_size: float = 6.0,
                       arrow_width: float = 0.004):
    """
    画：背景 + 圆圈 + 全体终点 + 稀疏中间轨迹点 + 箭头（起点->终点）
    """
    draw_radial_background(ax, R)
    T, N, _ = traj.shape
    # 全体终点（半透明，突出总体散布）
    final = traj[-1]
    ax.scatter(_to_numpy(final[:,0]), _to_numpy(final[:,1]),
               s=point_size, alpha=0.85, color='tab:blue', linewidths=0)

    # 稀疏关键中间帧（增强“向外运动”的感觉）
    ks = np.linspace(0, T-1, trail_keyframes, dtype=int)
    for j, k in enumerate(ks[:-1]):  # 不含最终帧
        P = traj[k]
        alpha = 0.25 * (j + 1) / (len(ks)) + 0.05
        ax.scatter(_to_numpy(P[:,0]), _to_numpy(P[:,1]),
                   s=point_size*0.7, alpha=alpha, color='tab:gray', linewidths=0)

    # 箭头（子集，避免过度拥挤）
    start = traj[0][arrow_idx]
    end   = traj[-1][arrow_idx]
    U = end - start
    ax.quiver(_to_numpy(start[:,0]), _to_numpy(start[:,1]),
              _to_numpy(U[:,0]), _to_numpy(U[:,1]),
              angles='xy', scale_units='xy', scale=1.0,
              width=arrow_width, headwidth=6, headlength=7,
              color='black', alpha=0.9, zorder=5)

# --------- 主程序 ---------
def parse_args():
    ap = argparse.ArgumentParser(description="2D Flow Comparison (arrows + big circle)")
    ap.add_argument('--K', type=int, default=600, help='点的数量（两幅图一致）')
    ap.add_argument('--steps', type=int, default=80, help='积分步数（越大轨迹越平滑）')
    ap.add_argument('--radius', type=float, default=3.0, help='大圆半径 R')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-arrows', type=int, default=120, help='每幅图绘制箭头的点数（从同一批点中子采样）')
    ap.add_argument('--trail-keyframes', type=int, default=4, help='中间轨迹的关键帧数（不含最终帧）')
    ap.add_argument('--method', type=str, default='toy2d')
    ap.add_argument('--out', type=str, default=None,
                    help='输出根目录；若不填，则保存到 outputs/<method>_toy/imgs/')
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 输出路径（固定到 outputs/...）
    outputs_root = os.path.join(os.getcwd(), 'outputs')
    base_dir = args.out.strip() if args.out else os.path.join(outputs_root, f'{args.method}_toy')
    img_dir = os.path.join(base_dir, 'imgs'); os.makedirs(img_dir, exist_ok=True)
    _log(f'Output dir: {img_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始点：集中在中心的小圆内，便于体现“向外推”
    R = float(args.radius)
    K = int(args.K)
    r0 = 0.25 * R
    # 在小圆内均匀采样
    u = torch.rand(K, 1)
    theta = 2 * math.pi * torch.rand(K, 1)
    rad = r0 * torch.sqrt(u)
    z0 = torch.cat([rad * torch.cos(theta), rad * torch.sin(theta)], dim=1).to(device)

    # 模拟两种过程（同一批起点）
    traj_fm  = simulate_flow_matching(z0, steps=args.steps, R=R,
                                      k_rad=4.0, k_attract=2.2,
                                      theta_star=0.0, noise_std=0.02)
    traj_ours = simulate_our_method(z0, steps=args.steps, R=R,
                                    k_rad=4.0, repulse_sigma=0.6,
                                    repulse_scale=2.0, noise_std=0.02)

    # 箭头子样本索引（两幅图共用）
    n_ar = min(args.n_arrows, K)
    arrow_idx = np.random.choice(K, size=n_ar, replace=False)

    # 画图
    fig = plt.figure(figsize=(12, 5))
    axL = plt.subplot(1,2,1)
    plot_process_panel(axL, traj_fm,  R, arrow_idx,
                       trail_keyframes=args.trail_keyframes,
                       point_size=6.0, arrow_width=0.004)
    axL.set_title("Flow Matching")
    axL.set_aspect('equal', adjustable='box')
    axL.set_xlim(-R, R); axL.set_ylim(-R, R); axL.set_xticks([]); axL.set_yticks([])

    axR = plt.subplot(1,2,2)
    plot_process_panel(axR, traj_ours, R, arrow_idx,
                       trail_keyframes=args.trail_keyframes,
                       point_size=6.0, arrow_width=0.004)
    axR.set_title("Our Method")
    axR.set_aspect('equal', adjustable='box')
    axR.set_xlim(-R, R); axR.set_ylim(-R, R); axR.set_xticks([]); axR.set_yticks([])

    plt.tight_layout()
    png_path = os.path.join(img_dir, "comparison_with_arrows.png")
    pdf_path = os.path.join(img_dir, "comparison_with_arrows.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)  # 矢量 PDF
    plt.close(fig)
    _log(f"Saved: {png_path}")
    _log(f"Saved: {pdf_path}")

if __name__ == "__main__":
    main()