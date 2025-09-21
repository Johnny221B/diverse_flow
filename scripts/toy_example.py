# -*- coding: utf-8 -*-
# 6 子图（2 行 × 3 列）：一行 Baseline，一行 OSCAR-balanced；
# 三列分别展示 early / mid / final 三个时间点。
# - 同一批 z0~N(0,I)，同步数、同一模型、同一目标分布（3x3 GMM）
# - 连线使用“自适应透明度”（按起点局部密度；越拥挤越透明）
# - 隐藏坐标轴数字；红色“+”为 GMM 均值

import os, math, random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl

# mpl.rcParams['axes.titlesize'] = 18

# ---------------- Utils ----------------
def set_seed(s=0):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def project_partial_orth(grad: torch.Tensor, v: Optional[torch.Tensor], alpha: float):
    if v is None: return grad
    v_flat = v.view(v.size(0), -1)
    g_flat = grad.view(grad.size(0), -1)
    v2 = (v_flat*v_flat).sum(dim=1, keepdim=True) + 1e-12
    coef = ((g_flat*v_flat).sum(dim=1, keepdim=True))/v2
    parallel = coef * v_flat
    return (g_flat - alpha * parallel).view_as(grad)

def rotate90(v: torch.Tensor):
    """2D 旋转 +90°：(x,y)->(-y,x)"""
    return torch.stack([-v[...,1], v[...,0]], dim=-1)

# ---------------- Data: one GMM ----------------
class GMM2D:
    def __init__(self, means, sigma=0.22, weights=None):
        self.means = torch.as_tensor(means, dtype=torch.float32)
        self.sigma = float(sigma)
        K = self.means.shape[0]
        self.weights = torch.ones(K)/K if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.K = K
    @staticmethod
    def grid(grid=3, span=(-2,2), sigma=0.22):
        xs = np.linspace(span[0], span[1], grid)
        mus = np.array([(x,y) for y in xs for x in xs], dtype=np.float32)
        w = np.ones(mus.shape[0], dtype=np.float32); w = w / w.sum()
        return GMM2D(mus, sigma, w)
    def sample(self, n, device=None):
        idx = torch.multinomial(self.weights, n, replacement=True)
        mu = self.means[idx]
        x = mu + torch.randn(n,2)*self.sigma
        if device is not None: x = x.to(device); idx = idx.to(device)
        return x, idx

# ---------------- Model: Tiny CFM ----------------
class TinyCFM(nn.Module):
    def __init__(self, K: int, hidden: int = 128):
        super().__init__()
        self.K = K
        self.emb = nn.Linear(K, 32)
        self.tproj = nn.Linear(1, 32)
        self.net = nn.Sequential(
            nn.Linear(2+32+32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, xt, t, y_onehot):
        h = torch.cat([xt, self.emb(y_onehot), self.tproj(t)], dim=-1)
        return self.net(h)

def fm_loss(model, x0, y_onehot, p_drop=0.2):
    B = x0.shape[0]
    z = torch.randn_like(x0)                 # 训练里 z~N(0,I)
    t = torch.rand(B,1, device=x0.device)
    xt = (1 - t)*z + t*x0
    u  = x0 - z
    drop = (torch.rand(B,1, device=x0.device) < p_drop).float()
    y_used = (1 - drop)*y_onehot
    u_hat = model(xt, t, y_used)
    return F.mse_loss(u_hat, u)

# ---------------- Repulsion & density ----------------
def repulsive_grad_and_density(x: torch.Tensor, sigma_r: float = 0.28):
    X = x.unsqueeze(1); Y = x.unsqueeze(0)
    diff = X - Y
    dist2 = (diff*diff).sum(-1)
    w = torch.exp(-0.5*dist2/(sigma_r**2))
    w = w - torch.diag(torch.diag(w))
    g = (w.unsqueeze(-1)*diff).sum(dim=1) / (sigma_r**2)
    dens = w.sum(dim=1)
    return -g, dens

# ---------------- Inference: Baseline FM ----------------
@torch.no_grad()
def integrate_flowmatch(model, y_onehot, x0, steps: int):
    n = y_onehot.size(0); dev = next(model.parameters()).device
    x = x0.clone().to(dev)
    ts = torch.linspace(0,1,steps+1, device=dev)
    traj = [x.clone()]
    for i in range(steps):
        t0 = ts[i].expand(n,1)
        u  = model(x, t0, y_onehot)
        dt = float(ts[i+1]-ts[i])
        x  = x + dt*u
        traj.append(x.clone())
    return x, torch.stack(traj, dim=0)   # [steps+1, n, 2]

# ---------------- Inference: OSCAR-balanced ----------------
@torch.no_grad()
def integrate_oscar_balanced(
    model, y_onehot, x0, steps: int, means: torch.Tensor, gmm_sigma: float,
    gamma0: float = 0.22, t_gate: Tuple[float,float] = (0.18,0.85), sched_shape: str = 'sin2',
    partial_ortho: float = 1.0, repel_sigma: float = 0.26, repulsion_weight: float = 1.1,
    dens_alpha: float = 0.8, swirl_coef: float = 0.20,
    gamma_max_ratio: float = 0.35, r_cut_mult: float = 1.0, dist_tau_mult: float = 0.40,
    micro_every: int = 10, micro_step: float = 0.18, micro_iters: int = 1,
):
    dev = next(model.parameters()).device
    n = y_onehot.size(0); x = x0.clone().to(dev)
    ts = torch.linspace(0,1,steps+1, device=dev)
    traj = [x.clone()]

    def sched_factor(t_norm: float) -> float:
        t0, t1 = t_gate
        if not (t0 <= t_norm <= t1): return 0.0
        u = (t_norm - t0) / max(t1 - t0, 1e-8)
        return (math.sin(math.pi*min(max(u,0.0),1.0))**2) if sched_shape=='sin2' else (u*(1-u))

    def micro_polish(X: torch.Tensor, r_min: float, step: float, iters: int = 1):
        Y = X.clone()
        for _ in range(iters):
            D = torch.cdist(Y, Y); mask = (D > 0) & (D < r_min)
            push = torch.zeros_like(Y)
            for i in range(Y.size(0)):
                idx = mask[i].nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0: continue
                vecs = Y[i].unsqueeze(0) - Y[idx]
                norms = vecs.norm(dim=1, keepdim=True) + 1e-12
                push[i] = (vecs / norms).mean(dim=0)
            Y = Y + step * 0.5 * r_min * push
        return Y

    for i in range(steps):
        t0 = float(ts[i]); t1 = float(ts[i+1]); dt = t1 - t0
        tvec = torch.full((n,1), t0, device=dev)
        u = model(x, tvec, y_onehot)                 # 主干流
        base_disp = dt * u

        g_rep, dens = repulsive_grad_and_density(x, sigma_r=repel_sigma)
        dens_scale = ((dens/(dens.mean()+1e-12)).clamp_min(1e-3)).pow(dens_alpha).view(-1,1)
        dist = torch.cdist(x, means).min(dim=1).values
        r_cut = r_cut_mult * max(float(repel_sigma), float(gmm_sigma))
        gate_dist = torch.sigmoid((r_cut - dist) / (dist_tau_mult * r_cut + 1e-12)).view(-1,1)
        g_rep = repulsion_weight * g_rep * dens_scale * gate_dist

        g_perp  = project_partial_orth(g_rep, u, partial_ortho)
        g_swirl = rotate90(g_perp) * swirl_coef
        g_div   = project_partial_orth(g_perp + g_swirl, u, 1.0)

        gamma    = gamma0 * sched_factor(t0)
        div_disp = dt * gamma * g_div

        # 信赖域：||div|| ≤ ratio * ||base||
        bn_base = (base_disp.view(n,-1).norm(dim=1)+1e-12).view(-1,1)
        bn_div  = (div_disp.view(n,-1).norm(dim=1)+1e-12).view(-1,1)
        cap     = gamma_max_ratio * bn_base
        scale   = torch.minimum(torch.ones_like(cap), cap / bn_div)
        div_disp= scale * div_disp

        x = x + base_disp + div_disp

        if micro_every > 0 and (i+1) % micro_every == 0:
            x = micro_polish(x, r_min=0.6*repel_sigma, step=micro_step, iters=micro_iters)

        traj.append(x.clone())

    return x, torch.stack(traj, dim=0)   # [steps+1, n, 2]

# ---------------- Line alpha (adaptive) ----------------
def _knn_stat(x0: torch.Tensor, k: int = 8, approx_max_n: int = 1500):
    n = x0.size(0)
    if n <= approx_max_n:
        D = torch.cdist(x0, x0)
        D = D + torch.eye(n, device=x0.device)*1e9
        return D.topk(k, largest=False).values.mean(dim=1)
    else:
        M = min(512, n)
        idx = torch.randperm(n, device=x0.device)[:M]
        D = torch.cdist(x0, x0[idx])
        return D.topk(min(k, M), largest=False).values.mean(dim=1)

def compute_line_alpha(x0: torch.Tensor, xT: torch.Tensor,
                       mode: str = 'density', k: int = 8,
                       min_alpha: float = 0.03, max_alpha: float = 0.45):
    n = x0.size(0)
    if mode == 'density':
        knn = _knn_stat(x0, k=k)  # 大→稀疏，小→稠密
        s = (knn - knn.min()) / (knn.max() - knn.min() + 1e-12)
        a = min_alpha + (max_alpha - min_alpha) * s
    elif mode == 'length':
        L = (xT - x0).norm(dim=1)
        s = (L - L.min()) / (L.max() - L.min() + 1e-12)
        a = min_alpha + (max_alpha - min_alpha) * s
    else:  # global
        base = 0.4 * (200.0 / max(float(n), 200.0))**0.5
        a = torch.full((n,), float(np.clip(base, min_alpha, max_alpha)), device=x0.device)
    return a.clamp(min_alpha, max_alpha)

# ---------------- Plot helpers ----------------
def draw_means(ax, means: torch.Tensor):
    m = means.cpu().numpy()
    ax.scatter(m[:,0], m[:,1], marker='+', s=60, c='red', alpha=0.9)

def subplot_snapshot(ax, x0: torch.Tensor, x_now: torch.Tensor,
                     line_alpha_mode='density', max_lines: Optional[int]=None,
                     min_alpha=0.02, max_alpha=0.40):
    xi = x0.detach(); xT = x_now.detach()
    n  = xi.size(0)

    # 初始点（浅色）
    _xi = xi.cpu().numpy()
    ax.scatter(_xi[:,0], _xi[:,1], s=9, alpha=0.26, marker='o')

    # 自适应透明度的连线（init -> current）
    alphas = compute_line_alpha(xi, xT, mode=line_alpha_mode,
                                min_alpha=min_alpha, max_alpha=max_alpha).cpu().numpy()
    idxs = np.arange(n)
    if (max_lines is not None) and (n > max_lines):
        idxs = np.random.default_rng(0).choice(n, size=max_lines, replace=False)
    segs = np.stack([xi[idxs].cpu().numpy(), xT[idxs].cpu().numpy()], axis=1)
    cols = np.zeros((segs.shape[0],4), dtype=np.float32); cols[:,3] = alphas[idxs]
    lc = LineCollection(segs, colors=cols, linewidths=0.7, linestyles='--', antialiased=True)
    ax.add_collection(lc)

    # 当前点
    _xT = xT.cpu().numpy()
    ax.scatter(_xT[:,0], _xT[:,1], s=14, alpha=0.95, marker='x')

    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([])

# ---------------- Run (2 行 × 3 列：方法 × 时间) ----------------
set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 目标分布 & 模型训练
gmm   = GMM2D.grid(grid=3, sigma=0.22)
means = gmm.means.to(device)
model = TinyCFM(K=gmm.K, hidden=128).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
for it in range(1800):
    x0_b, idx = gmm.sample(512, device=device)
    y_b = F.one_hot(idx, num_classes=gmm.K).float()
    loss = fm_loss(model, x0_b, y_b, p_drop=0.2)
    opt.zero_grad(); loss.backward(); opt.step()

# 同一批初始点
K_SAMPLES = 1200
z0 = torch.randn(K_SAMPLES, 2, device=device)
ids = torch.multinomial(gmm.weights.to(device), K_SAMPLES, replacement=True)
y_onehot = F.one_hot(ids, num_classes=gmm.K).float()

# 推理（保留完整轨迹）
STEPS = 2000
x_base_final,  traj_base  = integrate_flowmatch(model, y_onehot, x0=z0, steps=STEPS)
x_osc_final,   traj_oscar = integrate_oscar_balanced(
    model, y_onehot, x0=z0, steps=STEPS, means=means, gmm_sigma=float(gmm.sigma),
    gamma0=0.22, t_gate=(0.18,0.85), sched_shape='sin2',
    partial_ortho=1.0, repel_sigma=0.26, repulsion_weight=1.1,
    dens_alpha=0.8, swirl_coef=0.20,
    gamma_max_ratio=0.35, r_cut_mult=1.0, dist_tau_mult=0.40,
    micro_every=10, micro_step=0.18, micro_iters=1
)

# 选择三个快照（early / mid / final）
snap_fracs   = (0.2, 0.6, 1.0)
snap_indices = [min(int(f*STEPS), STEPS) for f in snap_fracs]

# 2 行 × 3 列：第 1 行 Baseline；第 2 行 OSCAR
out_dir = "/mnt/data6t/yyz/flow_grpo/flow_base/outputs/results"
os.makedirs(out_dir, exist_ok=True)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

# 顶部列标题
col_titles = [f"step {idx}/{STEPS}" for f,idx in zip(snap_fracs, snap_indices)]
TITLE_FONTSIZE=20
for c in range(3):
    axes[0, c].set_title(f"Standard FM — {col_titles[c]}", fontsize=TITLE_FONTSIZE, pad=6)
    axes[1, c].set_title(f"Ourmethod — {col_titles[c]}", fontsize=TITLE_FONTSIZE, pad=6)

# 第一行：Baseline
for c, idx in enumerate(snap_indices):
    ax = axes[0, c]
    draw_means(ax, means); subplot_snapshot(ax, z0, traj_base[idx], line_alpha_mode='density', max_lines=None)

# 第二行：OSCAR
for c, idx in enumerate(snap_indices):
    ax = axes[1, c]
    draw_means(ax, means); subplot_snapshot(ax, z0, traj_oscar[idx], line_alpha_mode='density', max_lines=None)

plt.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(out_dir, "grid2x3.png")
pdf = os.path.join(out_dir, "grid2x3.pdf")
plt.savefig(png, dpi=300); plt.savefig(pdf); plt.close(fig)

print("Saved:", png, pdf)