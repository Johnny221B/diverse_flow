# -*- coding: utf-8 -*-
# 2×3 grid snapshots: (row) Baseline FM vs OSCAR  (col) early/mid/final
# Geometry: circle-GMM (K=24) on radius R; init points uniform in inner disk (r_init<R)
# Fairness: same z0, same NFE, same trained model; only sampler differs.
# Outputs: grid png/pdf with titles & labels; means (black '+'), init (light blue o), final (orange x), dashed links.

import os, math, random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------- Utils ----------------
def set_seed(s=0):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def rotate90(v: torch.Tensor):
    return torch.stack([-v[...,1], v[...,0]], dim=-1)

# ---------------- Data: circle GMM ----------------
class GMM2D:
    def __init__(self, means, sigma=0.14, weights=None):
        self.means = torch.as_tensor(means, dtype=torch.float32)
        self.sigma = float(sigma)
        K = self.means.shape[0]
        self.weights = torch.ones(K)/K if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        self.K = K

    @staticmethod
    def circle(K: int = 24, radius: float = 2.2, sigma: float = 0.14, phase: float = 0.0):
        theta = np.linspace(phase, 2*np.pi+phase, K, endpoint=False)
        means = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1).astype(np.float32)
        return GMM2D(means, sigma=sigma)

    def sample(self, n, device=None):
        idx = torch.multinomial(self.weights, n, replacement=True)
        mu = self.means[idx]
        x = mu + torch.randn(n, 2) * self.sigma
        if device is not None: x = x.to(device); idx = idx.to(device)
        return x, idx

def sample_in_disk(n: int, radius: float, device=None, shift=(0.0, 0.0)):
    """Uniform in a disk: r = R*sqrt(U), theta ~ U[0,2pi). Optional small shift to make task harder."""
    u = torch.rand(n)
    r = radius * torch.sqrt(u)
    theta = 2 * math.pi * torch.rand(n)
    x = torch.stack([r*theta.cos(), r*theta.sin()], dim=1)
    x += torch.tensor(shift, dtype=x.dtype, device=x.device)
    return x.to(device) if device is not None else x

# ---------------- Model: Unconditional Tiny CFM ----------------
class TinyCFMUncond(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.tproj = nn.Linear(1, 32)
        self.net = nn.Sequential(
            nn.Linear(2 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, xt, t):
        h = torch.cat([xt, self.tproj(t)], dim=-1)
        return self.net(h)

def fm_loss_uncond(model, x0):
    B = x0.shape[0]
    z = torch.randn_like(x0)
    t = torch.rand(B, 1, device=x0.device)
    xt = (1 - t) * z + t * x0
    u  = x0 - z
    u_hat = model(xt, t)
    return F.mse_loss(u_hat, u)

# ---------------- Repulsion / density ----------------
def repulsive_grad_and_density(x: torch.Tensor, sigma_r: float = 0.28):
    X = x.unsqueeze(1); Y = x.unsqueeze(0)
    diff = X - Y
    dist2 = (diff*diff).sum(-1)
    w = torch.exp(-0.5 * dist2 / (sigma_r ** 2))
    w = w - torch.diag(torch.diag(w))
    g = (w.unsqueeze(-1) * diff).sum(dim=1) / (sigma_r ** 2)
    dens = w.sum(dim=1)
    return -g, dens

# ---------------- Baseline FM (unconditional, with trajectory) ----------------
@torch.no_grad()
def integrate_flowmatch_uncond_traj(model, x0, steps: int):
    n = x0.size(0); dev = next(model.parameters()).device
    x = x0.clone().to(dev)
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    traj = [x.clone()]
    for i in range(steps):
        t0 = ts[i].expand(n, 1)
        u  = model(x, t0)
        dt = float(ts[i+1] - ts[i])
        x  = x + dt * u
        traj.append(x.clone())
    return x, torch.stack(traj, dim=0)   # [steps+1, n, 2]

# ---------------- OSCAR (unconditional, orthogonal explore, with trajectory) ----------------
@torch.no_grad()
def integrate_oscar_uncond_traj(
    model, x0, steps: int, means: torch.Tensor, gmm_sigma: float, R_target: float,
    gamma0: float = 1.0, t_gate: Tuple[float, float] = (0.05, 0.90), sched_shape: str = 'sin2',
    gamma_max_ratio: float = 1.40,
    partial_ortho: float = 1.0, repel_sigma: float = 0.28, repulsion_weight: float = 1.6,
    dens_alpha: float = 0.8, dist_tau_mult: float = 0.40, r_cut_mult: float = 1.0,
    sigma_theta: float = 0.35, beta_ang: float = 1.6, radial_anchor: float = 0.25,
    swirl_coef: float = 0.55,
    noise_scale_max: float = 0.55, noise_t_end: float = 0.55,
):
    dev = next(model.parameters()).device
    n = x0.size(0); x = x0.clone().to(dev)
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    traj = [x.clone()]

    def sched_factor(tn):
        t0, t1 = t_gate
        if not (t0 <= tn <= t1): return 0.0
        u = (tn - t0) / max(t1 - t0, 1e-8)
        return (math.sin(math.pi * min(max(u, 0.0), 1.0)) ** 2) if sched_shape == 'sin2' else (u * (1 - u))

    for i in range(steps):
        t0 = float(ts[i]); dt = float(ts[i+1] - ts[i])
        tvec = torch.full((n, 1), t0, device=dev)

        # main FM velocity
        u = model(x, tvec)                # [n,2]
        base_disp = dt * u

        # Euclidean repulsion (gated by density & distance to means)
        g_rep, dens = repulsive_grad_and_density(x, sigma_r=repel_sigma)
        dens_scale  = ((dens / (dens.mean() + 1e-12)).clamp_min(1e-3)).pow(dens_alpha).view(-1, 1)
        dist_mean = torch.cdist(x, means).min(dim=1).values
        r_cut = r_cut_mult * max(float(repel_sigma), float(gmm_sigma))
        gate_dist = torch.sigmoid((r_cut - dist_mean) / (dist_tau_mult * r_cut + 1e-12)).view(-1, 1)
        g_rep = repulsion_weight * g_rep * dens_scale * gate_dist

        # angular repulsion (equalize angles)
        eps = 1e-9
        r = torch.sqrt((x*x).sum(dim=1) + eps)             # [n]
        theta = torch.atan2(x[:,1], x[:,0])                # [n]
        dtheta = theta.unsqueeze(1) - theta.unsqueeze(0)
        dtheta = (dtheta + math.pi) % (2*math.pi) - math.pi
        Wth = torch.exp(-0.5 * (dtheta / sigma_theta) ** 2)
        torque = (torch.sin(dtheta) * Wth)
        torque = torque - torch.diag(torch.diag(torque))
        tau = torque.sum(dim=1).view(-1,1)
        t_hat = torch.stack([-x[:,1]/r, x[:,0]/r], dim=1)  # unit tangent
        g_ang = beta_ang * tau * t_hat

        # radial anchor toward the ring
        r_hat = x / (r.view(-1,1) + 1e-9)
        g_rad = radial_anchor * (R_target - r).view(-1,1) * r_hat

        # combine + project orthogonally to u + swirl
        v_flat = u.view(u.size(0), -1)
        g_mix  = g_rep + g_ang + g_rad
        g_flat = g_mix.view(g_mix.size(0), -1)
        v2 = (v_flat*v_flat).sum(dim=1, keepdim=True) + 1e-12
        coef = ((g_flat*v_flat).sum(dim=1, keepdim=True)) / v2
        g_perp = (g_flat - coef * v_flat).view_as(g_mix)
        g_swirl = rotate90(g_perp) * swirl_coef
        g_div   = (g_perp + g_swirl)
        # remove any residual component along u
        g_flat2 = g_div.view(g_div.size(0), -1)
        coef2 = ((g_flat2*v_flat).sum(dim=1, keepdim=True)) / v2
        g_div = (g_flat2 - coef2 * v_flat).view_as(g_div)

        # orthogonal exploration noise (annealed)
        noise_strength = noise_scale_max * max(0.0, 1.0 - t0 / max(noise_t_end, 1e-6))
        if noise_strength > 0:
            xi = torch.randn_like(u)
            xi_proj = ((xi*v_flat).sum(dim=1, keepdim=True) / v2) * u
            xi_ortho = xi - xi_proj
            xi_ortho = xi_ortho / (xi_ortho.norm(dim=1, keepdim=True) + 1e-12)
            noise_disp = dt * noise_strength * xi_ortho
        else:
            noise_disp = torch.zeros_like(u)

        # gamma schedule + trust region (on combined div+noise)
        gamma    = gamma0 * sched_factor(t0)
        div_disp = dt * gamma * g_div
        comb = div_disp + noise_disp

        bn_base = (base_disp.view(n, -1).norm(dim=1) + 1e-12).view(-1, 1)
        bn_div  = (comb.view(n, -1).norm(dim=1) + 1e-12).view(-1, 1)
        cap     = gamma_max_ratio * bn_base
        scale   = torch.minimum(torch.ones_like(cap), cap / bn_div)
        comb    = scale * comb

        x = x + base_disp + comb
        traj.append(x.clone())

    return x, torch.stack(traj, dim=0)   # [steps+1, n, 2]

# ---------------- Plot helpers ----------------
def draw_means(ax, means: torch.Tensor):
    m = means.cpu().numpy()
    ax.scatter(m[:, 0], m[:, 1], marker='+', s=200, c='black', alpha=1.0, linewidths=1.0, zorder=10)

def subplot_snapshot(ax, x0: torch.Tensor, x_now: torch.Tensor, max_lines: Optional[int] = None,
                     line_alpha: float = 0.5):
    xi = x0.detach().cpu().numpy()
    xT = x_now.detach().cpu().numpy()
    n  = xi.shape[0]

    # initial (light blue)
    ax.scatter(xi[:, 0], xi[:, 1], s=22, alpha=0.7, marker='o', c='lightblue', zorder=2)

    # dashed connections
    idxs = np.arange(n)
    if (max_lines is not None) and (n > max_lines):
        idxs = np.random.default_rng(0).choice(n, size=max_lines, replace=False)
    segs = np.stack([xi[idxs], xT[idxs]], axis=1)
    cols = np.zeros((segs.shape[0], 4), dtype=np.float32); cols[:, 3] = line_alpha
    lc = LineCollection(segs, colors=cols, linewidths=0.9, linestyles='--', antialiased=True, zorder=1)
    ax.add_collection(lc)

    # final (orange x)
    ax.scatter(xT[:, 0], xT[:, 1], s=36, alpha=0.9, marker='x', c='orange', zorder=3)

    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([])

# ---------------- Run (2×3 grid) ----------------
set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Geometry & data
R = 2.2
Kmodes = 16
sigma = 0.14
gmm = GMM2D.circle(K=Kmodes, radius=R, sigma=sigma)
means = gmm.means.to(device)

# Train unconditional CFM (shallower to magnify sampler difference)
model = TinyCFMUncond(hidden=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for it in range(400):  # shallower
    x0_b, _ = gmm.sample(512, device=device)
    loss = fm_loss_uncond(model, x0_b)
    opt.zero_grad(); loss.backward(); opt.step()

# Same z0 for both samplers: inner disk + small shift (harder geometry, optional)
N_PARTICLES = 10
r_init = 0.3 * R
z0 = sample_in_disk(N_PARTICLES, radius=r_init, device=device, shift=(0.10*R, 0.0))

# Inference with trajectory (same NFE)
STEPS = 800
x_base_final,  traj_base  = integrate_flowmatch_uncond_traj(model, x0=z0, steps=STEPS)
x_oscar_final, traj_oscar = integrate_oscar_uncond_traj(
    model, x0=z0, steps=STEPS, means=means, gmm_sigma=float(gmm.sigma), R_target=R,
    gamma0=1.0, gamma_max_ratio=1.4,
    sigma_theta=0.35, beta_ang=1.6, radial_anchor=0.25,
    noise_scale_max=0.55, noise_t_end=0.55,
    swirl_coef=0.55,
    repel_sigma=0.28, repulsion_weight=1.6,
)

# Pick three snapshots: early / mid / final
snap_indices = [100, 400, STEPS]  # (or use fractions -> int(f*STEPS))
col_labels   = [f"Step {i}/{STEPS}" for i in snap_indices]
row_labels   = ["Standard FM", "OSCAR"]

# Figure
out_dir = "outputs/circle24_innerdisk_2x3"
os.makedirs(out_dir, exist_ok=True)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 5.5), constrained_layout=False)

# Global title
fig.suptitle("Flow Matching vs. OSCAR — Progress Over Time", fontsize=20, y=0.99)

# Titles & row labels
for r in range(2):
    for c in range(3):
        if r == 0:
            axes[r, c].set_title(col_labels[c], fontsize=15, pad=2)
        if c == 0:
            axes[r, c].annotate(row_labels[r], xy=(-0.1, 0.5), xycoords='axes fraction',
                                fontsize=15, rotation=90, va='center', ha='center')

# Draw snapshots
for c, idx in enumerate(snap_indices):
    # baseline
    draw_means(axes[0, c], means)
    subplot_snapshot(axes[0, c], z0, traj_base[idx], max_lines=None, line_alpha=0.45)
    # oscar
    draw_means(axes[1, c], means)
    subplot_snapshot(axes[1, c], z0, traj_oscar[idx], max_lines=None, line_alpha=0.45)

# Tight layout (compact like before)
plt.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.06, wspace=0.05, hspace=0.08)

# Save
png = os.path.join(out_dir, "grid2x3_circle24.png")
pdf = os.path.join(out_dir, "grid2x3_circle24.pdf")
plt.savefig(png, dpi=300, bbox_inches='tight')
plt.savefig(pdf, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved:", png, pdf)