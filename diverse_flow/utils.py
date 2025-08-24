from typing import Tuple
import math
import torch
import torch.nn.functional as F


def make_t_steps(n:int, t0:float=1.0, t1:float=0.0) -> torch.Tensor:
    return torch.linspace(t0, t1, n+1)


def sched_factor(t: float, t_gate: Tuple[float,float], mode:str="sin2") -> float:
    t1, t2 = t_gate
    if not (t1 <= t <= t2):
        return 0.0
    u = (t - t1) / max(t2 - t1, 1e-8)
    if mode == "sin2":
        return math.sin(math.pi * u)**2
    elif mode == "t1mt":
        return u * (1.0 - u)
    else:
        return 1.0


def project_partial_orth(g: torch.Tensor, v: torch.Tensor, lam: float, eps: float=1e-12) -> torch.Tensor:
    v_flat = v.view(v.size(0), -1)
    g_flat = g.view(g.size(0), -1)
    v2 = (v_flat*v_flat).sum(dim=1, keepdim=True) + eps
    coeff = ((g_flat*v_flat).sum(dim=1, keepdim=True) / v2)
    proj = coeff * v_flat
    g_orth = g_flat - lam * proj
    return g_orth.view_as(g)


def batched_norm(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    return x.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)


def pairwise_cosine_angles(Z: torch.Tensor) -> torch.Tensor:
    S = Z @ Z.t()
    S = S - torch.eye(S.size(0), device=S.device)
    S = S.clamp(-1, 1)
    angles = torch.acos(S) * 180.0 / math.pi
    return angles