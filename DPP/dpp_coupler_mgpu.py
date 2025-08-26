# -*- coding: utf-8 -*-
# File: flow_base/DPP/dpp_coupler_mgpu.py
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F

# ---- utils ----
def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5

def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def _pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    return (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)

def _logdet_stable(mat: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    I = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
    return torch.linalg.slogdet(mat + eps * I)[1]

# ---- config ----
@dataclass
class MGPUConfig:
    decode_size: int = 256           # 解码到这么大再送 CLIP
    kernel_spread: float = 3.0       # DPP 核宽
    gamma_max: float = 0.12          # 全局强度（会除以梯度均范数）
    gamma_sched: str = "sqrt"        # "sqrt" | "sin2" | "poly"
    clip_grad_norm: float = 5.0
    chunk_size: int = 2              # 分块反传
    use_quality_term: bool = False   # 关闭质量项，做纯 DPP baseline

def _gamma_t(t: float, grad_mean_norm: torch.Tensor, cfg: MGPUConfig) -> float:
    if cfg.gamma_sched == "sqrt":
        sched = (1.0 - t) ** 0.5
    elif cfg.gamma_sched == "sin2":
        sched = math.sin(math.pi * t) ** 2
    else:
        sched = t * (1.0 - t)
    return float(cfg.gamma_max * sched / (grad_mean_norm.item() + 1e-8))

# ---- core ----
class DPPCouplerMGPU:
    """
    多 GPU 版本 DPP 耦合：
    - transformer/text_encoders 在 dev_tr
    - VAE 在 dev_vae
    - CLIP(JIT) 在 dev_clip
    回调里按 chunk：
      z(dev_vae, requires_grad) -> decode(imgs) -> to(dev_clip) -> DPP loss -> dL/dimgs
      dimgs 回到 dev_vae -> autograd.grad(imgs, z, dimgs) -> dL/dz
      dL/dz 回到 latents.device(dev_tr) 更新该 chunk 的 latents
    """
    def __init__(
        self,
        pipe,
        feat_module,                   # CLIP 特征塔（已在 dev_clip）
        dev_tr: torch.device,
        dev_vae: torch.device,
        dev_clip: torch.device,
        cfg: MGPUConfig = MGPUConfig(),
    ):
        self.pipe = pipe
        self.feat = feat_module
        self.dev_tr = dev_tr
        self.dev_vae = dev_vae
        self.dev_clip = dev_clip
        self.cfg = cfg
        self.vae = pipe.vae
        self.vae_scale = getattr(self.vae.config, "scaling_factor", 1.0)
        self.vae_dtype = next(self.vae.parameters()).dtype

    # ---- helpers ----
    def _vae_decode(self, z_vae: torch.Tensor) -> torch.Tensor:
        # z_vae 在 dev_vae
        img = self.vae.decode(z_vae / self.vae_scale, return_dict=False)[0]  # [-1,1]
        img = _to01(img).float()
        if img.shape[-2] > self.cfg.decode_size or img.shape[-1] > self.cfg.decode_size:
            img = _resize(img, (self.cfg.decode_size, self.cfg.decode_size))
        return img

    def _dpp_ll(self, feats_full: torch.Tensor) -> torch.Tensor:
        # feats_full 在 dev_clip，已 L2-normalized
        D2 = _pairwise_sqdist(feats_full.float())
        triu = torch.triu(D2, diagonal=1)
        med = torch.median(triu[triu > 0]).clamp_min(1e-6)
        K = torch.exp(-self.cfg.kernel_spread * D2 / med)
        I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        return _logdet_stable(K) - _logdet_stable(K + I)

    # 预计算“全批常量特征”（不建图），返回 list[Tensor(B_i, D)]，对应批内顺序
    @torch.no_grad()
    def _precompute_feats_const(self, latents: torch.Tensor) -> List[torch.Tensor]:
        B = latents.size(0)
        feats_const: List[torch.Tensor] = []
        for s in range(0, B, self.cfg.chunk_size):
            e = min(B, s + self.cfg.chunk_size)
            z = latents[s:e].to(self.dev_vae, dtype=self.vae_dtype, non_blocking=True)
            imgs = self._vae_decode(z)                                      # dev_vae
            imgs_c = imgs.to(self.dev_clip, dtype=torch.float32, non_blocking=True)  # CLIP 侧统一用 fp32
            feats = self.feat(imgs_c).detach()                               # dev_clip, no grad
            feats_const += [feats]
            del z, imgs, imgs_c, feats
        return feats_const  # list of chunks; 拼接时注意顺序

    def __call__(self, step_index: int, t_norm: float, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B,C,H,W] on dev_tr（transformer 卡）
        返回更新后的 latents（仍在 dev_tr）
        """
        B = latents.size(0)
        if B < 2:
            return latents

        # 第一遍：常量特征（others），不建图
        feats_const_chunks = self._precompute_feats_const(latents)  # 每块 [bs,D]，在 dev_clip
        feats_const = torch.cat(feats_const_chunks, dim=0)          # [B,D] dev_clip

        # 逐块反传
        lat_new = latents.clone()
        grad_norm_acc = []

        # 为构造 full feats：提前准备一个索引表
        idx_all = torch.arange(B, device=self.dev_clip)

        for s in range(0, B, self.cfg.chunk_size):
            e = min(B, s + self.cfg.chunk_size)

            # 1) leaf latent on VAE card
            # z = latents[s:e].to(self.dev_vae, non_blocking=True).detach().clone().requires_grad_(True)
            z = (
                latents[s:e]
                .to(self.dev_vae, dtype=self.vae_dtype, non_blocking=True)
                .detach()
                .clone()
                .requires_grad_(True)
            )

            with torch.enable_grad():
                # 2) decode -> imgs (dev_vae) -> to CLIP (dev_clip)
                imgs = self._vae_decode(z)                                   # dev_vae
                # imgs_clip = imgs.to(self.dev_clip, non_blocking=True)        # dev_clip
                imgs_clip = imgs.to(self.dev_clip, dtype=torch.float32, non_blocking=True)

                # 3) feats for this chunk WITH grad
                feats_chunk = self.feat(imgs_clip)                           # dev_clip, requires grad

                # 4) assemble full feats (others const, this chunk variable)
                feats_full = feats_const.clone()
                feats_full[s:e] = feats_chunk

                # 5) DPP loss on dev_clip
                ll = self._dpp_ll(feats_full)

                # 6) dL/d imgs_clip (dev_clip)
                grad_img_clip = torch.autograd.grad(ll, imgs_clip, retain_graph=False, create_graph=False)[0]
                grad_img_clip = grad_img_clip.to(dtype=imgs_clip.dtype)

            # 7) 拉回到 VAE 卡，算 dL/d z via VJP
            grad_img_vae = grad_img_clip.to(self.dev_vae, non_blocking=True).to(dtype=imgs.dtype)
            grad_z = torch.autograd.grad(outputs=imgs, inputs=z, grad_outputs=grad_img_vae,
                                         retain_graph=False, create_graph=False, allow_unused=False)[0]  # dev_vae

            # 8) 归一化、裁剪、γ(t)
            gn = grad_z.flatten(1).norm(dim=1).mean().clamp_min(1e-8)
            grad_norm_acc.append(gn)
            if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                per = grad_z.flatten(1).norm(dim=1, p=2)
                scale = (self.cfg.clip_grad_norm / (per + 1e-6)).clamp_max(1.0)
                grad_z = grad_z * scale.view(-1, 1, 1, 1)
            gamma = _gamma_t(t_norm, gn, self.cfg)

            # 9) 写回 transformer 卡上的对应切片
            delta = (gamma * grad_z).to(lat_new.device, non_blocking=True).to(dtype=lat_new.dtype)
            lat_new[s:e] = lat_new[s:e] - delta

            # 10) 清理
            del z, imgs, imgs_clip, feats_chunk, feats_full, ll, grad_img_clip, grad_img_vae, grad_z, delta

            if self.dev_clip.type == 'cuda':
                torch.cuda.synchronize(self.dev_clip)
            if self.dev_vae.type == 'cuda':
                torch.cuda.synchronize(self.dev_vae)

        return lat_new
