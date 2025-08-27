# cads/cads.py
# -*- coding: utf-8 -*-
"""
CADS for Flow-Matching on SD-3.5 (Diffusers)
--------------------------------------------

This module implements Condition-Annealed Diffusion Sampling (CADS)
as a *sampling-time* hook for Diffusers' SD3/SD3.5 pipelines (rectified
flow / flow matching). It requires **no extra training** and uses only
your base model.

Core updates per step (i = 0..N-1):
  y_hat = sqrt(gamma)*y + s*sqrt(1-gamma)*eps, eps~N(0,I)
  (optional) rescale y_hat back to original (mu, sigma)
  (optional) mix with psi: y_final = psi*y_rescaled + (1-psi)*y_noised
  (optional) dynamic CFG: guidance_scale <- gamma * guidance_scale

Use it as Diffusers' `callback_on_step_end` or call `apply_to_dict`
in your own sampling loop.

Keys handled (may vary across Diffusers versions):
  prompt_embeds, negative_prompt_embeds,
  pooled_prompt_embeds, negative_pooled_prompt_embeds,
  text_embeds, negative_text_embeds, add_text_embeds, negative_add_text_embeds
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, MutableMapping

import torch

Tensor = torch.Tensor


@torch.no_grad()
def _stats_lastdim(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (mean, std) along the last dimension (numerically stable)."""
    mu = x.mean(dim=-1, keepdim=True)
    sig = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mu, sig


@dataclass
class CADSConfig:
    num_inference_steps: int
    tau1: float = 0.6
    tau2: float = 0.9
    s: float = 0.10
    psi: float = 1.0
    rescale: bool = True
    dynamic_cfg: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        assert 0.0 <= self.tau1 <= self.tau2 <= 1.0, "Require 0<=tau1<=tau2<=1"
        assert self.num_inference_steps >= 1
        assert 0.0 <= self.s <= 1.0
        assert 0.0 <= self.psi <= 1.0


class CADSConditionAnnealer:
    """Callable usable as Diffusers `callback_on_step_end`.
    It modifies text-conditioning tensors each step per CADS and can shrink
    CFG with the same gamma schedule.
    """

    COND_KEYS = (
        "prompt_embeds", "negative_prompt_embeds",
        "pooled_prompt_embeds", "negative_pooled_prompt_embeds",
        # alternative names (compat across versions)
        "text_embeds", "negative_text_embeds",
        "add_text_embeds", "negative_add_text_embeds",
    )

    def __init__(self, **cfg_kwargs):
        # accept either CADSConfig or kwargs
        if len(cfg_kwargs) == 1 and isinstance(next(iter(cfg_kwargs.values())), CADSConfig):
            self.cfg: CADSConfig = next(iter(cfg_kwargs.values()))
        else:
            self.cfg = CADSConfig(**cfg_kwargs)

        self._orig_stats: Dict[str, tuple[Tensor, Tensor]] = {}
        self._rng = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            self._rng.manual_seed(int(self.cfg.seed))

    # -------- scheduling --------
    def gamma(self, step_idx: int) -> float:
        # Step index -> t in [0,1], decreasing from 1->0
        Nm1 = max(self.cfg.num_inference_steps - 1, 1)
        t = 1.0 - float(step_idx) / float(Nm1)
        if t <= self.cfg.tau1:
            return 1.0
        if t >= self.cfg.tau2:
            return 0.0
        # piecewise linear between (tau1, tau2)
        return (self.cfg.tau2 - t) / (self.cfg.tau1 - self.cfg.tau2)

    # -------- core op --------
    @torch.no_grad()
    def _anneal_tensor(self, key: str, y: Tensor, gamma: float) -> Tensor:
        device, dtype = y.device, y.dtype
        g = torch.tensor(gamma, device=device, dtype=dtype)
        eps = torch.randn_like(y, generator=self._rng)
        y_noised = torch.sqrt(g) * y + self.cfg.s * torch.sqrt(1 - g) * eps

        if not self.cfg.rescale:
            return y_noised

        # memorize original stats on first occurrence
        if key not in self._orig_stats:
            mu_in, sig_in = _stats_lastdim(y)
            self._orig_stats[key] = (mu_in.detach(), sig_in.detach())

        mu_in, sig_in = self._orig_stats[key]
        mu_now, sig_now = _stats_lastdim(y_noised)
        y_rescaled = (y_noised - mu_now) / sig_now * sig_in + mu_in

        if self.cfg.psi == 1.0:
            return y_rescaled
        return self.cfg.psi * y_rescaled + (1 - self.cfg.psi) * y_noised

    # -------- public helper for custom loops --------
    @torch.no_grad()
    def apply_to_dict(
        self, step_idx: int, cond: MutableMapping[str, Tensor],
        guidance_scale: Optional[float] = None
    ) -> Optional[float]:
        """Apply CADS to a dict of conditioning tensors.
        Returns possibly updated guidance_scale (if dynamic_cfg), else None.
        """
        g = self.gamma(step_idx)
        new_gs = None
        if self.cfg.dynamic_cfg and guidance_scale is not None:
            new_gs = float(guidance_scale) * float(g)

        for k in list(cond.keys()):
            v = cond.get(k, None)
            if isinstance(v, torch.Tensor):
                cond[k] = self._anneal_tensor(k, v, g)
        return new_gs

    # -------- Diffusers callback entry --------
    @torch.no_grad()
    def __call__(self, pipe, step_index: int, timestep, callback_kwargs: dict):
        g = self.gamma(step_index)

        # dynamic CFG (if available)
        if self.cfg.dynamic_cfg and "guidance_scale" in callback_kwargs:
            callback_kwargs["guidance_scale"] = float(callback_kwargs["guidance_scale"]) * float(g)

        # apply to visible conditioning tensors
        for k in self.COND_KEYS:
            v = callback_kwargs.get(k, None)
            if isinstance(v, torch.Tensor):
                callback_kwargs[k] = self._anneal_tensor(k, v, g)
        return callback_kwargs


# export-friendly alias
CADS = CADSConditionAnnealer