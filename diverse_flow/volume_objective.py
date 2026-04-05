from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .config import DiversityConfig
from .clip_wrapper import CLIPWrapper
from .utils import pairwise_cosine_angles

class VolumeObjective:
    def __init__(self, clip_model: CLIPWrapper, cfg: DiversityConfig):
        self.clip = clip_model
        self.cfg = cfg

    def _features(self, x: torch.Tensor, allow_whiten: bool = True) -> torch.Tensor:
        """
        x: [B,3,H,W] in [0,1]
        allow_whiten: set False when called from Pass B (VJP) to preserve grad graph.
                      Whitening uses .detach() internally and breaks backprop.
        """
        phi = self.clip.encode_image_from_pixels(x, size=self.cfg.clip_image_size)  # [B,D]

        if self.cfg.feature_l2norm:
            phi = F.normalize(phi, dim=-1)

        if self.cfg.feature_center:
            phi = phi - phi.mean(dim=0, keepdim=True)

        if (allow_whiten
                and getattr(self.cfg, "whiten", False)
                and phi.size(0) >= getattr(self.cfg, "whiten_min_B", 16)):
            B, D = phi.shape
            phi_cpu = phi.detach().float().to("cpu")
            cov = (phi_cpu.t() @ phi_cpu) / max(B - 1, 1)
            cov = cov + self.cfg.eps_logdet * torch.eye(D, device=cov.device, dtype=cov.dtype)
            evals, evecs = torch.linalg.eigh(cov)
            inv_sqrt = (evecs * evals.clamp_min(1e-5).rsqrt()).mm(evecs.t())
            phi = (phi_cpu @ inv_sqrt).to(phi.device, dtype=phi.dtype)

        return phi


    def _kernel(self, phi: torch.Tensor) -> torch.Tensor:
        return phi @ phi.t()

    def _logdetI_tauK(self, K: torch.Tensor) -> torch.Tensor:
        B = K.size(0)
        I = torch.eye(B, device=K.device, dtype=K.dtype)
        # A：trace-scaled ε
        eps_eff = self.cfg.eps_logdet * (torch.trace(K) / max(B, 1))
        M = I + self.cfg.tau * K + eps_eff * I
        sign, logabsdet = torch.linalg.slogdet(M)
        return (sign.clamp_min(0.0) * logabsdet)


    def _leverage_weights(self, K: torch.Tensor) -> torch.Tensor:
        if self.cfg.leverage_alpha <= 0:
            return torch.ones(K.size(0), device=K.device, dtype=K.dtype)
        B = K.size(0)
        I = torch.eye(B, device=K.device, dtype=K.dtype)
        M = K + self.cfg.eps_logdet * I
        L = torch.linalg.cholesky(M)
        Minv = torch.cholesky_inverse(L)
        s = Minv.diag().clamp_min(1e-8)
        w = s.pow(self.cfg.leverage_alpha)
        w = w * (B / w.sum().clamp_min(1e-8))
        return w

    def volume_loss_and_grad(self, x_in: torch.Tensor, phi_memory: torch.Tensor = None):
        """
        Compute -logdet diversity loss and gradient w.r.t. x_in.

        Args:
            x_in:       [B, 3, H, W]  current batch of (decoded) images, range [0,1]
            phi_memory: [M, D] optional detached features from a memory bank; if given,
                        the Gram matrix is built over [B+M] features but gradients only
                        flow through the B current features (memory is treated as constant).
        Returns:
            (loss_scalar, grad_x [B,3,H,W], logs dict)
        """
        dev = self.clip.device
        use_amp = (dev.type == "cuda")
        B = x_in.size(0)
        chunk = int(getattr(self.cfg, "clip_grad_chunk", B))  # default: full batch
        chunk = max(1, min(chunk, B))

        # ── Pass A: no-grad, compute phi for current batch ────────────────
        with torch.no_grad():
            x_all = x_in.detach().to(dev, dtype=torch.float32)
            phi_list = []
            for s in range(0, B, chunk):
                phi_list.append(self._features(x_all[s:s+chunk], allow_whiten=True))
            phi_all = torch.cat(phi_list, dim=0)  # [B, D]

        # ── Build Gram matrix (optionally augmented with memory) ──────────
        with torch.enable_grad():
            phi_leaf = phi_all.detach().to(torch.float32).clone().requires_grad_(True)  # [B, D]

            if phi_memory is not None and phi_memory.size(0) > 0:
                # Memory features are constants — no gradient flows through them.
                # The cross-terms still push current features away from past ones.
                phi_mem = phi_memory.detach().to(dev, dtype=torch.float32)
                phi_aug = torch.cat([phi_leaf, phi_mem], dim=0)   # [B+M, D]
            else:
                phi_aug = phi_leaf                                 # [B, D]

            K = self._kernel(phi_aug)   # [(B+M), (B+M)]
            w = self._leverage_weights(K)
            if (w is not None) and (w.ndim == 1):
                Wsqrt = torch.diag(w.clamp_min(1e-8).sqrt())
                K = Wsqrt @ K @ Wsqrt

            logdet = 0.5 * self._logdetI_tauK(K)
            loss   = -logdet

            # grad w.r.t. phi_leaf only ([B, D]); memory rows have no leaf
            grad_phi = torch.autograd.grad(
                loss, phi_leaf, retain_graph=False, create_graph=False
            )[0].detach()   # [B, D]

        # ── Pass B: VJP — chunked CLIP re-forward to get grad w.r.t. x ──
        grads = []
        for s in range(0, B, chunk):
            x_chunk = (
                x_in[s:s+chunk].detach()
                .to(dev, dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            with torch.enable_grad():
                # allow_whiten=False: whitening uses .detach() and breaks the grad graph
                phi_chunk = self._features(x_chunk, allow_whiten=False)   # [cs, D]
                go = grad_phi[s:s+chunk].to(phi_chunk.dtype)
                g_chunk = torch.autograd.grad(
                    phi_chunk, x_chunk, grad_outputs=go,
                    retain_graph=False, create_graph=False
                )[0]
            grads.append(g_chunk.detach())
            del x_chunk, phi_chunk, g_chunk, go

        grad_x = torch.cat(grads, dim=0)   # [B, 3, H, W]

        with torch.no_grad():
            from .utils import pairwise_cosine_angles
            angles = pairwise_cosine_angles(F.normalize(phi_all, dim=-1))
            pos = angles > 0
            min_angle  = float(angles[pos].min())  if pos.any() else 0.0
            mean_angle = float(angles[pos].mean()) if pos.any() else 0.0
            logs = dict(
                logdet=float(logdet.detach()),
                loss=float(loss.detach()),
                min_angle_deg=min_angle,
                mean_angle_deg=mean_angle,
            )

        return loss.detach(), grad_x, logs