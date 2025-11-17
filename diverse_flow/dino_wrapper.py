from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOEncoderWrapper(nn.Module):
    """
    DINO / DINOv2 image-encoder wrapper (timm backend by default).

    Goals:
      - Duck-typed with CLIP/Inception wrapper: encode_image_from_pixels(x, size) -> [B,D] L2-normalized
      - ImageNet normalization inside; default input size 224
      - Supports local checkpoint (.pth/.pt). If None and pretrained=True, timm will try to load pretrained (online) weights.

    Args:
      backend: 'timm' (default)
      arch   : timm model name, e.g. 'vit_base_patch14_dinov2' (default),
               or 'vit_large_patch14_dinov2','vit_small_patch16_224.dino'
      checkpoint_path: local .pth/.pt (optional). If provided, we load it with strict=False
      device: cuda/cpu
      pool  : 'cls' or 'mean' token pooling (default 'cls')
      in_size: input resolution (default 224)
    """

    def __init__(
        self,
        backend: str = "timm",
        arch: str = "vit_base_patch14_dinov2",
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        pool: str = "cls",
        in_size: int = 224,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.backend = backend
        self.arch = arch
        self.pool = pool
        self.in_size = in_size

        if backend != "timm":
            raise ValueError("Only backend='timm' is supported right now")

        try:
            import timm  # type: ignore
        except Exception as e:
            raise ImportError("timm is required for DINOEncoderWrapper. pip install timm") from e

        # create model; no classifier head
        self.core = timm.create_model(arch, pretrained=False, num_classes=0)

        if checkpoint_path is not None:
            ckpt = torch.load(os.path.expanduser(checkpoint_path), map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.core.load_state_dict(state, strict=False)

        self.core.eval().to(self.device)

        # infer feature dim
        D = getattr(self.core, "num_features", None)
        if D is None:
            # fallback probe
            with torch.no_grad():
                x = torch.zeros(1, 3, in_size, in_size)
                y = self._forward_tokens(x)
                D = y.shape[-1]
        self.D = int(D)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT forward_features returns sequence/token embeddings or pooled
        feats = self.core.forward_features(x)  # shape depends on model; for ViT it's [B, N, C] or dict
        if isinstance(feats, dict):
            # dinov2 timm returns dict with 'x_norm' (tokens) or 'feat' depending on version
            if 'x_norm' in feats:
                tokens = feats['x_norm']
            elif 'forward_features' in feats:
                tokens = feats['forward_features']
            elif 'feat' in feats:
                tokens = feats['feat']
            else:
                # best effort
                tokens = next((v for v in feats.values() if torch.is_tensor(v) and v.dim()==3), None)
                if tokens is None:
                    raise RuntimeError("Unexpected timm forward_features output")
        else:
            tokens = feats
        return tokens  # [B,N,C]

    @torch.enable_grad()
    def encode_image_from_pixels(
        self,
        x: torch.Tensor,
        size: Union[int, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if size is None:
            size = self.in_size
        if isinstance(size, int):
            size = (size, size)
        if not torch.is_tensor(x) or x.dim()!=4 or x.size(1)!=3:
            raise ValueError(f"x must be [B,3,H,W], got {tuple(x.shape) if torch.is_tensor(x) else type(x)}")

        x = x.to(self.device, dtype=torch.float32)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)

        mean = self.mean.to(x.device, dtype=x.dtype)
        std  = self.std.to(x.device, dtype=x.dtype)
        x = (x - mean) / std

        tokens = self._forward_tokens(x)  # [B,N,C]
        if tokens.dim() == 2:
            # already pooled
            z = tokens
        else:
            if self.pool == "cls":
                z = tokens[:, 0]   # [B,C]
            else:  # mean pool (exclude cls if N>1)
                if tokens.size(1) > 1:
                    z = tokens[:, 1:].mean(dim=1)
                else:
                    z = tokens.mean(dim=1)
        z = z.float()
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z

    def get_feature_dim(self) -> int:
        return int(self.D)
