from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHFWrapper(nn.Module):
    """
    HuggingFace Dinov2 encoder wrapper (uses transformers AutoModel).

    - API is aligned with our other encoders:
        encode_image_from_pixels(x, size) -> [B,D] (L2-normalized)
    - Loads from a local snapshot dir (e.g., ~/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/<hash>)
    - Performs resize & normalization inside using config.image_mean/std and image_size if present
    - Gradients flow from features back to pixels (no no_grad here)

    Args:
      repo_or_path: local snapshot directory (recommended) or HF repo id
      device: device for the model; normalization done on the same device
      pool: 'cls' or 'mean' token pooling (default 'cls')
      in_size: override input size (int). If None, try to read from config/image_size, else 224
    """

    def __init__(
        self,
        repo_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        pool: str = "cls",
        in_size: Optional[int] = None,
        local_only: bool = True,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.pool = pool

        try:
            from transformers import AutoConfig, AutoModel
        except Exception as e:
            raise ImportError("transformers is required for DINOHFWrapper. pip install transformers") from e

        # load config & model (prefer local)
        self.config = AutoConfig.from_pretrained(repo_or_path, local_files_only=local_only)
        self.core = AutoModel.from_pretrained(repo_or_path, local_files_only=local_only)
        self.core.eval().to(self.device)

        # infer input size & normalization
        def _to_int(sz):
            if isinstance(sz, (list, tuple)):
                return int(sz[0])
            try:
                return int(sz)
            except Exception:
                return None

        self.in_size = int(in_size) if in_size is not None else (_to_int(getattr(self.config, 'image_size', None)) or 224)

        mean = getattr(self.config, 'image_mean', [0.485, 0.456, 0.406])
        std  = getattr(self.config, 'image_std',  [0.229, 0.224, 0.225])
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std",  std_t,  persistent=False)

        # feature dim: for ViT-like models, use hidden_size
        self.D = int(getattr(self.config, 'hidden_size', 768))

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

        # Transformers AutoModel expects channels-last? No; we pass as is by creating pixel_values
        # The HF Dinov2 models expect input named 'pixel_values' in NCHW with float32
        outputs = self.core(pixel_values=x)
        # last_hidden_state: [B, N, C]
        tokens = outputs.last_hidden_state
        if tokens.dim() != 3:
            raise RuntimeError("Unexpected HF model output shape")

        if self.pool == 'cls':
            z = tokens[:, 0]  # [B, C]
        else:
            if tokens.size(1) > 1:
                z = tokens[:, 1:].mean(dim=1)
            else:
                z = tokens.mean(dim=1)

        z = z.float()
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
        return z

    def get_feature_dim(self) -> int:
        return int(self.D)

    def get_input_size(self) -> int:
        return int(self.in_size)
