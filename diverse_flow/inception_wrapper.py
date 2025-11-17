from typing import Optional, Tuple, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV3Wrapper(nn.Module):
    """
    Inception v3 image-encoder wrapper for volume/diversity objectives.

    - Loads torchvision Inception v3 backbone with aux_logits=False
    - Accepts a local checkpoint (.pth/.pt). If None, uses randomly-initialized weights (not recommended)
    - Exposes a single API compatible with your CLIP wrapper:
        encode_image_from_pixels(x, size) -> [B, D] L2-normalized features
    - Uses ImageNet mean/std and default input size 299x299
    - Feature is taken from avgpool (penultimate, 2048-d)

    Notes on gradients:
      * Do NOT wrap the forward in torch.no_grad(); the volume objective needs
        gradients from features back to pixels. This wrapper keeps the graph.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Lazily import to avoid hard dep when not used
        from torchvision.models import inception_v3

        # Build backbone (we don't rely on the classification head)
        self.core = inception_v3(weights=None, aux_logits=False)

        # Load local checkpoint if provided (robust to various dict formats)
        if checkpoint_path is not None:
            ckpt = torch.load(os.path.expanduser(checkpoint_path), map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            # Strip "module." prefix if present
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = self.core.load_state_dict(state, strict=False)
            if unexpected:
                print(f"[InceptionV3Wrapper] Unexpected keys ignored: {sorted(unexpected)[:8]} ...")
            if missing:
                # Missing fc/aux keys are fine since we only use avgpool features
                print(f"[InceptionV3Wrapper] Missing keys (ok if mostly fc/aux): {sorted(missing)[:8]} ...")

        self.core.eval().to(self.device)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Public feature dim (avgpool -> 2048)
        self.D = 2048

    @torch.enable_grad()
    def encode_image_from_pixels(
        self,
        x: torch.Tensor,
        size: Union[int, Tuple[int, int]] = 299,
    ) -> torch.Tensor:
        """
        Args:
            x: [B,3,H,W] in [0,1]
            size: int or (H,W); default 299 for Inception v3
        Returns:
            feats: [B,2048] L2-normalized float32
        """
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"x shape must be [B,3,H,W], got {tuple(x.shape)}")

        x = x.to(self.device, dtype=torch.float32)

        if isinstance(size, int):
            size = (size, size)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)

        # Normalize to ImageNet stats
        # move buffers to the same device/dtype as x
        mean = self.mean.to(x.device, dtype=x.dtype)
        std  = self.std.to(x.device, dtype=x.dtype)
        x = (x - mean) / std

        # Hook avgpool output (shape [B,2048,1,1]) so we can backprop to pixels
        feats_holder = {}

        def _hook(_m, _inp, out):
            feats_holder["feat"] = out  # keep graph

        h = self.core.avgpool.register_forward_hook(_hook)
        # Forward WITHOUT no_grad (we need autograd graph)
        _ = self.core(x)
        h.remove()

        feats = feats_holder.get("feat", None)
        if feats is None:
            raise RuntimeError("Failed to capture Inception v3 avgpool features.")

        feats = torch.flatten(feats, 1).float()  # [B, 2048]
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        return feats

    def get_feature_dim(self) -> int:
        return int(self.D)
