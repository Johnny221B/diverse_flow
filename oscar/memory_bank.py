"""
OfflineMemoryBank — enables diversity guidance when m is small.

Usage pattern:
    bank = OfflineMemoryBank(max_size=128, decay=0.0)

    # For each new generation batch:
    # 1. At each diversity step, pass bank.get() to volume_loss_and_grad:
    #        _loss, grad, logs = vol.volume_loss_and_grad(imgs_current, phi_memory=bank.get())
    # 2. After generation is complete, update the bank with the endpoint features:
    #        bank.update(phi_new)   # phi_new: [B, D] computed from final images

This lets a batch of m=4 images repel against the last ~128 features from prior
generations of the same prompt, effectively making the diversity signal global.
"""

from __future__ import annotations
from typing import Optional
import torch


class OfflineMemoryBank:
    """
    Circular buffer of CLIP endpoint features from previous generations.

    Args:
        max_size:   Maximum number of feature vectors to retain.
        decay:      Exponential decay weight for older entries (0.0 = uniform).
                    When > 0, older features contribute less to the Gram matrix.
        device:     Where to store the bank ('cpu' keeps VRAM free).
    """

    def __init__(
        self,
        max_size: int = 128,
        decay: float = 0.0,
        device: str = "cpu",
    ):
        self.max_size = max_size
        self.decay = decay
        self.device = torch.device(device)
        self._features: list[torch.Tensor] = []   # each entry: [D]
        self._ages: list[int] = []                 # step count when added
        self._clock = 0

    # ------------------------------------------------------------------ #

    def update(self, phi: torch.Tensor) -> None:
        """
        Add new features to the bank.

        Args:
            phi: [B, D] float tensor — detached CLIP features of newly generated images.
        """
        phi = phi.detach().to(self.device, dtype=torch.float32)
        for vec in phi:
            self._features.append(vec)
            self._ages.append(self._clock)
            self._clock += 1
        # Trim to max_size (drop oldest first)
        if len(self._features) > self.max_size:
            excess = len(self._features) - self.max_size
            self._features = self._features[excess:]
            self._ages     = self._ages[excess:]

    def get(self, query_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Return stacked memory features [M, D], or None if bank is empty.
        Applies exponential decay weighting if self.decay > 0.

        Args:
            query_device: Move output to this device (default: self.device).
        """
        if not self._features:
            return None

        stack = torch.stack(self._features, dim=0)   # [M, D]

        if self.decay > 0.0 and len(self._ages) > 1:
            max_age = self._clock - 1
            ages    = torch.tensor(self._ages, dtype=torch.float32, device=stack.device)
            weights = torch.exp(-self.decay * (max_age - ages))   # [M]
            # Scale features by sqrt(weight) so K_cross = phi_mem^T @ phi_cur
            # contributes proportionally; grad still only flows through phi_cur.
            stack = stack * weights.unsqueeze(1).sqrt()

        if query_device is not None:
            stack = stack.to(query_device)

        return stack

    def size(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        self._features.clear()
        self._ages.clear()
        self._clock = 0

    def __repr__(self) -> str:
        return (f"OfflineMemoryBank(size={self.size()}/{self.max_size}, "
                f"decay={self.decay}, device={self.device})")
