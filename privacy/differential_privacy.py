"""
Differential Privacy Module
Protects patient data by adding calibrated Gaussian noise to model gradients
during local training, preventing reconstruction of individual records.

Two approaches are provided:
  1. Manual gradient perturbation (no extra dependencies).
  2. Opacus-based DP training (recommended for production).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utility: add Gaussian noise to a tensor (used for weight perturbation)
# ─────────────────────────────────────────────────────────────────────────────
def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_multiplier: float = 1.1,
    sensitivity: float = 1.0,
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise to tensor values.

    sigma = noise_multiplier * sensitivity
    noise ~ N(0, sigma^2 * I)

    Args:
        tensor:           the tensor to perturb
        noise_multiplier: ratio of noise std to sensitivity (controls privacy budget)
        sensitivity:      L2 sensitivity of the mechanism (≈ clipping threshold)

    Returns:
        Perturbed tensor (same shape, same device).
    """
    sigma = noise_multiplier * sensitivity
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


# ─────────────────────────────────────────────────────────────────────────────
# 2. Manual DP-SGD implementation (per-sample gradient clipping + noise)
# ─────────────────────────────────────────────────────────────────────────────
class DPGradientClipper:
    """
    Per-sample gradient clipper and noise adder (manual DP-SGD).

    Standard DP-SGD (Abadi et al., 2016):
      1. Compute per-sample gradients.
      2. Clip each per-sample gradient to L2 norm ≤ max_grad_norm.
      3. Average clipped gradients.
      4. Add Gaussian noise with std = noise_multiplier * max_grad_norm.

    NOTE: True per-sample gradient computation requires hooks or Opacus.
          This simplified version clips the *batch-averaged* gradient —
          suitable for demonstrations and when batch size == 1.
    """

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        self.model            = model
        self.max_grad_norm    = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def clip_and_noise(self) -> None:
        """
        Call after loss.backward() and before optimizer.step().
        Clips gradient norms and adds Gaussian noise.
        """
        # Step 1 — clip
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Step 2 — add noise
        sigma = self.noise_multiplier * self.max_grad_norm
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad += torch.randn_like(param.grad) * sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3. Privacy budget accounting (moments accountant approximation)
# ─────────────────────────────────────────────────────────────────────────────
def compute_epsilon(
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float = 1e-5,
) -> float:
    """
    Estimate (epsilon, delta)-DP guarantee using the Gaussian mechanism
    with the simplified moments accountant formula.

    Args:
        steps:            total number of gradient update steps
        noise_multiplier: sigma / sensitivity ratio
        sample_rate:      batch_size / dataset_size
        delta:            target delta (privacy failure probability)

    Returns:
        epsilon: privacy budget consumed

    Reference: https://arxiv.org/abs/1607.00133
    """
    if noise_multiplier <= 0 or sample_rate <= 0:
        return float("inf")

    # Simplified Gaussian DP bound
    # epsilon ≈ sqrt(2 * log(1.25/delta)) / sigma  * sqrt(steps) * sample_rate
    sigma = noise_multiplier
    eps   = (
        math.sqrt(2 * math.log(1.25 / delta))
        / sigma
        * math.sqrt(steps)
        * sample_rate
    )
    return eps


# ─────────────────────────────────────────────────────────────────────────────
# 4. High-level DP trainer
# ─────────────────────────────────────────────────────────────────────────────
class DifferentialPrivacyTrainer:
    """
    Wraps a hospital node's training loop with differential privacy.

    Supports two backends:
      - "manual"  : simple gradient clipping + noise (no extra deps)
      - "opacus"  : production-grade Opacus library (pip install opacus)

    Example::

        dp_trainer = DifferentialPrivacyTrainer(
            model            = hospital_model,
            optimizer        = optimizer,
            dataloader       = train_loader,
            max_grad_norm    = 1.0,
            noise_multiplier = 1.1,
            backend          = "manual",
        )
        for epoch in range(num_epochs):
            metrics = dp_trainer.train_epoch(criterion)
            print(f"epsilon = {dp_trainer.epsilon:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        delta: float = 1e-5,
        backend: str = "manual",
        device: Optional[str] = None,
    ):
        self.model            = model
        self.optimizer        = optimizer
        self.dataloader       = dataloader
        self.max_grad_norm    = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.delta            = delta
        self.device           = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._steps = 0
        self._n     = len(dataloader.dataset)
        self._bs    = dataloader.batch_size or 32
        self.sample_rate = self._bs / self._n

        if backend == "opacus":
            self._init_opacus()
            self._backend = "opacus"
        else:
            self._clipper = DPGradientClipper(model, max_grad_norm, noise_multiplier)
            self._backend = "manual"

        print(
            f"[DP] Backend={self._backend} | "
            f"max_grad_norm={max_grad_norm} | noise_multiplier={noise_multiplier}"
        )

    def _init_opacus(self) -> None:
        """Wrap model and optimiser with Opacus PrivacyEngine."""
        try:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()
            (
                self.model,
                self.optimizer,
                self.dataloader,
            ) = privacy_engine.make_private(
                module       = self.model,
                optimizer    = self.optimizer,
                data_loader  = self.dataloader,
                noise_multiplier  = self.noise_multiplier,
                max_grad_norm     = self.max_grad_norm,
            )
            self._privacy_engine = privacy_engine
            print("[DP] Opacus PrivacyEngine attached successfully.")
        except ImportError:
            print("[DP] WARNING: Opacus not installed, falling back to manual backend.")
            self._clipper = DPGradientClipper(
                self.model, self.max_grad_norm, self.noise_multiplier
            )
            self._backend = "manual"

    @property
    def epsilon(self) -> float:
        """Current privacy budget consumed."""
        if self._backend == "opacus" and hasattr(self, "_privacy_engine"):
            return self._privacy_engine.get_epsilon(self.delta)
        return compute_epsilon(
            self._steps, self.noise_multiplier, self.sample_rate, self.delta
        )

    def train_epoch(self, criterion: nn.Module) -> Dict[str, float]:
        """
        Run one epoch of DP training.

        Returns:
            {"loss": float, "epsilon": float, "delta": float}
        """
        self.model.train()
        self.model.to(self.device)
        total_loss = 0.0

        for images, labels in self.dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            if self._backend == "manual":
                self._clipper.clip_and_noise()

            self.optimizer.step()
            total_loss += loss.item()
            self._steps += 1

        avg_loss = total_loss / max(len(self.dataloader), 1)
        return {
            "loss":    avg_loss,
            "epsilon": self.epsilon,
            "delta":   self.delta,
        }

    def get_dp_summary(self) -> str:
        """Human-readable summary of the current DP guarantee."""
        eps = self.epsilon
        return (
            f"DP Guarantee: (ε={eps:.4f}, δ={self.delta:.1e})\n"
            f"  Steps completed : {self._steps}\n"
            f"  Noise multiplier: {self.noise_multiplier}\n"
            f"  Max grad norm   : {self.max_grad_norm}\n"
            f"  Sample rate     : {self.sample_rate:.4f}\n"
            f"  Interpretation  : {'Strong' if eps < 1 else 'Moderate' if eps < 10 else 'Weak'} privacy"
        )
