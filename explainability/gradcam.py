"""
Gradient-weighted Class Activation Mapping (GradCAM)
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017).

Generates heatmaps highlighting the lung regions that most influenced
the model's prediction for each disease class.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from PIL import Image


# ── GradCAM Core ──────────────────────────────────────────────────────────────
class GradCAM:
    """
    GradCAM for any CNN model using PyTorch forward/backward hooks.

    Registers hooks on a specified target layer, then:
      1. Forward pass → capture feature maps A.
      2. Backward pass for target class → capture gradients dL/dA.
      3. Weight = global-average-pool(gradients).
      4. CAM = ReLU(Σ weight_k * A_k).
      5. Resize CAM to input resolution and overlay on original image.

    Args:
        model:        trained CNN (ResNet50 etc.)
        target_layer: the conv layer to hook (e.g. model.backbone.layer4[-1])

    Usage::

        gradcam = GradCAM(model, model.backbone.layer4[-1])
        heatmap, overlay = gradcam(image_tensor, class_idx=6)  # 6 = Pneumonia
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ──────────────────────────────────────────────────────────────────
    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Clean up hooks when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── Core computation ───────────────────────────────────────────────────────
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GradCAM heatmap for the given input and class.

        Args:
            input_tensor: (1, 3, H, W) or (3, H, W) — already normalised
            class_idx:    target class index (0-13 for 14 diseases).
                          If None, uses the class with highest predicted probability.

        Returns:
            heatmap: (H, W) float array in [0, 1]
            overlay: (H, W, 3) uint8 RGB image with heatmap blended onto original
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device).requires_grad_(False)

        # ── Forward pass ───────────────────────────────────────────────────
        output = self.model(input_tensor)   # (1, num_classes)

        if class_idx is None:
            class_idx = output.squeeze().argmax().item()

        # ── Backward pass for selected class ──────────────────────────────
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # ── Compute weights (global average pooling of gradients) ──────────
        # gradients: (1, C, H_f, W_f)
        gradients   = self._gradients        # (1, C, H_f, W_f)
        activations = self._activations      # (1, C, H_f, W_f)

        weights     = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # ── Weighted combination of activations ────────────────────────────
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H_f, W_f)
        cam = F.relu(cam)                                        # only positive influence

        # ── Resize to input resolution ─────────────────────────────────────
        cam = F.interpolate(
            cam,
            size  = (input_tensor.shape[-2], input_tensor.shape[-1]),
            mode  = "bilinear",
            align_corners = False,
        )  # (1, 1, H, W)

        cam_np = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # ── Create coloured heatmap overlay ────────────────────────────────
        overlay = _make_overlay(input_tensor.squeeze(0), cam_np)

        return cam_np, overlay

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ── Overlay helper ─────────────────────────────────────────────────────────────
def _make_overlay(
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend heatmap onto the original image.

    Args:
        img_tensor: (3, H, W) normalised tensor
        heatmap:    (H, W) float in [0, 1]
        alpha:      weight of heatmap vs original
        colormap:   OpenCV colormap id

    Returns:
        overlay: (H, W, 3) uint8 RGB
    """
    # Denormalise tensor → uint8 RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = img_tensor.cpu() * std + mean
    img  = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Heatmap → coloured image
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)   # BGR

    # Blend
    blended = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    overlay = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return overlay


# ── Convenience function ───────────────────────────────────────────────────────
def generate_gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level entry point for GradCAM generation.

    Automatically selects the last conv layer from a ResNet50 backbone.

    Args:
        model:        ResNetChestXray or any CNN with .backbone attribute
        image_tensor: (1, 3, 224, 224) preprocessed tensor
        class_idx:    disease class index (0-13)
        save_path:    if given, saves the overlay as PNG

    Returns:
        (heatmap, overlay)
    """
    # Locate the last convolutional layer
    target_layer = _find_last_conv(model)

    gradcam = GradCAM(model, target_layer)
    heatmap, overlay = gradcam(image_tensor, class_idx)
    gradcam.remove_hooks()

    if save_path:
        Image.fromarray(overlay).save(save_path)
        print(f"[GradCAM] Saved overlay → {save_path}")

    return heatmap, overlay


def _find_last_conv(model: nn.Module) -> nn.Module:
    """
    Traverse model to find the last Conv2d layer.
    Works for ResNet (layer4[-1]) and other architectures.
    """
    # Try ResNet-specific path first
    try:
        return model.backbone.layer4[-1]
    except AttributeError:
        pass

    # Generic fallback: walk modules in reverse
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not find any Conv2d layer in the model.")
    return last_conv


# ── Disease-specific multi-class heatmap ──────────────────────────────────────
def generate_multi_class_heatmaps(
    model: nn.Module,
    image_tensor: torch.Tensor,
    disease_labels: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float, np.ndarray, np.ndarray]]:
    """
    Generate GradCAM overlays for the top-k predicted diseases.

    Args:
        model:          trained model
        image_tensor:   (1, 3, 224, 224)
        disease_labels: list of disease name strings (length = num_classes)
        top_k:          number of top diseases to explain

    Returns:
        list of (disease_name, probability, heatmap, overlay)
        sorted by probability descending.
    """
    model.eval()
    device = next(model.parameters()).device
    inp    = image_tensor.to(device)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(inp)).squeeze().cpu()

    top_indices = probs.topk(min(top_k, len(disease_labels))).indices.tolist()
    target_layer = _find_last_conv(model)

    results = []
    for idx in top_indices:
        gradcam          = GradCAM(model, target_layer)
        heatmap, overlay = gradcam(inp, class_idx=idx)
        gradcam.remove_hooks()
        results.append((disease_labels[idx], probs[idx].item(), heatmap, overlay))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
