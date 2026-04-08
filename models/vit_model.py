"""
Vision Transformer (ViT-Base/16) for Multi-label Chest X-ray Classification.
Built with the `timm` library for easy pretrained weight access.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class ViTChestXray(nn.Module):
    """
    ViT-Base/16 backbone from timm with a custom multi-label head.

    Architecture:
        vit_base_patch16_224 (ImageNet-21k pretrained by default)
        └─ CLS token embedding (768-d)
        └─ Dropout(p)
        └─ Linear(768 → num_classes)
        [No sigmoid — use BCEWithLogitsLoss during training]
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_p: float = 0.3,
        model_name: str = "vit_base_patch16_224",
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Load ViT backbone via timm, with original head stripped ───────
        self.backbone = timm.create_model(
            model_name,
            pretrained   = pretrained,
            num_classes  = 0,     # remove default head → outputs raw CLS features
            drop_rate    = dropout_p,
        )

        # The CLS feature dimension varies by model variant
        embed_dim = self.backbone.embed_dim  # 768 for ViT-Base

        # ── Custom classification head ─────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone(x)   # (B, embed_dim)
        logits   = self.head(features) # (B, 14)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only train the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        """
        Unfreeze only the last n transformer blocks (common fine-tuning strategy).
        """
        # Freeze everything first
        self.freeze_backbone()
        # Then selectively unfreeze last n blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Always keep the head unfrozen
        for param in self.head.parameters():
            param.requires_grad = True


def build_vit(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout_p: float = 0.3,
    model_name: str = "vit_base_patch16_224",
    checkpoint_path: Optional[str] = None,
    unfreeze_last_n: Optional[int] = None,
) -> ViTChestXray:
    """
    Factory function: build and optionally load weights.

    Args:
        model_name:      any timm ViT variant (e.g. 'vit_large_patch16_224')
        unfreeze_last_n: if set, freeze backbone and only unfreeze last n blocks
        checkpoint_path: path to a saved state_dict (.pth file)
    """
    model = ViTChestXray(
        num_classes = num_classes,
        pretrained  = pretrained,
        dropout_p   = dropout_p,
        model_name  = model_name,
    )

    if unfreeze_last_n is not None:
        model.unfreeze_last_n_blocks(unfreeze_last_n)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[ViT] Loaded weights from {checkpoint_path}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ViT] Parameters — total: {total:,}  trainable: {trainable:,}")
    return model
