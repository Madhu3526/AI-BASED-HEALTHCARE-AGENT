import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetChestXray(nn.Module):

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_p: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 2048)
        logits   = self.head(features) # (B, 14)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_feature_extractor(self):
        """Return the backbone (useful for GradCAM hook registration)."""
        return self.backbone

    def unfreeze_layers(self, layer_names: Optional[list] = None) -> None:
        """
        Unfreeze specific backbone layers for fine-tuning.
        If layer_names is None, unfreezes all layers.
        """
        if layer_names is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for name, module in self.backbone.named_modules():
                if any(ln in name for ln in layer_names):
                    for param in module.parameters():
                        param.requires_grad = True


def build_resnet(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout_p: float = 0.5,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
) -> ResNetChestXray:
    """
    Factory function: build and optionally load weights.

    Args:
        checkpoint_path: path to a saved state_dict (.pth file).
    """
    model = ResNetChestXray(
        num_classes      = num_classes,
        pretrained       = pretrained,
        dropout_p        = dropout_p,
        freeze_backbone  = freeze_backbone,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        # Support both raw state_dict and wrapped checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[ResNet] Loaded weights from {checkpoint_path}")

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ResNet] Parameters — total: {total:,}  trainable: {trainable:,}")
    return model
