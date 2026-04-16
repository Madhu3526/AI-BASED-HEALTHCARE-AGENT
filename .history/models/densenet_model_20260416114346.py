import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SEBlock(nn.Module):no e

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class DenseNetChestXray(nn.Module):
    """
    DenseNet-121 backbone for multi-label chest X-ray classification
    with an SE attention block and custom classifier head.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_p: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.attention = SEBlock(in_features)
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.attention(features)
        logits = self.head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_feature_extractor(self):
        return self.backbone


def build_densenet(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout_p: float = 0.4,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
) -> DenseNetChestXray:
    model = DenseNetChestXray(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[DenseNet] Loaded weights from {checkpoint_path}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DenseNet] Parameters - total: {total:,}  trainable: {trainable:,}")
    return model
