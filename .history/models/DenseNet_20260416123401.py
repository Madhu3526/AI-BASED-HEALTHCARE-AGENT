import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """
    Channel attention mechanism.
    Helps the model focus on important X-ray features.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape

        weights = self.fc(x)
        return x * weights


# --------------------------------
# Improved X-ray Model
# --------------------------------
class DenseNetXray(nn.Module):

    def __init__(
        self,
        num_classes=14,
        pretrained=True,
        dropout_p=0.4,
        freeze_backbone=False
    ):
        super().__init__()

        # DenseNet backbone (strong for X-ray tasks)
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.classifier.in_features  # 1024

        # remove original classifier
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        # attention layer
        self.attention = SEBlock(in_features)

        # improved classifier head
        self.classifier = nn.Sequential(

            nn.BatchNorm1d(in_features),

            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_p),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_p/2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = self.backbone(x)  # (B,1024)

        features = self.attention(features)

        logits = self.classifier(features)

        return logits


    def predict_proba(self, x):

        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# --------------------------------
# Builder function
# --------------------------------
def build_xray_model(
    num_classes=14,
    pretrained=True,
    dropout_p=0.4,
    freeze_backbone=False
):

    model = DenseNetXray(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
        freeze_backbone=freeze_backbone
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    return model