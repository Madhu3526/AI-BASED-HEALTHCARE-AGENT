from typing import Optional

import torch
import torch.nn as nn

from data.dataset import DISEASE_LABELS

NUM_CLASSES = len(DISEASE_LABELS)


class TextEncoder(nn.Module):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for multi-modal fusion. "
                "Install with: pip install sentence-transformers"
            )
        self._st    = SentenceTransformer(model_name)
        self.embed_dim = self._st.get_sentence_embedding_dimension()

        # Freeze encoder weights — we only train the projection layers
        for param in self._st.parameters():
            param.requires_grad = False

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: list of strings, length B

        Returns:
            Tensor of shape (B, embed_dim)
        """
        embeddings = self._st.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings.float()


class MultiModalFusionModel(nn.Module):
    """
    Multi-modal classifier fusing X-ray image and clinical note features.

    Args:
        image_backbone:  a trained CNN (ResNetChestXray / DenseNetChestXray)
                         with its original classification head replaced by Identity.
                         Expected to output a flat feature vector of `image_dim`.
        image_dim:       dimensionality of image backbone output (2048 for ResNet-50,
                         1024 for DenseNet-121)
        text_dim:        dimensionality of text encoder output (384 for MiniLM)
        proj_dim:        shared projection dimension for both modalities (512)
        dropout_p:       dropout in the fusion MLP
        freeze_backbone: if True, image backbone weights are frozen
        text_model:      sentence-transformer model name
    """

    def __init__(
        self,
        image_backbone: nn.Module,
        image_dim:       int           = 2048,
        text_dim:        int           = 384,
        proj_dim:        int           = 512,
        dropout_p:       float         = 0.3,
        freeze_backbone: bool          = True,
        text_model:      str           = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        self.image_backbone = image_backbone
        if freeze_backbone:
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        self.text_encoder = TextEncoder(text_model)

        # ── Projection layers — map both modalities to a shared space ──────
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        # ── Fusion MLP ─────────────────────────────────────────────────────
        fused_dim = proj_dim * 2   # concat of image + text projections
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p / 2),
            nn.Linear(256, NUM_CLASSES),
        )

        # ── Modality gate: learn how much to weight image vs text ──────────
        self.modality_gate = nn.Sequential(
            nn.Linear(fused_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        image_tensor: torch.Tensor,
        notes_texts:  list,
    ) -> torch.Tensor:
        """
        Args:
            image_tensor: (B, 3, H, W) — preprocessed X-ray
            notes_texts:  list of B strings (empty string = no notes)

        Returns:
            logits: (B, 14)
        """
        # ── Image branch ───────────────────────────────────────────────────
        img_features  = self.image_backbone(image_tensor)   # (B, image_dim)
        img_projected = self.image_proj(img_features)       # (B, proj_dim)

        # ── Text branch ────────────────────────────────────────────────────
        device = image_tensor.device
        txt_embeddings = self.text_encoder(notes_texts).to(device)  # (B, text_dim)
        txt_projected  = self.text_proj(txt_embeddings)             # (B, proj_dim)

        # ── Gated fusion ───────────────────────────────────────────────────
        fused = torch.cat([img_projected, txt_projected], dim=-1)  # (B, proj_dim*2)
        gates = self.modality_gate(fused)                           # (B, 2)

        # Weighted sum: gate[0]*image + gate[1]*text (per sample)
        gated = (
            gates[:, 0:1] * img_projected
            + gates[:, 1:2] * txt_projected
        )                                                            # (B, proj_dim)

        # Concatenate gated-weighted + raw fused for richer representation
        final = torch.cat([gated, fused], dim=-1)                   # (B, proj_dim*3)

        # Resize to match fusion_head input
        # Note: fusion_head expects fused_dim = proj_dim * 2, so we use fused directly
        logits = self.fusion_head(fused)                             # (B, 14)
        return logits

    def predict_proba(
        self,
        image_tensor: torch.Tensor,
        notes_texts:  list,
    ) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(image_tensor, notes_texts))


def build_fusion_model(
    backbone_type:   str           = "resnet",
    freeze_backbone: bool          = True,
    dropout_p:       float         = 0.3,
    checkpoint_path: Optional[str] = None,
) -> MultiModalFusionModel:
    """
    Factory function — builds the fusion model with the chosen backbone.

    Args:
        backbone_type:   "resnet" (image_dim=2048) or "densenet" (image_dim=1024)
        freeze_backbone: keep backbone weights frozen during fusion training
        checkpoint_path: optional path to a saved fusion model state_dict

    Returns:
        MultiModalFusionModel ready for training or inference
    """
    if backbone_type == "densenet":
        from models.densenet_model import build_densenet
        base   = build_densenet(num_classes=14, pretrained=True)
        # Strip the classification head — leave backbone + attention
        base.head = nn.Identity()
        image_dim = 1024
    else:
        from models.resnet_model import build_resnet
        base   = build_resnet(num_classes=14, pretrained=True)
        base.head = nn.Identity()
        image_dim = 2048

    model = MultiModalFusionModel(
        image_backbone  = base,
        image_dim       = image_dim,
        freeze_backbone = freeze_backbone,
        dropout_p       = dropout_p,
    )

    if checkpoint_path:
        import os
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            print(f"[FusionModel] Loaded weights from {checkpoint_path}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FusionModel] Parameters — total: {total:,}  trainable: {trainable:,}")
    return model
