from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from data.dataset import DISEASE_LABELS
#softvoti

class EnsembleModel:

    def __init__(
        self,
        models:  List[nn.Module],
        weights: Optional[List[float]] = None,
        names:   Optional[List[str]]   = None,
        device:  Optional[str]         = None,
    ):
        assert len(models) > 0, "Need at least one model."
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.models  = models
        self.names   = names or [f"Model_{i}" for i in range(len(models))]
        self.device  = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        for m in self.models:
            m.to(self.device).eval()

    def predict(
        self,
        image_tensor: torch.Tensor,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Run inference with all models and return weighted-average probabilities.

        Args:
            image_tensor: preprocessed X-ray (1, 3, H, W) or (3, H, W)

        Returns:
            ensemble_probs:    {disease: weighted_mean_probability}
            individual_probs:  list of {disease: probability} per model
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        individual: List[Dict[str, float]] = []
        ensemble_tensor = torch.zeros(len(DISEASE_LABELS))

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                logits = model(image_tensor)
                probs  = torch.sigmoid(logits).squeeze().cpu()
                individual.append(
                    {label: float(probs[i]) for i, label in enumerate(DISEASE_LABELS)}
                )
                ensemble_tensor += weight * probs

        ensemble_probs = {
            label: float(ensemble_tensor[i])
            for i, label in enumerate(DISEASE_LABELS)
        }
        return ensemble_probs, individual

    def model_agreement(
        self,
        individual_probs: List[Dict[str, float]],
        threshold: float = 0.45,
    ) -> Dict[str, Dict]:
        """
        For each disease, report whether the models agree on the binary decision.

        Returns a dict keyed by disease with:
          - votes:     how many models predict positive
          - agreement: True/False
          - decisions: list of per-model binary decisions
        """
        agreement: Dict[str, Dict] = {}
        for disease in DISEASE_LABELS:
            decisions = [p[disease] >= threshold for p in individual_probs]
            votes = sum(decisions)
            agreement[disease] = {
                "votes":     votes,
                "total":     len(self.models),
                "agreement": votes == 0 or votes == len(self.models),
                "decisions": [
                    {"model": name, "positive": dec, "prob": round(p[disease], 4)}
                    for name, dec, p in zip(self.names, decisions, individual_probs)
                ],
            }
        return agreement

    def disagreement_flags(
        self,
        individual_probs: List[Dict[str, float]],
        threshold: float = 0.45,
    ) -> List[str]:
        """
        Return diseases where models disagree on the binary outcome.
        These should be flagged for radiologist review.
        """
        flags: List[str] = []
        agreement = self.model_agreement(individual_probs, threshold)
        for disease, info in agreement.items():
            if not info["agreement"]:
                detail = ", ".join(
                    f"{d['model']}={'✓' if d['positive'] else '✗'} ({d['prob']*100:.1f}%)"
                    for d in info["decisions"]
                )
                flags.append(f"{disease}: models disagree — {detail}")
        return flags


def load_ensemble(
    resnet_ckpt:   Optional[str] = None,
    densenet_ckpt: Optional[str] = None,
    vit_ckpt:      Optional[str] = None,
    weights:       Optional[List[float]] = None,
    device:        Optional[str] = None,
) -> EnsembleModel:
    """
    Convenience factory: build and load checkpoints for all three model types.

    Any checkpoint path that is None / missing will be replaced by a
    freshly-initialised (ImageNet pretrained) model.
    """
    from models.resnet_model   import build_resnet
    from models.densenet_model import build_densenet
    from models.vit_model      import build_vit

    models_list: List[nn.Module] = []
    names: List[str] = []

    resnet = build_resnet(num_classes=14, pretrained=True, checkpoint_path=resnet_ckpt)
    models_list.append(resnet)
    names.append("ResNet-50")

    densenet = build_densenet(num_classes=14, pretrained=True, checkpoint_path=densenet_ckpt)
    models_list.append(densenet)
    names.append("DenseNet-121")

    if vit_ckpt:
        vit = build_vit(num_classes=14, pretrained=True, checkpoint_path=vit_ckpt)
        models_list.append(vit)
        names.append("ViT-Base/16")

    return EnsembleModel(models=models_list, weights=weights, names=names, device=device)
