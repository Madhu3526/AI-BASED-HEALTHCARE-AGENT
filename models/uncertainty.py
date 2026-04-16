from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from data.dataset import DISEASE_LABELS


def mc_dropout_predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    n_passes: int = 30,
    device: str = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    # Put model in eval mode first (keeps BatchNorm using running stats),
    # then selectively enable only Dropout layers so MC sampling works
    # with batch size = 1 without triggering BatchNorm's batch-size check.
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    all_probs: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor)
            probs  = torch.sigmoid(logits).squeeze().cpu()
            all_probs.append(probs)

    model.eval()            # restore all layers to eval

    stacked = torch.stack(all_probs)   # (n_passes, 14)
    mean    = stacked.mean(dim=0)
    std     = stacked.std(dim=0)

    mean_probs = {label: float(mean[i]) for i, label in enumerate(DISEASE_LABELS)}
    std_probs  = {label: float(std[i])  for i, label in enumerate(DISEASE_LABELS)}

    return mean_probs, std_probs


def uncertainty_level(std: float) -> str:
    """Classify a single std value into a human-readable confidence tier."""
    if std < 0.05:
        return "low"
    if std < 0.12:
        return "moderate"
    return "high"


def flag_uncertain_predictions(
    mean_probs: Dict[str, float],
    std_probs: Dict[str, float],
    threshold: float = 0.45,
    boundary_margin: float = 0.15,
    uncertainty_cutoff: float = 0.08,
) -> List[str]:
    """
    Return diseases where the model is uncertain near the decision boundary.
    These should be highlighted for radiologist review.

    A flag is raised when:
      |mean_prob - threshold| < boundary_margin   (close to the decision line)
      AND std > uncertainty_cutoff                  (high variance across passes)
    """
    flags: List[str] = []
    for disease in DISEASE_LABELS:
        prob = mean_probs[disease]
        std  = std_probs[disease]
        if abs(prob - threshold) < boundary_margin and std > uncertainty_cutoff:
            flags.append(
                f"{disease}: {prob*100:.1f}% ± {std*100:.1f}pp  "
                f"[{uncertainty_level(std)} uncertainty — review recommended]"
            )
    return flags


def build_uncertainty_summary(
    mean_probs: Dict[str, float],
    std_probs: Dict[str, float],
) -> List[Dict]:
    """
    Build a list of dicts suitable for display in a DataFrame or table.

    Each row:
        disease, mean_prob, std, uncertainty_level
    Sorted by std descending so the most uncertain diseases appear first.
    """
    rows = [
        {
            "Disease":       disease,
            "Mean Prob (%)": round(mean_probs[disease] * 100, 2),
            "Std Dev (pp)":  round(std_probs[disease] * 100, 2),
            "Uncertainty":   uncertainty_level(std_probs[disease]),
        }
        for disease in DISEASE_LABELS
    ]
    return sorted(rows, key=lambda r: r["Std Dev (pp)"], reverse=True)
