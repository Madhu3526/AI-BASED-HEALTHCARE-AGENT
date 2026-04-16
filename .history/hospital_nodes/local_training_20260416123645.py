
import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import (
    ChestXrayDataset, get_dataloader, DISEASE_LABELS,
    DEFAULT_CSV, DEFAULT_IMG_DIRS,
)


# ── Metrics helpers ────────────────────────────────────────────────────────────
def _compute_metrics(
    all_probs: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute accuracy, F1, and per-class ROC-AUC."""
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    probs  = all_probs.cpu().numpy()
    labels = all_labels.cpu().numpy()
    preds  = (probs >= threshold).astype(int)

    acc = accuracy_score(labels.flatten(), preds.flatten())
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = 0.0  # When only one class present in a small batch

    return {"accuracy": acc, "f1_macro": f1, "roc_auc_macro": auc}


# ── Hospital Node ──────────────────────────────────────────────────────────────
class HospitalNode:
    """
    Represents a single hospital participating in federated learning.

    Each node:
      - holds its own local DataLoader
      - maintains a local copy of the global model
      - trains for `local_epochs` and returns updated weights
    """

    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        csv_path: Optional[str] = None,
        image_dirs: Optional[List[str]] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_workers: int = 0,            # 0 = main process (safe on Windows)
        subset_fraction: float = 1.0,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.hospital_id   = hospital_id
        self.learning_rate = learning_rate
        self.device        = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Use defaults if not provided
        csv_path   = csv_path   or str(DEFAULT_CSV)
        image_dirs = image_dirs or DEFAULT_IMG_DIRS

        # ── Local dataset ─────────────────────────────────────────────────
        self.train_loader, self.train_dataset = get_dataloader(
            csv_path         = csv_path,
            image_dirs       = image_dirs,
            split            = "train",
            batch_size       = batch_size,
            num_workers      = num_workers,
            subset_fraction  = subset_fraction,
            seed             = seed,
        )
        self.val_loader, _ = get_dataloader(
            csv_path         = csv_path,
            image_dirs       = image_dirs,
            split            = "val",
            batch_size       = batch_size,
            num_workers      = num_workers,
            subset_fraction  = subset_fraction,
            seed             = seed,
        )

        # ── Local copy of the global model ────────────────────────────────
        self.model = copy.deepcopy(model).to(self.device)

        # ── Positive class weights for imbalanced labels ──────────────────
        pos_weights = self.train_dataset.get_pos_weights().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        print(
            f"[{self.hospital_id}] Device={self.device} | "
            f"Train={len(self.train_dataset)} | Val={len(self.val_loader.dataset)}"
        )

    # ── Update local model with global weights ─────────────────────────────────
    def set_weights(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Load global model weights into this node's local model."""
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of this node's current model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def get_num_samples(self) -> int:
        """Return the number of local training samples (used for FedAvg weighting)."""
        return len(self.train_dataset)

    # ── Local training ─────────────────────────────────────────────────────────
    def train_local(
        self,
        local_epochs: int = 3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run local_epochs of SGD on this hospital's data.

        Returns:
            metrics dict with final validation metrics.
        """
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=local_epochs, eta_min=1e-6)

        for epoch in range(1, local_epochs + 1):
            train_loss = self._train_epoch(optimizer, verbose, epoch, local_epochs)
            scheduler.step()

        val_metrics = self._validate()
        if verbose:
            print(
                f"[{self.hospital_id}] Local training done — "
                f"val_loss={val_metrics['val_loss']:.4f}  "
                f"auc={val_metrics['roc_auc_macro']:.4f}  "
                f"f1={val_metrics['f1_macro']:.4f}"
            )
        return val_metrics

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        verbose: bool,
        epoch: int,
        total_epochs: int,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        loader = tqdm(
            self.train_loader,
            desc=f"[{self.hospital_id}] Epoch {epoch}/{total_epochs}",
            leave=False,
            disable=not verbose,
        )
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            # Gradient clipping to stabilise training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / max(len(self.train_loader), 1)

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss  = 0.0
        all_probs   = []
        all_labels  = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                total_loss += loss.item()

                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        all_probs  = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics    = _compute_metrics(all_probs, all_labels)
        metrics["val_loss"] = total_loss / max(len(self.val_loader), 1)
        return metrics


# ── Convenience function used by federated_server ─────────────────────────────
def train_local_model(
    node: HospitalNode,
    global_weights: Dict[str, torch.Tensor],
    local_epochs: int = 3,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
    """
    Sync global weights → train locally → return updated weights.

    Returns:
        (updated_weights, num_samples, val_metrics)
    """
    node.set_weights(global_weights)
    metrics       = node.train_local(local_epochs=local_epochs, verbose=verbose)
    updated_weights = node.get_weights()
    return updated_weights, node.get_num_samples(), metrics


# ── Standalone script ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from models import build_resnet

    parser = argparse.ArgumentParser(description="Train a single hospital node")
    parser.add_argument("--csv",        default=None,
                        help="CSV path (defaults to data/nih_xray/multilabel_dataset.csv)")
    parser.add_argument("--image-dirs", nargs="+", default=None,
                        help="Image directories (defaults to images_1, images_2)")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save-path",  default="models/hospital_local.pth")
    args = parser.parse_args()

    global_model = build_resnet(num_classes=14, pretrained=True)
    node = HospitalNode(
        hospital_id   = "Hospital_A",
        model         = global_model,
        csv_path      = args.csv,       # None → auto-detected
        image_dirs    = args.image_dirs, # None → auto-detected
        batch_size    = args.batch_size,
        learning_rate = args.lr,
        num_workers   = 0,
    )

    weights, n_samples, metrics = train_local_model(
        node           = node,
        global_weights = global_model.state_dict(),
        local_epochs   = args.epochs,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"model_state_dict": weights, "metrics": metrics}, args.save_path)
    print(f"Saved to {args.save_path}")
    print(f"Final metrics: {metrics}")
