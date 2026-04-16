import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ── Core aggregation ───────────────────────────────────────────────────────────
def federated_average(
    weights_list: List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of model state_dicts.

    Args:
        weights_list:  list of state_dicts from each hospital node
        sample_counts: number of training samples each node used

    Returns:
        aggregated state_dict (CPU tensors)
    """
    assert len(weights_list) == len(sample_counts), \
        "weights_list and sample_counts must have the same length."
    assert len(weights_list) > 0, "Need at least one set of weights."

    total_samples = sum(sample_counts)
    fractions     = [n / total_samples for n in sample_counts]

    # Initialise accumulator with zeros matching first state_dict
    agg = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in weights_list[0].items()
    }

    for weights, frac in zip(weights_list, fractions):
        for key in agg:
            agg[key] += frac * weights[key].float()

    return agg


# ── Federated Server ───────────────────────────────────────────────────────────
class FederatedServer:
    """
    Central coordinator for federated learning.

    Responsibilities:
      1. Maintain the current global model.
      2. Broadcast global weights to hospital nodes.
      3. Collect local weights after each round.
      4. Aggregate via FedAvg → update global model.
      5. Optionally save checkpoints.

    Usage::

        server = FederatedServer(global_model)

        for round_num in range(num_rounds):
            # Each hospital trains locally and returns weights
            updates = []
            for node in hospital_nodes:
                w, n, metrics = train_local_model(
                    node, server.get_global_weights(), local_epochs=3
                )
                updates.append((w, n, metrics))

            server.aggregate(updates)
            server.save_checkpoint(f"checkpoints/round_{round_num}.pth")
    """

    def __init__(
        self,
        global_model: nn.Module,
        save_dir: str = "models/federated",
    ):
        self.global_model = global_model
        self.save_dir     = save_dir
        self.round_num    = 0
        self.history: List[Dict] = []   # tracks metrics per round
        os.makedirs(save_dir, exist_ok=True)
        print(f"[FederatedServer] Initialised. Save dir: {save_dir}")

    # ── Weight access ──────────────────────────────────────────────────────────
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return CPU copy of global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def set_global_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated weights into the global model."""
        self.global_model.load_state_dict(state_dict)

    # ── Aggregation ────────────────────────────────────────────────────────────
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, torch.Tensor], int, Dict]],
        verbose: bool = True,
    ) -> None:
        """
        Perform one round of FedAvg.

        Args:
            updates: list of (state_dict, num_samples, metrics_dict) tuples,
                     one per participating hospital.
        """
        self.round_num += 1
        weights_list  = [w for w, _, _ in updates]
        sample_counts = [n for _, n, _ in updates]
        per_node_metrics = [m for _, _, m in updates]

        aggregated = federated_average(weights_list, sample_counts)
        self.set_global_weights(aggregated)

        # ── Record round summary ───────────────────────────────────────────
        avg_auc = sum(m.get("roc_auc_macro", 0) for m in per_node_metrics) / len(per_node_metrics)
        avg_f1  = sum(m.get("f1_macro", 0) for m in per_node_metrics) / len(per_node_metrics)
        avg_acc = sum(m.get("accuracy", 0) for m in per_node_metrics) / len(per_node_metrics)
        avg_val_loss = sum(m.get("val_loss", 0) for m in per_node_metrics) / len(per_node_metrics)
        round_summary = {
            "round":        self.round_num,
            "n_clients":    len(updates),
            "total_samples": sum(sample_counts),
            "avg_auc":       round(avg_auc, 4),
            "avg_f1":        round(avg_f1, 4),
            "avg_accuracy":  round(avg_acc, 4),
            "avg_val_loss":  round(avg_val_loss, 4),
            "per_hospital_metrics": [
                {
                    "hospital_id": m.get("hospital_id", f"Hospital_{idx + 1}"),
                    "roc_auc_macro": round(m.get("roc_auc_macro", 0), 4),
                    "f1_macro": round(m.get("f1_macro", 0), 4),
                    "accuracy": round(m.get("accuracy", 0), 4),
                    "val_loss": round(m.get("val_loss", 0), 4),
                    "num_samples": sample_counts[idx],
                }
                for idx, m in enumerate(per_node_metrics)
            ],
        }
        self.history.append(round_summary)

        if verbose:
            print(
                f"[FederatedServer] Round {self.round_num} complete | "
                f"clients={len(updates)} | "
                f"avg_auc={avg_auc:.4f} | avg_f1={avg_f1:.4f} | avg_acc={avg_acc:.4f}"
            )

    # ── Checkpointing ──────────────────────────────────────────────────────────
    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save global model weights and training history."""
        if filename is None:
            filename = f"global_round_{self.round_num:03d}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "round":            self.round_num,
                "model_state_dict": self.get_global_weights(),
                "history":          self.history,
            },
            path,
        )
        print(f"[FederatedServer] Saved checkpoint → {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume from a saved checkpoint."""
        ckpt            = torch.load(path, map_location="cpu")
        self.round_num  = ckpt.get("round", 0)
        self.history    = ckpt.get("history", [])
        self.set_global_weights(ckpt["model_state_dict"])
        print(f"[FederatedServer] Loaded checkpoint from {path} (round {self.round_num})")

    def print_history(self) -> None:
        print(f"\n{'Round':>6}  {'Clients':>8}  {'Samples':>8}  {'AUC':>8}  {'F1':>8}")
        print("-" * 50)
        for h in self.history:
            print(
                f"{h['round']:>6}  {h['n_clients']:>8}  "
                f"{h['total_samples']:>8}  {h['avg_auc']:>8.4f}  {h['avg_f1']:>8.4f}"
            )


# ── Standalone federated simulation script ────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from models import build_resnet
    from hospital_nodes.local_training import HospitalNode, train_local_model

    parser = argparse.ArgumentParser(description="Run a federated learning simulation")
    parser.add_argument("--csv",          required=True)
    parser.add_argument("--image-dirs",   required=True, nargs="+")
    parser.add_argument("--num-rounds",   type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--num-hospitals",type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--save-dir",     default="models/federated")
    args = parser.parse_args()

    # Build global model
    global_model = build_resnet(num_classes=14, pretrained=True)
    server       = FederatedServer(global_model, save_dir=args.save_dir)

    # Create hospital nodes (each gets a random subset of the data)
    nodes = []
    for i in range(args.num_hospitals):
        node = HospitalNode(
            hospital_id      = f"Hospital_{chr(65+i)}",
            csv_path         = args.csv,
            image_dirs       = args.image_dirs,
            model            = global_model,
            batch_size       = args.batch_size,
            subset_fraction  = 1.0 / args.num_hospitals,
            seed             = i * 100,
        )
        nodes.append(node)

    # Federated training loop
    for round_idx in range(args.num_rounds):
        global_weights = server.get_global_weights()
        updates = []

        for node in nodes:
            w, n, metrics = train_local_model(
                node, global_weights, local_epochs=args.local_epochs
            )
            updates.append((w, n, metrics))

        server.aggregate(updates)
        server.save_checkpoint()

    server.print_history()
    print("Federated training complete.")
