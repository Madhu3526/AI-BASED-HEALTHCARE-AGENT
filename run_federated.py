"""
Federated Learning Training Runner
Orchestrates the full federated training pipeline:
  - Initialises a global model
  - Creates N hospital nodes
  - Runs R communication rounds
  - Saves global checkpoint after each round
  - Prints final performance summary

Usage:
    # Use auto-detected paths (recommended):
    python run_federated.py --num-hospitals 3 --num-rounds 10

    # Or specify paths explicitly:
    python run_federated.py \
        --csv data/nih_xray/multilabel_dataset.csv \
        --image-dirs data/nih_xray/images_1 data/nih_xray/images_2 \
        --num-hospitals 3 --num-rounds 10 --local-epochs 3 \
        --model resnet --use-dp
"""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.resnet_model import build_resnet
from models.densenet_model import build_densenet
from models.vit_model import build_vit
from data.dataset import DEFAULT_CSV, DEFAULT_IMG_DIRS
from hospital_nodes.local_training import HospitalNode, train_local_model
from federated_server.fedavg import FederatedServer
from privacy.differential_privacy import DifferentialPrivacyTrainer
import torch.nn as nn
from torch.optim import Adam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Medical AI — Training Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths default to the actual dataset layout
    parser.add_argument("--csv",
                        default=str(DEFAULT_CSV),
                        help="Path to multilabel_dataset.csv or Data_Entry_2017_v2020.csv")
    parser.add_argument("--image-dirs",
                        nargs="+",
                        default=DEFAULT_IMG_DIRS,
                        help="Flat image directories (images_1, images_2 ...)")
    parser.add_argument("--num-hospitals", type=int,   default=3,     help="Number of hospital nodes")
    parser.add_argument("--num-rounds",    type=int,   default=10,    help="Federated communication rounds (10+ recommended)")
    parser.add_argument("--local-epochs",  type=int,   default=5,     help="Local epochs per round (5 gives better convergence)")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--model",         choices=["resnet", "densenet", "vit"], default="resnet")
    parser.add_argument("--save-dir",      default="models/federated")
    parser.add_argument("--use-dp",        action="store_true", help="Enable differential privacy")
    parser.add_argument("--dp-noise",      type=float, default=1.1)
    parser.add_argument("--dp-clip",       type=float, default=1.0)
    parser.add_argument("--subset",        type=float, default=1.0,   help="Dataset fraction (0-1)")
    parser.add_argument("--num-workers",   type=int,   default=0,
                        help="DataLoader workers (0 = main process, safe on Windows)")
    return parser.parse_args()


def build_model(model_type: str) -> nn.Module:
    if model_type == "resnet":
        return build_resnet(num_classes=14, pretrained=True)
    if model_type == "densenet":
        return build_densenet(num_classes=14, pretrained=True)
    return build_vit(num_classes=14, pretrained=True, unfreeze_last_n=4)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Federated Medical AI — Training")
    print(f"{'='*60}")
    print(f"  Model       : {args.model}")
    print(f"  Hospitals   : {args.num_hospitals}")
    print(f"  FL rounds   : {args.num_rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Privacy (DP): {'YES' if args.use_dp else 'NO'}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    # ── Global model and server ────────────────────────────────────────────
    global_model = build_model(args.model)
    server       = FederatedServer(global_model, save_dir=args.save_dir)

    # ── Hospital nodes ─────────────────────────────────────────────────────
    # Each hospital gets the full --subset fraction of the dataset but with a
    # unique seed so its train/val shuffle differs, simulating data heterogeneity
    # without starving any node of training samples.
    nodes = []
    for i in range(args.num_hospitals):
        node = HospitalNode(
            hospital_id      = f"Hospital_{chr(65 + i)}",
            csv_path         = args.csv,
            image_dirs       = args.image_dirs,
            model            = global_model,
            batch_size       = args.batch_size,
            learning_rate    = args.lr,
            num_workers      = args.num_workers,
            subset_fraction  = args.subset,   # full fraction per hospital (not divided)
            device           = device,
            seed             = i * 100,       # different seed → different shuffle/split
        )
        nodes.append(node)

    # ── Federated training ─────────────────────────────────────────────────
    for round_idx in range(1, args.num_rounds + 1):
        print(f"\n{'─'*50}")
        print(f"  Round {round_idx}/{args.num_rounds}")
        print(f"{'─'*50}")

        global_weights = server.get_global_weights()
        updates = []

        for node in nodes:
            if args.use_dp:
                # Apply DP during local training
                node.set_weights(global_weights)
                node.model.to(device)
                optimizer    = Adam(node.model.parameters(), lr=args.lr)
                dp_criterion = nn.BCEWithLogitsLoss(
                    pos_weight=node.train_dataset.get_pos_weights().to(device)
                )
                dp_trainer = DifferentialPrivacyTrainer(
                    model            = node.model,
                    optimizer        = optimizer,
                    dataloader       = node.train_loader,
                    max_grad_norm    = args.dp_clip,
                    noise_multiplier = args.dp_noise,
                    backend          = "manual",
                    device           = device,
                )
                for _ in range(args.local_epochs):
                    dp_trainer.train_epoch(dp_criterion)

                metrics = node._validate()
                metrics["hospital_id"] = node.hospital_id
                print(f"  [{node.hospital_id}] DP: ε={dp_trainer.epsilon:.4f} "
                      f"| val_auc={metrics.get('roc_auc_macro', 0):.4f}")
                w = node.get_weights()
                updates.append((w, node.get_num_samples(), metrics))

            else:
                w, n, metrics = train_local_model(
                    node, global_weights, local_epochs=args.local_epochs
                )
                metrics["hospital_id"] = node.hospital_id
                updates.append((w, n, metrics))

        server.aggregate(updates)
        server.save_checkpoint()

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Federated Training Complete")
    print(f"{'='*60}")
    server.print_history()

    final_path = os.path.join(args.save_dir, "global_model_final.pth")
    server.save_checkpoint("global_model_final.pth")
    print(f"\nFinal model saved to: {final_path}")
    print("Launch dashboard: streamlit run dashboard/app.py")
    print(f"  -> Enter checkpoint path: {final_path}")


if __name__ == "__main__":
    main()
