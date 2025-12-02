"""
Small script to run the full PyTorch training pipeline.

We follow the same steps as the original TensorFlow code:
  1) Baseline training on labeled data
  2) Pseudo-label generation
  3) Joint training (segmentation + contrastive)
  4) Final evaluation on the validation set
"""

import sys
import argparse
import os
import torch
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))

from Our_pytorch_version.trainers.contrastive_trainer import ContrastiveTrainer


def main():
    parser = argparse.ArgumentParser(description="Run complete training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="Our_pytorch_version/configs/config_chaos_8.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs_baseline",
        type=int,
        default=None,
        help="Override baseline epochs (optional)",
    )
    parser.add_argument(
        "--epochs_joint",
        type=int,
        default=None,
        help="Override joint epochs (optional)",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline phase (if pre-trained model exists)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE START")
    print("=" * 80)
    print(
        f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}"
    )
    print(f"Config: {args.config}")
    print("=" * 80)


    trainer = ContrastiveTrainer(args.config)

    if args.epochs_baseline is not None:
        trainer.config["training"]["baseline_epochs"] = args.epochs_baseline
    if args.epochs_joint is not None:
        trainer.config["training"]["joint_epochs"] = args.epochs_joint

    trainer.prepare_data()

    # Quick check of input shapes (we just print one batch)
    for imgs, _ in trainer.train_loader_labeled:
        print(imgs.shape)
        break


    # Phase 1: Baseline (pure supervised segmentation)

    if not args.skip_baseline:
        print("\n" + "=" * 80)
        print(" BASELINE TRAINING")
        print("=" * 80)
        trainer.train_baseline()
    else:
        print("\n  Baseline skipped (existing pretrained model assumed).")
        trainer.load_model("baseline_best.pth")


    # Phase 2: Pseudo-label generation
    print("\n" + "=" * 80)
    print(" GÉNÉRATION DES PSEUDO-LABELS")
    print("=" * 80)
    trainer.generate_pseudo_labels()


    # Phase 3: Joint training (segmentation + contrastive loss)

    print("\n" + "=" * 80)
    print("  JOINT TRAINING (seg + contrastive)")
    print("=" * 80)
    trainer.train_joint()   


    # Phase 4: Final validation

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    dice = trainer.validate()
    print(f"Final Dice score: {dice:.4f}")


if __name__ == "__main__":
    main()