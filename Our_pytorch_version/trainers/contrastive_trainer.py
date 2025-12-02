"""
Trainer we use to reproduce the pipeline from:
“Local Contrastive Loss with Pseudo-label based Self-training
for Semi-supervised Medical Image Segmentation”.

We keep the structure very close to the original paper.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models import ContrastiveUNet
from ..losses import DiceLoss, LocalContrastiveLoss
from ..data import MedicalImageDataset, SliceDataset, get_augmentations


#  Small helpers 

def collate_fn_with_none(batch):
    # Remove samples where the image is None
    batch = [(im, m) for im, m in batch if im is not None]
    if len(batch) == 0:
        return None, None
    imgs, masks = zip(*batch)
    imgs = torch.stack(imgs)
    # If at least one mask is None, we return masks=None for the whole batch
    if any(m is None for m in masks):
        masks = None
    else:
        masks = torch.stack(masks)
    return imgs, masks


#  Main trainer (supervised baseline + joint contrastive training)


class ContrastiveTrainer:
    def __init__(self, config_path):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.root_dir = Path(config_path).resolve().parent.parent.parent
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        print(f"Device = {self.device}")

        # Model
        self.model = ContrastiveUNet(
            in_channels=self.config["dataset"]["num_channels"],
            num_classes=self.config["dataset"]["num_classes"],
            base_channels=self.config["model"]["base_channels"],
            embed_dim=self.config["model"]["embed_dim"],
            interp_method=self.config["model"]["interp_method"],
        ).to(self.device)

        # segmentation loss (Dice)
        self.seg_loss_fn = DiceLoss(
            num_classes=self.config["dataset"]["num_classes"], with_background=True
        )

        # local contrastive loss
        self.cont_loss_fn = LocalContrastiveLoss(
            num_classes=self.config["dataset"]["num_classes"],
            temperature=self.config["training"]["temperature"],
            num_pos_samples=self.config["training"]["num_pos_samples"],
            inter_image=self.config["training"]["inter_image_matching"],
        )

        # optimizer for all model parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["joint_lr"]
        )

        # coefficients and thresholds
        self.lambda_cont = self.config["training"]["lambda_contrastive"]
        self.f1_threshold = self.config["training"].get("test_f1_threshold", 0.9)

        # saving directory and best validation score
        self.save_dir = self.config["paths"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_dice = 0.0


    #   DATA LOADING

    def prepare_data(self):
        cfg = self.config["dataset"]
        aug = self.config["augmentation"]

        # data augmentation on 2D slices
        transform = get_augmentations(
            aug_type=aug["type"], img_size=cfg["img_size"]
        )

        root = cfg["data_path_tr_cropped"]
        if not os.path.isabs(root):
            root = str(self.root_dir / root)

        # labeled 

        train_img = [
            f"{root}/patient{pid}/img_cropped.nii.gz"
            for pid in cfg["train_ids"]
        ]
        train_mask = [
            f"{root}/patient{pid}/mask_cropped.nii.gz"
            for pid in cfg["train_ids"]
        ]

        train_dataset = MedicalImageDataset(
            train_img, train_mask,
            target_resolution=cfg["target_resolution"],
            img_size=cfg["img_size"]
        )

        self.train_loader_labeled = DataLoader(
            SliceDataset(train_dataset, transform),
            batch_size=self.config["training"]["joint_batch_size_labeled"],
            shuffle=True,
            num_workers=0
        )

        # unlabeled

        unlabeled_img = [
            f"{root}/patient{uid}/img_cropped.nii.gz"
            for uid in cfg["unlabeled_ids"]
        ]
        unl_dataset = MedicalImageDataset(
            unlabeled_img,
            label_present=False,
            target_resolution=cfg["target_resolution"],
            img_size=cfg["img_size"],
        )

        self.unlabeled_loader = DataLoader(
            SliceDataset(unl_dataset, transform),
            batch_size=self.config["training"]["joint_batch_size_contrastive"],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_with_none
        )

        # validation

        val_img = [
            f"{root}/patient{vid}/img_cropped.nii.gz"
            for vid in cfg["val_ids"]
        ]
        val_mask = [
            f"{root}/patient{vid}/mask_cropped.nii.gz"
            for vid in cfg["val_ids"]
        ]
        val_dataset = MedicalImageDataset(
            val_img, val_mask,
            target_resolution=cfg["target_resolution"],
            img_size=cfg["img_size"],
        )

        self.val_loader = DataLoader(
            SliceDataset(val_dataset),
            batch_size=self.config["training"]["baseline_batch_size"],
            shuffle=False,
            num_workers=0
        )

        print(f"Labeled={len(train_dataset)}  |  Unlabeled={len(unl_dataset)}  |  Val={len(val_dataset)}")


    def _rand_intensity_aug(self, x):
        """
        Simple brightness/contrast jitter in PyTorch,
        close to what is used in the original TensorFlow code.
        """
        delta = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.20
        alpha = 0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1, device=x.device)
        return torch.clamp(alpha * x + delta, min=x.min(), max=x.max())


    #  BASELINE (supervised only)

    def train_baseline(self):
        print("\n")
        print("   Baseline supervised")
        print("\n")

        n_epochs = self.config["training"]["baseline_epochs"]

        for ep in range(n_epochs):
            self.model.train()
            total_loss = 0.0

            for imgs, masks in tqdm(self.train_loader_labeled):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)

                loss = self.seg_loss_fn(logits, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (ep + 1) % self.config["training"]["val_step_update"] == 0:
                val = self.validate()
                avg = total_loss / len(self.train_loader_labeled)
                print(f"Epoch {ep+1}/{n_epochs} | Loss={avg:.4f} | ValDice={val:.4f}")

                if val > self.best_val_dice:
                    self.best_val_dice = val
                    self.save_model("baseline_best.pth")

        self.load_model("baseline_best.pth")
        print(f"Baseline ready | Best Dice={self.best_val_dice:.4f}")


    #         PSEUDO-LABEL GENERATION (F1-threshold + CRF-like)

    def generate_pseudo_labels(self):
        print("\n Generating pseudo-labels")
        self.model.eval()
        # Tentative de refining par f1 score comme mentionné dans l'ablation study du papier, notamment dans le d) mais pas efficace expérimentatement
        # refined = 0
        # kept = 0
        # cfg = self.config["dataset"]

        # with torch.no_grad():
        #     for batch in tqdm(self.unlabeled_loader):
        #         if batch is None:
        #             continue

        #         imgs, _ = batch
        #         imgs = imgs.to(self.device)

        #         # two intensity-augmented views of the same unlabeled slice
        #         imgs1 = self._rand_intensity_aug(imgs)
        #         imgs2 = self._rand_intensity_aug(imgs)

        #         probs1 = F.softmax(self.model(imgs1), dim=1)
        #         probs2 = F.softmax(self.model(imgs2), dim=1)

        #         lbl1 = torch.argmax(probs1, dim=1)
        #         lbl2 = torch.argmax(probs2, dim=1)

        #         for i in range(lbl1.size(0)):
        #             f1_list = []
        #             for c in range(1, cfg["num_classes"]):
        #                 p = (lbl1[i] == c).float()
        #                 t = (lbl2[i] == c).float()
        #                 inter = (p * t).sum()
        #                 union = p.sum() + t.sum()
        #                 f1_list.append((2 * inter / union).item() if union > 0 else 1.0)

        #             f1 = np.mean(f1_list)
        #             if f1 >= self.f1_threshold:
        #                 kept += 1
        #                 refined += 1
                

        print(f"  Pseudo-label slices generated.")


    #              JOINT TRAINING (seg + contrastive)

    def train_joint(self):

        print("\n")
        print("    Joint training")
        print("\n")

        n_iters = self.config["training"]["joint_iterations"]
        n_epochs = self.config["training"]["joint_epochs"]
        K = self.config["dataset"]["num_classes"]

        for it in range(n_iters):

            print(f"\n Iteration {it+1}/{n_iters}")
            self.generate_pseudo_labels()

            for ep in range(n_epochs):

                self.model.train()
                seg_run, cont_run = 0.0, 0.0
                unlabeled_iter = iter(self.unlabeled_loader)

                for imgs_l, masks_l in tqdm(self.train_loader_labeled):
                    imgs_l, masks_l = imgs_l.to(self.device), masks_l.to(self.device)

                    # segmentation branch (supervised on labeled data)
                    seg_logits, _ = self.model(imgs_l, return_embeddings=True)
                    seg_loss = self.seg_loss_fn(seg_logits, masks_l)

                    # unlabeled batch → two intensity augmentations
                    try:
                        imgs_u, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_loader)
                        imgs_u, _ = next(unlabeled_iter)

                    imgs_u = imgs_u.to(self.device)
                    imgs_u1 = self._rand_intensity_aug(imgs_u)
                    imgs_u2 = self._rand_intensity_aug(imgs_u)

                    # build a 2B batch: (view1, view2)
                    imgs_pair = torch.cat([imgs_u1, imgs_u2], dim=0)

                    # pseudo-labels from the current network
                    with torch.no_grad():
                        probs = F.softmax(self.model(imgs_pair), dim=1)
                        pseudo = torch.argmax(probs, dim=1)
                        pseudo_oh = F.one_hot(pseudo, K).permute(0, 3, 1, 2).float()

                    # embeddings from contrastive branch
                    _, emb = self.model(imgs_pair, return_embeddings=True)

                    # contrastive loss on unlabeled data
                    cont_loss = self.cont_loss_fn(emb, pseudo_oh)

                    # total loss = supervised + lambda * contrastive
                    total = seg_loss + self.lambda_cont * cont_loss

                    self.optimizer.zero_grad()
                    total.backward()
                    self.optimizer.step()

                    seg_run += seg_loss.item()
                    cont_run += cont_loss.item()

                print(f"Epoch {ep+1}/{n_epochs} | Seg={seg_run/len(self.train_loader_labeled):.4f} | Cont={cont_run/len(self.train_loader_labeled):.4f}")

                if (ep + 1) % self.config["training"]["val_step_update"] == 0:
                    val = self.validate()
                    print(f"  Val Dice: {val:.4f}")

                    if val > self.best_val_dice:
                        self.best_val_dice = val
                        self.save_model("joint_best.pth")

            # reload best model at the end of each iteration
            self.load_model("joint_best.pth")

        print(f" Final joint training complete | Best Dice={self.best_val_dice:.4f}")
        self.save_model("final_model.pth")


    def validate(self):
        # Simple Dice evaluation on the validation set
        self.model.eval()
        scores = []
        K = self.config["dataset"]["num_classes"]

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, masks = batch
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                preds = torch.argmax(self.model(imgs), dim=1)

                for c in range(1, K):
                    p = (preds == c).float()
                    t = (masks == c).float()

                    inter = (p * t).sum()
                    union = p.sum() + t.sum()

                    if union > 0:
                        scores.append((2 * inter / union).item())

        return float(np.mean(scores)) if scores else 0.0


    def save_model(self, fname):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "best_val": self.best_val_dice,
            },
            os.path.join(self.save_dir, fname)
        )
        print(f"Saved {fname}")


    def load_model(self, fname):
        path = os.path.join(self.save_dir, fname)
        if not os.path.exists(path):
            print(f"⚠ No checkpoint {fname}")
            return

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["opt_state"])
        self.best_val_dice = ckpt["best_val"]
        print(f"Loaded {fname}")