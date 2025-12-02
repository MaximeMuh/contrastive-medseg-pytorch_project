import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from Our_pytorch_version.data import MedicalImageDataset

# Helpers for config

def get_num_classes(config):
    """We read the number of classes from the config."""
    return config["dataset"]["num_classes"]


def get_class_names(config):
    """
    We build class names from the config.
    We try to use 'structure_names' if it exists.
    """
    num_classes = get_num_classes(config)
    struct_names = config.get("structure_names", None)

    # Case 1: structure_names has num_classes - 1 entries (no background)
    if struct_names is not None and len(struct_names) == num_classes - 1:
        class_names = ["Background"] + list(struct_names)

    # Case 2: structure_names has exactly num_classes entries
    elif struct_names is not None and len(struct_names) == num_classes:
        class_names = list(struct_names)

    # Fallback: generic names
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    return class_names


# Dice per class on batch

def compute_dice_per_class_batch(preds, targets, num_classes):
    """
    We compute the Dice score for each class on a whole set of predictions.
    preds, targets: tensors of shape [N, H, W] (or can be flattened).
    """
    dice_scores = torch.zeros(num_classes, dtype=torch.float32)

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            # Class not present in preds and targets
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union

        dice_scores[cls] = dice

    return dice_scores.numpy()


# Collect predictions on val set

def collect_val_predictions(trainer, device):
    """
    We run the model on the whole validation loader.
    We return stacked preds and targets on CPU.
    """
    all_preds = []
    all_targets = []

    trainer.model.eval()
    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = trainer.model(imgs)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)      # [N, H, W]
    all_targets = torch.cat(all_targets, dim=0)  # [N, H, W]

    return all_preds, all_targets


# Per-class Dice plotting


def plot_dice_per_class(dice_per_class, class_names, title_suffix="validation set"):
    """
    We plot a bar chart of Dice scores per class.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, dice_per_class, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Dice score")
    plt.title(f"Dice score per class on {title_suffix}")
    plt.grid(True, alpha=0.3, axis="y")

    for bar, dice in zip(bars, dice_per_class):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{dice:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


# Slice-wise Dice distribution

def compute_slice_dice_all(trainer, device, num_classes, compute_dice_score_fn):
    """
    We compute a Dice score (all classes together) for each slice in the val set.
    We also keep the corresponding image, pred, mask.
    """
    dice_scores_list = []
    images_list = []
    preds_list = []
    masks_list = []

    trainer.model.eval()
    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = trainer.model(imgs)
            preds = torch.argmax(logits, dim=1)

            for i in range(imgs.shape[0]):
                pred_i = preds[i:i + 1].cpu()
                mask_i = masks[i:i + 1].cpu()

                # compute_dice_score_fn is the global function from your utils
                dice_i = compute_dice_score_fn(
                    pred_i, mask_i, num_classes=num_classes
                )

                dice_scores_list.append(dice_i)
                images_list.append(imgs[i].cpu())
                preds_list.append(preds[i].cpu())
                masks_list.append(masks[i].cpu())

    dice_scores_arr = np.array(dice_scores_list)
    return dice_scores_arr, images_list, preds_list, masks_list


def plot_dice_hist_and_box(dice_scores_arr, title_prefix="validation set"):
    """
    We plot histogram and box plot for Dice scores.
    """
    print("\nDice score statistics")
    print("=" * 60)
    print(f"Mean   : {dice_scores_arr.mean():.4f}")
    print(f"Std    : {dice_scores_arr.std():.4f}")
    print(f"Min    : {dice_scores_arr.min():.4f}")
    print(f"Max    : {dice_scores_arr.max():.4f}")
    print(f"Median : {np.median(dice_scores_arr):.4f}")
    print("=" * 60)

    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(dice_scores_arr, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(dice_scores_arr.mean(), linestyle="--",
                linewidth=2, label=f"Mean: {dice_scores_arr.mean():.3f}")
    plt.axvline(np.median(dice_scores_arr), linestyle="--",
                linewidth=2, label=f"Median: {np.median(dice_scores_arr):.3f}")
    plt.xlabel("Dice score")
    plt.ylabel("Frequency")
    plt.title(f"Dice score distribution ({title_prefix})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(dice_scores_arr, vert=True, patch_artist=True)
    plt.ylabel("Dice score")
    plt.title(f"Box plot of Dice scores ({title_prefix})")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_best_worst_examples(
    dice_scores_arr, images_list, preds_list, masks_list, num_classes
):
    """
    We show the best and worst slices (input, GT, prediction).
    """
    best_idx = np.argmax(dice_scores_arr)
    worst_idx = np.min(np.where(dice_scores_arr == dice_scores_arr.min()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Best example
    axes[0, 0].imshow(images_list[best_idx].squeeze().numpy(), cmap="gray")
    axes[0, 0].set_title(f"Best: Input (Dice = {dice_scores_arr[best_idx]:.3f})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        masks_list[best_idx].numpy(),
        cmap="tab10",
        vmin=0,
        vmax=num_classes - 1,
    )
    axes[0, 1].set_title("Best: Ground truth")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(
        preds_list[best_idx].numpy(),
        cmap="tab10",
        vmin=0,
        vmax=num_classes - 1,
    )
    axes[0, 2].set_title("Best: Prediction")
    axes[0, 2].axis("off")

    # Worst example
    axes[1, 0].imshow(images_list[worst_idx].squeeze().numpy(), cmap="gray")
    axes[1, 0].set_title(f"Worst: Input (Dice = {dice_scores_arr[worst_idx]:.3f})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        masks_list[worst_idx].numpy(),
        cmap="tab10",
        vmin=0,
        vmax=num_classes - 1,
    )
    axes[1, 1].set_title("Worst: Ground truth")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(
        preds_list[worst_idx].numpy(),
        cmap="tab10",
        vmin=0,
        vmax=num_classes - 1,
    )
    axes[1, 2].set_title("Worst: Prediction")
    axes[1, 2].axis("off")

    plt.suptitle("Best vs worst predictions on validation set",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Simple batch visualization

def visualize_batch_predictions(trainer, device, num_classes, n_samples=8):
    """
    We show input / ground truth / prediction for one batch.
    """
    trainer.model.eval()
    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = trainer.model(imgs)
            preds = torch.argmax(logits, dim=1)
            break

    n_samples = min(n_samples, imgs.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        axes[i, 0].imshow(imgs[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i, 0].set_title(f"Input {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(
            masks[i].cpu().numpy(),
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[i, 1].set_title(f"Ground truth {i + 1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(
            preds[i].cpu().numpy(),
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[i, 2].set_title(f"Prediction {i + 1}")
        axes[i, 2].axis("off")

    plt.suptitle(
        "Predictions vs Ground truth (validation batch)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()


# Overlays and error maps

def visualize_overlays_and_errors(trainer, device, num_classes, n_samples=6):
    """
    We build overlays (GT and prediction) and error maps on one batch.
    """
    trainer.model.eval()
    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = trainer.model(imgs)
            preds = torch.argmax(logits, dim=1)
            break

    n_samples = min(n_samples, imgs.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        img = imgs[i].squeeze().cpu().numpy()
        pred = preds[i].cpu().numpy()
        mask = masks[i].cpu().numpy()

        error_map = (pred != mask).astype(float)

        img_overlay_gt = img.copy()
        img_overlay_gt[mask > 0] = (
            0.7 * img_overlay_gt[mask > 0] + 0.3 * mask[mask > 0]
        )

        img_overlay_pred = img.copy()
        img_overlay_pred[pred > 0] = (
            0.7 * img_overlay_pred[pred > 0] + 0.3 * pred[pred > 0]
        )

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(
            img_overlay_gt,
            cmap="jet",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[i, 1].set_title("Overlay Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(
            img_overlay_pred,
            cmap="jet",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[i, 2].set_title("Overlay Prediction")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img, cmap="gray", alpha=0.5)
        axes[i, 3].imshow(error_map, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        axes[i, 3].set_title("Error map (red = error)")
        axes[i, 3].axis("off")

    plt.suptitle(
        "Qualitative analysis: overlays and error maps",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# Volume-level evaluation

def build_val_volume_dataset(config, root_dir):
    """
    We build a MedicalImageDataset for validation volumes.
    This needs 'data_path_tr_cropped' and 'val_ids' in the config.
    """
    base_path = Path(root_dir) / config["dataset"]["data_path_tr_cropped"]
    img_paths = [
        base_path / f"patient{val_id}" / "img_cropped.nii.gz"
        for val_id in config["dataset"]["val_ids"]
    ]
    mask_paths = [
        base_path / f"patient{val_id}" / "mask_cropped.nii.gz"
        for val_id in config["dataset"]["val_ids"]
    ]

    val_dataset = MedicalImageDataset(
        [str(p) for p in img_paths],
        [str(p) for p in mask_paths],
        target_resolution=config["dataset"]["target_resolution"],
        img_size=config["dataset"]["img_size"],
    )
    return val_dataset


def predict_volume_slices(model, img_vol, device):
    """
    We predict a full volume slice by slice.
    img_vol has shape [H, W, D].
    We return a tensor [H, W, D] of predicted labels.
    """
    model.eval()
    pred_slices = []
    num_slices = img_vol.shape[-1]

    with torch.no_grad():
        for slice_idx in range(num_slices):
            img_slice = img_vol[:, :, slice_idx]  # [H, W]
            img_slice = img_slice.unsqueeze(0).unsqueeze(0).to(device)

            logits = model(img_slice)
            pred_slice = torch.argmax(logits, dim=1)  # [1, H, W]
            pred_slices.append(pred_slice.squeeze(0).cpu())

    pred_vol = torch.stack(pred_slices, dim=-1)  # [H, W, D]
    return pred_vol


def compute_and_plot_volume_dice(
    model, val_dataset, config, device, compute_dice_score_fn
):
    """
    We compute and plot Dice per volume (patient).
    """
    num_classes = get_num_classes(config)
    val_ids = config["dataset"]["val_ids"]

    patient_dices = []

    print("Dice per volume (patient)")
    print("=" * 60)

    for vol_idx in range(len(val_dataset)):
        img_vol, mask_vol = val_dataset[vol_idx]  # [H, W, D]

        pred_vol = predict_volume_slices(model, img_vol, device)  # [H, W, D]

        dice_vol = compute_dice_score_fn(
            pred_vol.unsqueeze(0),     # [1, H, W, D]
            mask_vol.unsqueeze(0),     # [1, H, W, D]
            num_classes=num_classes,
        )

        patient_dices.append(dice_vol)
        print(f"Patient {val_ids[vol_idx]}: Dice = {dice_vol:.4f}")

    print("=" * 60)
    print(f"Average Dice over patients: {np.mean(patient_dices):.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    patients = [f"Patient {p}" for p in val_ids]
    bars = plt.bar(patients, patient_dices, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Dice score")
    plt.title("Dice score per patient on validation set")
    plt.xticks(rotation=45)

    for bar, dice in zip(bars, patient_dices):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{dice:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.axhline(
        np.mean(patient_dices),
        linestyle="--",
        label=f"Mean: {np.mean(patient_dices):.3f}",
    )
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    return patient_dices


def visualize_volume_slices_for_patient(
    model, img_vol, mask_vol, patient_name, device, num_classes, max_slices=12
):
    """
    We show selected slices for one volume: input, GT, prediction.
    """
    pred_vol = predict_volume_slices(model, img_vol, device)  # [H, W, D]
    num_slices = img_vol.shape[-1]

    slice_indices = np.linspace(
        0, num_slices - 1, min(max_slices, num_slices), dtype=int
    )
    n_rows = len(slice_indices)

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, slice_idx in enumerate(slice_indices):
        axes[row, 0].imshow(img_vol[:, :, slice_idx].numpy(), cmap="gray")
        axes[row, 0].set_title(f"Input - Slice {slice_idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            mask_vol[:, :, slice_idx].numpy(),
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[row, 1].set_title(f"Ground truth - Slice {slice_idx}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(
            pred_vol[:, :, slice_idx].numpy(),
            cmap="tab10",
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[row, 2].set_title(f"Prediction - Slice {slice_idx}")
        axes[row, 2].axis("off")

    plt.suptitle(
        f"Patient {patient_name} - Selected slices",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()


# Confusion matrix

def plot_confusion_matrix_from_preds(
    all_preds, all_targets, num_classes, class_names=None, normalize=True
):
    """
    We build and plot a confusion matrix from full-val preds/targets.
    """
    y_true = all_targets.view(-1).numpy()
    y_pred = all_preds.view(-1).numpy()

    if normalize:
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes)), normalize="true"
        )
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.show()

def compute_tsne_pixel_embeddings(
    trainer,
    device,
    num_classes,
    feature_fn,
    max_pixels=20000,
    use_gt_labels=True,
):
    """
    We build a t-SNE on pixel embeddings from one or several validation batches.

    - feature_fn(model, x) must return a tensor [N, C, H, W]
      with the features we want to visualize.
    - We sample up to max_pixels pixels from all images.
    - Each pixel is labeled by its class (GT or prediction).
    """

    model = trainer.model
    model.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # We get feature maps from the chosen layer
            feats = feature_fn(model, imgs)  # [N, C, H, W]
            # We use GT or predicted labels for coloring
            if use_gt_labels:
                labels = masks
            else:
                logits = model(imgs)
                labels = torch.argmax(logits, dim=1)

            # We move to CPU
            feats = feats.cpu()
            labels = labels.cpu()

            # We flatten: pixels become rows
            N, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)   # [N*H*W, C]
            labels_flat = labels.reshape(-1)                        # [N*H*W]

            # We remove background (class 0) to see structures better
            mask_fg = labels_flat > 0
            feats_fg = feats_flat[mask_fg]
            labels_fg = labels_flat[mask_fg]

            all_feats.append(feats_fg)
            all_labels.append(labels_fg)

            # We stop if we already have more than max_pixels
            total_pixels = sum(f.shape[0] for f in all_feats)
            if total_pixels >= max_pixels:
                break

    if len(all_feats) == 0:
        print("No foreground pixels found for t-SNE.")
        return None, None

    feats_all = torch.cat(all_feats, dim=0)   # [M, C]
    labels_all = torch.cat(all_labels, dim=0) # [M]

    # We subsample to max_pixels
    M = feats_all.shape[0]
    if M > max_pixels:
        idx = torch.randperm(M)[:max_pixels]
        feats_all = feats_all[idx]
        labels_all = labels_all[idx]

    # We run t-SNE
    print(f"Running t-SNE on {feats_all.shape[0]} pixels, feature dim = {feats_all.shape[1]}")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=42,
    )
    emb_2d = tsne.fit_transform(feats_all.numpy())  # [M, 2]

    return emb_2d, labels_all.numpy()

def compute_tsne_pixel_embeddings_with_back(
    trainer,
    device,
    num_classes,
    feature_fn,
    max_pixels=20000,
    use_gt_labels=True,
):
    """
    We build a t-SNE on pixel embeddings from one or several validation batches.

    - feature_fn(model, x) must return a tensor [N, C, H, W]
      with the features we want to visualize.
    - We sample up to max_pixels pixels from all images.
    - Each pixel is labeled by its class (GT or prediction).
    """

    model = trainer.model
    model.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # We get feature maps from the chosen layer
            feats = feature_fn(model, imgs)  # [N, C, H, W]
            # We use GT or predicted labels for coloring
            if use_gt_labels:
                labels = masks
            else:
                logits = model(imgs)
                labels = torch.argmax(logits, dim=1)

            # We move to CPU
            feats = feats.cpu()
            labels = labels.cpu()

            # We flatten: pixels become rows
            N, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)   # [N*H*W, C]
            labels_flat = labels.reshape(-1)                        # [N*H*W]

            # We remove background (class 0) to see structures better
            mask_fg = labels_flat >= 0
            feats_fg = feats_flat[mask_fg]
            labels_fg = labels_flat[mask_fg]

            all_feats.append(feats_fg)
            all_labels.append(labels_fg)

            # We stop if we already have more than max_pixels
            total_pixels = sum(f.shape[0] for f in all_feats)
            if total_pixels >= max_pixels:
                break

    if len(all_feats) == 0:
        print("No foreground pixels found for t-SNE.")
        return None, None

    feats_all = torch.cat(all_feats, dim=0)   # [M, C]
    labels_all = torch.cat(all_labels, dim=0) # [M]

    # We subsample to max_pixels
    M = feats_all.shape[0]
    if M > max_pixels:
        idx = torch.randperm(M)[:max_pixels]
        feats_all = feats_all[idx]
        labels_all = labels_all[idx]

    # We run t-SNE
    print(f"Running t-SNE on {feats_all.shape[0]} pixels, feature dim = {feats_all.shape[1]}")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=42,
    )
    emb_2d = tsne.fit_transform(feats_all.numpy())  # [M, 2]

    return emb_2d, labels_all.numpy()
from sklearn.manifold import TSNE

def compute_tsne_pixel_embeddings_mix(
    trainer,
    device,
    num_classes,
    feature_fn,
    max_per_class=3000,
    include_background=True,
    use_gt_labels=True,
):
    """
    We build a t-SNE on pixel embeddings from the validation set.

    - feature_fn(model, x) must return a tensor [N, C, H, W]
      with the features we want to visualize.
    - We sample up to max_per_class pixels PER CLASS.
    - We can include or exclude background with include_background.
    """

    model = trainer.model
    model.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # We get feature maps from the chosen layer
            feats = feature_fn(model, imgs)  # [N, C, H, W]

            # We use GT or predicted labels for coloring
            if use_gt_labels:
                labels = masks
            else:
                logits = model(imgs)
                labels = torch.argmax(logits, dim=1)

            # We move to CPU
            feats = feats.cpu()
            labels = labels.cpu()

            # We flatten: pixels become rows
            N, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)   # [N*H*W, C]
            labels_flat = labels.reshape(-1)                        # [N*H*W]

            all_feats.append(feats_flat)
            all_labels.append(labels_flat)

    if len(all_feats) == 0:
        print("No pixels found for t-SNE.")
        return None, None

    feats_all = torch.cat(all_feats, dim=0)   # [M, C]
    labels_all = torch.cat(all_labels, dim=0) # [M]

    # Balanced sampling per class
    indices_list = []
    unique_labels = labels_all.unique()

    for lbl in unique_labels:
        lbl_int = int(lbl.item())

        # We can skip background if we want
        if not include_background and lbl_int == 0:
            continue

        cls_idx = torch.nonzero(labels_all == lbl, as_tuple=False).squeeze(1)

        if cls_idx.numel() == 0:
            continue

        # We sample up to max_per_class indices for this class
        if cls_idx.numel() > max_per_class:
            perm = torch.randperm(cls_idx.numel())[:max_per_class]
            cls_idx = cls_idx[perm]

        indices_list.append(cls_idx)

    if len(indices_list) == 0:
        print("No classes selected for t-SNE.")
        return None, None

    indices = torch.cat(indices_list, dim=0)

    feats_bal = feats_all[indices]     # [K, C]
    labels_bal = labels_all[indices]   # [K]

    print(
        f"Running t-SNE on {feats_bal.shape[0]} pixels "
        f"(max {max_per_class} per class), feature dim = {feats_bal.shape[1]}"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=42,
    )
    emb_2d = tsne.fit_transform(feats_bal.numpy())  # [K, 2]

    return emb_2d, labels_bal.numpy()
import torch
import numpy as np
from sklearn.manifold import TSNE

def compute_tsne_pixel_embeddings_balanced(
    trainer,
    device,
    feature_fn,
    total_pixels=10000,     
    min_per_class=500,       
    include_background=True,
    use_gt_labels=True,
):
    """
    Builds a t-SNE on pixel embeddings with a 'Safety Floor + Fill' strategy.
    
    1. We guarantee at least 'min_per_class' pixels for every class (if available).
    2. We fill the remaining slots (up to 'total_pixels') randomly from the 
       remaining pool of pixels, preserving the natural class imbalance.
    """

    model = trainer.model
    model.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in trainer.val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Extraction  features [N, C, H, W]
            feats = feature_fn(model, imgs)

            if use_gt_labels:
                labels = masks
            else:
                logits = model(imgs)
                labels = torch.argmax(logits, dim=1)

            feats = feats.cpu()
            labels = labels.cpu()

            # Flatten: [N, C, H, W] -> [N*H*W, C]
            N, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)

            all_feats.append(feats_flat)
            all_labels.append(labels_flat)

    if len(all_feats) == 0:
        print("No pixels found for t-SNE.")
        return None, None


    feats_all = torch.cat(all_feats, dim=0)   # [M, C]
    labels_all = torch.cat(all_labels, dim=0) # [M]

    
    unique_labels = labels_all.unique()
    indices_kept = []
    
    remaining_indices_mask = torch.ones(labels_all.shape[0], dtype=torch.bool)

    for lbl in unique_labels:
        lbl_int = int(lbl.item())


        if not include_background and lbl_int == 0:
            remaining_indices_mask[labels_all == lbl] = False 
            continue


        cls_indices = torch.nonzero(labels_all == lbl, as_tuple=False).squeeze(1)
        n_available = cls_indices.numel()

        if n_available == 0:
            continue


        n_take = min(n_available, min_per_class)
        

        perm = torch.randperm(n_available)
        selected_idx = cls_indices[perm[:n_take]]
        
        indices_kept.append(selected_idx)
        

        remaining_indices_mask[selected_idx] = False


    if len(indices_kept) > 0:
        guaranteed_indices = torch.cat(indices_kept, dim=0)
    else:
        guaranteed_indices = torch.tensor([], dtype=torch.long)


    current_count = guaranteed_indices.numel()
    needed = total_pixels - current_count

    if needed > 0:

        pool_indices = torch.nonzero(remaining_indices_mask, as_tuple=False).squeeze(1)
        
        if pool_indices.numel() > 0:

            n_take_rest = min(needed, pool_indices.numel())
            perm_rest = torch.randperm(pool_indices.numel())
            fill_indices = pool_indices[perm_rest[:n_take_rest]]
            

            final_indices = torch.cat([guaranteed_indices, fill_indices], dim=0)
        else:
            final_indices = guaranteed_indices
    else:

        final_indices = guaranteed_indices[:total_pixels]


    
    feats_sampled = feats_all[final_indices]
    labels_sampled = labels_all[final_indices]

    print(
        f"Running t-SNE on {feats_sampled.shape[0]} pixels "
        f"(Target Total: {total_pixels}, Min/Class: {min_per_class}).\n"
        f"Class counts in t-SNE subset: {torch.unique(labels_sampled, return_counts=True)}"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=42,
    )
    emb_2d = tsne.fit_transform(feats_sampled.numpy())

    return emb_2d, labels_sampled.numpy()
def plot_tsne_embeddings(emb_2d, labels, class_names=None, title="t-SNE of pixel embeddings"):
    """
    We plot the 2D t-SNE embeddings colored by class.
    """
    if emb_2d is None or labels is None:
        print("No embeddings to plot.")
        return

    if class_names is None:
        num_classes = int(labels.max()) + 1
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(8, 8))

    for cls in np.unique(labels):
        mask = labels == cls
        if cls < len(class_names):
            label_name = class_names[cls]
        else:
            label_name = f"Class {cls}"
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=5,
            alpha=0.7,
            label=label_name,
        )

    plt.legend(markerscale=3)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()