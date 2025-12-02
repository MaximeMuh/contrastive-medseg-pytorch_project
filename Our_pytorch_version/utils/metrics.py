"""Simple evaluation helpers we use for our segmentation experiments."""

import torch
import numpy as np


def compute_dice_score(pred, target, num_classes, mean=True):
    """
    Compute Dice score for each class (and optionally the mean).

    Args:
        pred: predicted labels, shape (B, H, W)
        target: ground-truth labels, shape (B, H, W)
        num_classes: total number of classes (including background = 0)
        mean: if True, we return the mean Dice (without background).
              if False, we return the Dice score for each class.

    Returns:
        Either a single mean Dice value or a list of per-class Dice values.
    """
    dice_scores = []
    
    for cls in range(1, num_classes):  # Skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union > 0:
            dice = 2.0 * intersection / union
            dice_scores.append(dice.item())
        else:
            # if both prediction and target are empty for this class
            dice_scores.append(1.0)
    
    if mean:
        return np.mean(dice_scores) if dice_scores else 0.0
    else:
        return dice_scores


def get_surface(mask):
    """
    Get the boundary/surface of a binary mask using a simple erosion trick.

    Args:
        mask: binary mask (H, W, D) or (H, W)

    Returns:
        binary mask of the same shape, where only surface pixels/voxels are 1.
    """
    from scipy.ndimage import binary_erosion
    
    # Surface = original mask - eroded mask
    eroded = binary_erosion(mask)
    surface = mask.astype(bool) ^ eroded
    
    return surface.astype(np.uint8)