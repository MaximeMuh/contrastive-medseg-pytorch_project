"""Dice loss that we use for our segmentation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss with an option to include or ignore the background.
    
    We use this loss for medical image segmentation, 
    where some classes are much smaller than others.
    """
    
    def __init__(self, num_classes, with_background=True, epsilon=1e-10):
        super(DiceLoss, self).__init__()
        
        self.num_classes = num_classes
        self.with_background = with_background
        self.epsilon = epsilon
        
    def forward(self, logits, labels):
        """Compute the Dice loss for one batch.
        
        Args:
            logits: raw network outputs (B, num_classes, H, W)
            labels: ground-truth labels, either:
                    - class indices (B, H, W), or
                    - one-hot maps (B, num_classes, H, W)
        
        Returns:
            A single scalar: average Dice loss over the batch.
        """
        # Turn logits into probabilities
        probs = F.softmax(logits, dim=1)
        
        # If labels are class indices, we convert them to one-hot
        if len(labels.shape) == 3:  # (B, H, W)
            labels_onehot = F.one_hot(labels, self.num_classes)  # (B, H, W, num_classes)
            labels_onehot = labels_onehot.permute(0, 3, 1, 2).float()  # (B, num_classes, H, W)
        else:
            labels_onehot = labels
        
        # Intersection between prediction and ground truth
        intersection = probs * labels_onehot
        intersection = intersection.sum(dim=(2, 3))  # (B, num_classes)
        
        # Sums for each class (prediction and GT)
        probs_sum = probs.sum(dim=(2, 3))        # (B, num_classes)
        labels_sum = labels_onehot.sum(dim=(2, 3))  # (B, num_classes)
        union = probs_sum + labels_sum
        
        # Dice score per class and per sample
        dice = 2.0 * intersection / (union + self.epsilon)  # (B, num_classes)
        
        # Decide if we include the background class (0) in the loss
        if self.with_background:
            loss = 1.0 - dice.mean()
        else:
            loss = 1.0 - dice[:, 1:].mean()
        
        return loss