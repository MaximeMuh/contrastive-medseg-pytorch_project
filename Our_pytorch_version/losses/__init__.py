"""Loss functions for semi-supervised medical image segmentation."""

from .dice_loss import DiceLoss
from .contrastive_loss import LocalContrastiveLoss

__all__ = ['DiceLoss', 'LocalContrastiveLoss']

