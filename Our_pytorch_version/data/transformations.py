"""Data augmentations we use with torchvision.

We keep everything here so we can reuse the same transforms in our experiments.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional


class Compose:
    """Simple wrapper to apply several transforms one after another."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, mask=None):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class RandomRotation:
    """Randomly rotate the image (and the mask if there is one)."""
    
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range
    
    def __call__(self, img, mask=None):
        angle = np.random.uniform(*self.angle_range)
        
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        if mask is not None:
            mask = TF.rotate(mask.unsqueeze(0).float(), angle, 
                             interpolation=TF.InterpolationMode.NEAREST, fill=0).long().squeeze(0)
        
        return img, mask


class RandomScale:
    """Randomly zoom in or out and crop back to the original size."""
    
    def __init__(self, scale_range=(0.95, 1.05), img_size=(192, 192)):
        self.scale_range = scale_range
        self.img_size = img_size
    
    def __call__(self, img, mask=None):
        scale = np.random.uniform(*self.scale_range)
        new_size = (int(self.img_size[0] * scale), int(self.img_size[1] * scale))
        
        img = TF.resize(img, new_size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, self.img_size)
        
        if mask is not None:
            mask = TF.resize(mask.unsqueeze(0), new_size, 
                             interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.center_crop(mask, self.img_size).long().squeeze(0)
        
        return img, mask


class RandomFlipLR:
    """Flip the image left-right with some probability."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, mask=None):
        if np.random.rand() < self.p:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)
        
        return img, mask


class RandomBrightnessContrast:
    """Randomly change brightness and contrast of the image."""
    
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img, mask=None):
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        
        return img, mask


class ElasticDeformation:
    """Apply a simple elastic deformation using random displacement fields.
    
    This is a light version: good enough for our experiments,
    without pulling in extra libraries.
    """
    
    def __init__(self, sigma=10, alpha=50):
        self.sigma = sigma
        self.alpha = alpha
    
    def __call__(self, img, mask=None):
        # We create random displacement fields for x and y
        H, W = img.shape[-2:]
        
        dx = torch.randn(1, H, W) * self.sigma
        dy = torch.randn(1, H, W) * self.sigma
        
        # Smooth them a bit so the deformation is not too noisy
        dx = TF.gaussian_blur(dx.unsqueeze(0), kernel_size=15, sigma=5)[0]
        dy = TF.gaussian_blur(dy.unsqueeze(0), kernel_size=15, sigma=5)[0]
        
        # Control the strength of the deformation
        dx *= self.alpha / self.sigma
        dy *= self.alpha / self.sigma
        
        # Create coordinate grids
        coords_x, coords_y = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        
        coords_x = coords_x.unsqueeze(0) + dx
        coords_y = coords_y.unsqueeze(0) + dy
        
        # Normalize to [-1, 1] for grid_sample
        coords_x = 2 * coords_x / (H - 1) - 1
        coords_y = 2 * coords_y / (W - 1) - 1
        
        grid = torch.stack([coords_y, coords_x], dim=-1)
        
        # Deform the image
        img = F.grid_sample(img.unsqueeze(0), grid, padding_mode='border', 
                            align_corners=False).squeeze(0)
        
        if mask is not None:
            # Deform the mask with nearest interpolation
            mask = F.grid_sample(mask.float().unsqueeze(0).unsqueeze(0), grid, 
                                mode='nearest', padding_mode='border',
                                align_corners=False).squeeze(0).long()
        
        return img, mask


def get_augmentations(aug_type='basic', img_size=(192, 192)):
    """Build an augmentation pipeline based on a string name.
    
    Args:
        aug_type: which pipeline we want:
                  - 'basic': rotation, flip, brightness/contrast
                  - 'standard': basic + scaling
                  - 'strong': standard + elastic deformation
        img_size: image size used for scaling operations
    
    Returns:
        A Compose object or None if we do not use any augmentation.
    """
    transforms = []
    
    if aug_type == 'basic' or aug_type == 'standard' or aug_type == 'strong':
        transforms.append(RandomRotation())
        transforms.append(RandomFlipLR())
        transforms.append(RandomBrightnessContrast())
    
    if aug_type == 'standard' or aug_type == 'strong':
        transforms.append(RandomScale(img_size=img_size))
    
    if aug_type == 'strong':
        transforms.append(ElasticDeformation())
    
    if transforms:
        return Compose(transforms)
    else:
        return None


