"""Heads we put on top of the UNet: one for segmentation, one for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head g_xi.

    This small conv net takes the shared feature map from the decoder
    and outputs segmentation logits.
    """
    
    def __init__(self, in_channels, num_classes, base_channels=16):
        super(SegmentationHead, self).__init__()
        
        # Two 3x3 conv layers, as described in the paper
        self.seg_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.seg_bn1 = nn.BatchNorm2d(base_channels)
        self.seg_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.seg_bn2 = nn.BatchNorm2d(base_channels)
        
        # Final 1x1 conv that gives one logit map per class
        self.seg_final = nn.Conv2d(base_channels, num_classes, kernel_size=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the segmentation head.

        Args:
            x: feature map from the decoder, shape (B, C, H, W)

        Returns:
            seg_logits: segmentation logits of shape (B, num_classes, H, W)
        """
        x = F.relu(self.seg_bn1(self.seg_conv1(x)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = self.seg_final(x)
        return x


class ContrastiveHead(nn.Module):
    """
    Contrastive head h_phi.

    This head turns the shared feature map into pixel embeddings
    that we use for the local contrastive loss.
    """
    
    def __init__(self, in_channels, embed_dim, base_channels=16):
        super(ContrastiveHead, self).__init__()
        
        # Two 1x1 conv layers: simple projection to an embedding space
        self.cont_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.cont_bn1 = nn.BatchNorm2d(base_channels)
        self.cont_conv2 = nn.Conv2d(base_channels, embed_dim, kernel_size=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass of the contrastive head.

        Args:
            x: feature map from the decoder, shape (B, C, H, W)

        Returns:
            embeddings: pixel embeddings of shape (B, embed_dim, H, W)
        """
        x = F.relu(self.cont_bn1(self.cont_conv1(x)))
        x = self.cont_conv2(x)  # final layer, no activation
        return x