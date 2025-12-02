"""UNet model with two heads: one for segmentation, one for contrastive learning."""

import torch
import torch.nn as nn

from .unet import UNet
from .heads import SegmentationHead, ContrastiveHead


class ContrastiveUNet(nn.Module):
    """
    Contrastive UNet used for our semi-supervised experiments.

    We:
    - use a shared UNet encoder-decoder as backbone
    - add a segmentation head g_ξ for the usual segmentation loss
    - add a contrastive head h_φ to produce embeddings for the local contrastive loss

    This follows the idea from:
    "Local contrastive loss with pseudo-label based self-training
    for semi-supervised medical image segmentation".
    """
    
    def __init__(self, in_channels=1, num_classes=4, base_channels=16, 
                 embed_dim=16, interp_method='nearest'):
        """
        Build the Contrastive UNet.

        Args:
            in_channels: number of input channels.
            num_classes: number of segmentation classes.
            base_channels: base number of channels in the UNet backbone.
            embed_dim: size of the embedding vector for the contrastive branch.
            interp_method: upsampling interpolation method in the decoder.
        """
        super(ContrastiveUNet, self).__init__()
        
        # Shared encoder-decoder backbone
        self.backbone = UNet(in_channels, base_channels, interp_method)
        
        # Two heads on top of the same feature maps
        self.seg_head = SegmentationHead(base_channels, num_classes, base_channels)
        self.contrastive_head = ContrastiveHead(base_channels, embed_dim, base_channels)
        
        # Store dimensions for convenience
        self.num_classes = num_classes
        self.embed_dim = embed_dim
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass.

        Args:
            x: input image tensor of shape (B, C, H, W).
            return_embeddings: if True, we also return the contrastive embeddings.

        Returns:
            If return_embeddings is False:
                seg_logits
            If return_embeddings is True:
                (seg_logits, embeddings)
        """
        # Shared backbone
        features = self.backbone(x)
        
        # Segmentation prediction
        seg_logits = self.seg_head(features)
        
        if return_embeddings:
            # Embeddings for the local contrastive loss
            embeddings = self.contrastive_head(features)
            return seg_logits, embeddings
        else:
            return seg_logits
    
    def get_encoder_params(self):
        """Return parameters of the encoder so we can tune learning rates, etc."""
        return self.backbone.encoder.parameters()
    
    def get_decoder_params(self):
        """Return parameters of the decoder part of the UNet."""
        return self.backbone.decoder.parameters()
    
    def get_seg_head_params(self):
        """Return parameters of the segmentation head."""
        return self.seg_head.parameters()
    
    def get_contrastive_head_params(self):
        """Return parameters of the contrastive head."""
        return self.contrastive_head.parameters()