"""UNet encoder-decoder that we use for 2D medical image segmentation.

We keep the structure close to the original TF implementation:
one encoder with skip connections and one decoder that upsamples back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Small building block: Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        # return: (B, out_channels, H, W)
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """UNet encoder: we go down in resolution and store skip connections."""

    def __init__(self, in_channels, base_channels=16):
        super(Encoder, self).__init__()

        # [input, l1, l2, l3, l4, l5]
        # number of channels at each level
        self.no_filters = [
            in_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
        ]

        # Encoder path (downsampling)
        self.enc_conv1 = ConvBlock(self.no_filters[0], self.no_filters[1], kernel_size=3, padding=1)
        self.enc_conv1_b = ConvBlock(self.no_filters[1], self.no_filters[1], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc_conv2 = ConvBlock(self.no_filters[1], self.no_filters[2], kernel_size=3, padding=1)
        self.enc_conv2_b = ConvBlock(self.no_filters[2], self.no_filters[2], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_conv3 = ConvBlock(self.no_filters[2], self.no_filters[3], kernel_size=3, padding=1)
        self.enc_conv3_b = ConvBlock(self.no_filters[3], self.no_filters[3], kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc_conv4 = ConvBlock(self.no_filters[3], self.no_filters[4], kernel_size=3, padding=1)
        self.enc_conv4_b = ConvBlock(self.no_filters[4], self.no_filters[4], kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc_conv5 = ConvBlock(self.no_filters[4], self.no_filters[5], kernel_size=3, padding=1)
        self.enc_conv5_b = ConvBlock(self.no_filters[5], self.no_filters[5], kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.enc_conv6 = ConvBlock(self.no_filters[5], self.no_filters[5], kernel_size=3, padding=1)
        self.enc_conv6_b = ConvBlock(self.no_filters[5], self.no_filters[5], kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        # We go down: each pool halves H and W, and we store enc1..enc5 as skips.
        enc1 = self.enc_conv1_b(self.enc_conv1(x))   # (B, base, H,   W)
        x1 = self.pool1(enc1)                        # (B, base, H/2, W/2)

        enc2 = self.enc_conv2_b(self.enc_conv2(x1))  # (B, 2*base, H/2,   W/2)
        x2 = self.pool2(enc2)                        # (B, 2*base, H/4,   W/4)

        enc3 = self.enc_conv3_b(self.enc_conv3(x2))  # (B, 4*base, H/4,   W/4)
        x3 = self.pool3(enc3)                        # (B, 4*base, H/8,   W/8)

        enc4 = self.enc_conv4_b(self.enc_conv4(x3))  # (B, 8*base, H/8,   W/8)
        x4 = self.pool4(enc4)                        # (B, 8*base, H/16,  W/16)

        enc5 = self.enc_conv5_b(self.enc_conv5(x4))  # (B, 8*base, H/16,  W/16)
        x5 = self.pool5(enc5)                        # (B, 8*base, H/32,  W/32)

        enc6 = self.enc_conv6_b(self.enc_conv6(x5))  # bottleneck: (B, 8*base, H/32, W/32)

        # Return bottleneck and all skip connections
        # skips: [enc1, enc2, enc3, enc4, enc5]
        return enc6, [enc1, enc2, enc3, enc4, enc5]


class Decoder(nn.Module):
    """UNet decoder: we go back up and merge with skip connections."""

    def __init__(self, base_channels=16, interp_method='nearest'):
        super(Decoder, self).__init__()

        self.no_filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
        ]
        self.interp_method = interp_method

        # Decoder path (upsampling).
        # We upsample, then concatenate with the skip, then apply conv blocks.
        self.dec_conv1 = ConvBlock(self.no_filters[4], self.no_filters[4], kernel_size=3, padding=1)
        self.dec_conv1_b = ConvBlock(self.no_filters[4] * 2, self.no_filters[4], kernel_size=3, padding=1)
        self.dec_conv1_f = ConvBlock(self.no_filters[4], self.no_filters[4], kernel_size=3, padding=1)

        self.dec_conv2 = ConvBlock(self.no_filters[4], self.no_filters[3], kernel_size=3, padding=1)
        self.dec_conv2_b = ConvBlock(self.no_filters[3] * 2, self.no_filters[3], kernel_size=3, padding=1)
        self.dec_conv2_f = ConvBlock(self.no_filters[3], self.no_filters[3], kernel_size=3, padding=1)

        self.dec_conv3 = ConvBlock(self.no_filters[3], self.no_filters[2], kernel_size=3, padding=1)
        self.dec_conv3_b = ConvBlock(self.no_filters[2] * 2, self.no_filters[2], kernel_size=3, padding=1)
        self.dec_conv3_f = ConvBlock(self.no_filters[2], self.no_filters[2], kernel_size=3, padding=1)

        self.dec_conv4 = ConvBlock(self.no_filters[2], self.no_filters[1], kernel_size=3, padding=1)
        self.dec_conv4_b = ConvBlock(self.no_filters[1] * 2, self.no_filters[1], kernel_size=3, padding=1)
        self.dec_conv4_f = ConvBlock(self.no_filters[1], self.no_filters[1], kernel_size=3, padding=1)

        # Final upsampling (no skip connection here)
        self.dec_conv5 = ConvBlock(self.no_filters[1], self.no_filters[0], kernel_size=3, padding=1)

    def _upsample_to(self, x, ref):
        """Upsample x so that H,W match ref.shape[-2:]."""
        size = ref.shape[-2:]
        mode = self.interp_method
        if mode in ('bilinear', 'bicubic'):
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        else:
            return F.interpolate(x, size=size, mode=mode)

    def forward(self, bottleneck, skip_connections):
        """
        Args:
            bottleneck: tensor from encoder bottleneck, shape (B, 8*base, H/32, W/32)
            skip_connections: list [enc1, enc2, enc3, enc4, enc5]
        Returns:
            features: decoded feature map, same spatial size as enc1
        """
        enc1, enc2, enc3, enc4, enc5 = skip_connections

        # Level 5 -> 4
        x = self._upsample_to(bottleneck, enc5)     # (B, 8*base, H/16, W/16)
        x = self.dec_conv1(x)                      # (B, 8*base, H/16, W/16)
        x = torch.cat([x, enc5], dim=1)            # (B, 16*base, H/16, W/16)
        x = self.dec_conv1_f(self.dec_conv1_b(x))  # (B, 8*base, H/16, W/16)

        # Level 4 -> 3
        x = self._upsample_to(x, enc4)             # (B, 8*base, H/8, W/8)
        x = self.dec_conv2(x)                      # (B, 8*base, H/8, W/8)
        x = torch.cat([x, enc4], dim=1)            # (B, 16*base, H/8, W/8)
        x = self.dec_conv2_f(self.dec_conv2_b(x))  # (B, 4*base, H/8, W/8)

        # Level 3 -> 2
        x = self._upsample_to(x, enc3)             # (B, 4*base, H/4, W/4)
        x = self.dec_conv3(x)                      # (B, 4*base, H/4, W/4)
        x = torch.cat([x, enc3], dim=1)            # (B, 8*base, H/4, W/4)
        x = self.dec_conv3_f(self.dec_conv3_b(x))  # (B, 2*base, H/4, W/4)

        # Level 2 -> 1
        x = self._upsample_to(x, enc2)             # (B, 2*base, H/2, W/2)
        x = self.dec_conv4(x)                      # (B, 2*base, H/2, W/2)
        x = torch.cat([x, enc2], dim=1)            # (B, 4*base, H/2, W/2)
        x = self.dec_conv4_f(self.dec_conv4_b(x))  # (B, base, H/2, W/2)

        # Final upsampling to enc1 spatial size (no skip here in the TF arch)
        x = self._upsample_to(x, enc1)             # (B, base, H, W)
        x = self.dec_conv5(x)                      # (B, in_channels, H, W) if base == in_channels
        return x


class UNet(nn.Module):
    """Full UNet: encoder + decoder returning a feature map at input resolution."""

    def __init__(self, in_channels=1, base_channels=16, interp_method='nearest'):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels, interp_method)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        bottleneck, skip_connections = self.encoder(x)
        features = self.decoder(bottleneck, skip_connections)
        # features: (B, base_channels, H, W)
        return features