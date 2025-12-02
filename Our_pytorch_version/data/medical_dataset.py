"""Small PyTorch dataset we use to load our 3D medical images.

We wrote this helper so we don’t have to copy-paste the same loading code everywhere.
"""

import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import transform


class MedicalImageDataset(Dataset):
    """Dataset that loads 3D medical images and masks.
    
    It:
    - loads NIfTI volumes
    - normalizes intensities (if we want)
    - resamples to a fixed resolution
    - crops/pads slices to the same size
    """
    
    def __init__(self, img_paths, mask_paths=None, target_resolution=(1.36719, 1.36719),
                 img_size=(192, 192), normalize=True, label_present=True):
        """Set up the dataset.
        
        Args:
            img_paths: list of image file paths.
            mask_paths: list of mask file paths (or None if unlabeled).
            target_resolution: target pixel spacing (x, y) in mm.
            img_size: final slice size (height, width).
            normalize: if True, apply simple min-max normalization.
            label_present: True if we have masks for these images.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths if mask_paths else [None] * len(img_paths)
        self.target_resolution = target_resolution
        self.img_size = img_size
        self.normalize = normalize
        self.label_present = label_present
        
        # We keep all volumes in memory
        self.images = []
        self.masks = []
        
        for img_path, mask_path in zip(self.img_paths, self.mask_paths):
            # Load image volume
            img_data = self._load_nifti(img_path)
            self.images.append(img_data)
            
            # Load mask volume if available
            if self.label_present and mask_path is not None:
                mask_data = self._load_nifti(mask_path, is_mask=True)
                self.masks.append(mask_data)
            else:
                self.masks.append(None)
    
    def _normalize_minmax(self, image_data, min_val=1, max_val=99):
        """Put intensities roughly between 0 and 1 using percentiles."""
        min_1p = np.percentile(image_data, min_val)
        max_99p = np.percentile(image_data, max_val)
        
        normalized = (image_data - min_1p) / (max_99p - min_1p)
        return normalized
    
    def _load_nifti(self, filepath, is_mask=False):
        """Load one NIfTI file from disk and preprocess it."""
        if filepath is None:
            return None
        
        img_obj = nib.load(filepath)
        data = img_obj.get_fdata()
        pixel_size = img_obj.header['pixdim'][1:4]
        
        # Normalize if image (not mask)
        if not is_mask and self.normalize:
            data = self._normalize_minmax(data)
        
        # Resample and crop/pad
        data = self._preprocess(data, pixel_size, is_mask)
        
        return data
    
    def _preprocess(self, img, pixel_size, is_mask=False):
        """Resample the volume and make each slice the same size."""
        nx, ny = self.img_size
        
        # How much we scale in x and y
        scale_vector = [pixel_size[0] / self.target_resolution[0],
                        pixel_size[1] / self.target_resolution[1]]
        
        # Go through each axial slice
        processed_slices = []
        for slice_no in range(img.shape[2]):
            slice_img = img[:, :, slice_no]
            
            # Resample
            if is_mask:
                # nearest neighbor for masks
                slice_resampled = transform.rescale(
                    slice_img, scale_vector, order=0,
                    preserve_range=True, mode='constant')
            else:
                # linear interpolation for images
                slice_resampled = transform.rescale(
                    slice_img, scale_vector, order=1,
                    preserve_range=True, mode='constant')
            
            # Crop or pad to the desired size
            slice_final = self._crop_or_pad(slice_resampled, nx, ny)
            processed_slices.append(slice_final)
        
        # Stack slices back into a 3D volume
        processed = np.stack(processed_slices, axis=2)
        
        return processed
    
    def _crop_or_pad(self, img, nx, ny):
        """Center-crop or zero-pad a slice so it matches (nx, ny)."""
        x, y = img.shape
        
        slice_cropped = np.zeros((nx, ny))
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
        
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        
        return slice_cropped
    
    def __len__(self):
        """Return how many volumes we have."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Return one volume (and its mask if it exists)."""
        img = self.images[idx]
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img).float()
        
        if self.label_present and self.masks[idx] is not None:
            mask = self.masks[idx]
            mask_tensor = torch.from_numpy(mask).long()
            return img_tensor, mask_tensor
        else:
            return img_tensor


class SliceDataset(Dataset):
    """Takes 3D volumes and turns them into 2D slices for training."""
    
    def __init__(self, volume_dataset, transform=None):
        """Build a slice-level dataset on top of a volume dataset.
        
        Args:
            volume_dataset: MedicalImageDataset with 3D volumes.
            transform: optional transform applied to each (img_slice, mask_slice).
        """
        self.volume_dataset = volume_dataset
        self.transform = transform
        
        # Precompute all (volume_idx, slice_idx) pairs
        self.slices = []
        for vol_idx in range(len(volume_dataset)):
            num_slices = volume_dataset.images[vol_idx].shape[2]
            for slice_idx in range(num_slices):
                self.slices.append((vol_idx, slice_idx))
    
    def __len__(self):
        """Total number of 2D slices in this dataset."""
        return len(self.slices)
    
    def __getitem__(self, idx):
        """Return one slice (and its mask slice if available)."""
        vol_idx, slice_idx = self.slices[idx]
        
        # Get volume
        volume = self.volume_dataset[vol_idx]
        
        if isinstance(volume, tuple):
            img, mask = volume
            # Extract slice and add channel dim
            img_slice = img[:, :, slice_idx].unsqueeze(0)  # add channel dim
            mask_slice = mask[:, :, slice_idx]
            
            if self.transform:
                # Apply transform to this slice
                img_slice, mask_slice = self.transform(img_slice, mask_slice)
            
            return img_slice, mask_slice
        else:
            # Unlabeled data
            img_slice = volume[:, :, slice_idx].unsqueeze(0)
            
            if self.transform:
                img_slice, _ = self.transform(img_slice, None)
            
            # Always return tuple for compatibility with DataLoader
            return img_slice, None