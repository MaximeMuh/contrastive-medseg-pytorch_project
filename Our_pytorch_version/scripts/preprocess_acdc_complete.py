#!/usr/bin/env python3
"""
ACDC preprocessing script we use for our experiments.

We follow the same two-step pipeline as in the paper:
1. N4 bias correction on the original images → acdc_bias_corr
2. Normalization, resampling and center crop/pad → acdc_bias_corr_cropped
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from skimage import transform


# ACDC-specific parameters from the paper
ACDC_TARGET_RESOLUTION = (1.36719, 1.36719)  # mm
ACDC_IMAGE_SIZE = (192, 192)
N4_THRESHOLD = 0.001
N4_ITERATIONS = 50
N4_FITTING_LEVELS = 4


def n4_bias_correction(image, threshold=N4_THRESHOLD, iterations=N4_ITERATIONS, 
                       fitting_levels=N4_FITTING_LEVELS):
    """
    Run N4 bias field correction on one 3D image.

    Args:
        image: input volume as a numpy array (H, W, D)
        threshold: convergence threshold for N4
        iterations: number of iterations per level
        fitting_levels: number of fitting levels

    Returns:
        corrected: bias-corrected volume as a numpy array
    """
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    # Create N4 bias field corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(threshold)
    corrector.SetMaximumNumberOfIterations([int(iterations)] * fitting_levels)
    
    # Apply correction
    output = corrector.Execute(sitk_image)
    
    # Convert back to numpy
    corrected = sitk.GetArrayFromImage(output)
    
    return corrected


def min_max_normalize(image, min_percentile=1, max_percentile=99):
    """
    Simple percentile-based min-max normalization.

    We clip using two percentiles and then rescale the values
    so that most of the image ends up between 0 and 1.

    Args:
        image: input volume (H, W, D)
        min_percentile: lower percentile used as "min"
        max_percentile: upper percentile used as "max"

    Returns:
        normalized volume with values roughly in [0, 1]
    """
    # Compute percentiles
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    
    # Normalize
    normalized = (image - min_val) / (max_val - min_val + 1e-10)
    
    return normalized


def crop_or_pad_slice(img_slice, target_size):
    """
    Center-crop or zero-pad one 2D slice to a target size.

    Args:
        img_slice: 2D array (H, W)
        target_size: (target_h, target_w)

    Returns:
        2D slice with shape target_size
    """
    target_h, target_w = target_size
    h, w = img_slice.shape
    
    # Create output array
    output = np.zeros(target_size, dtype=img_slice.dtype)
    
    # Compute crop/pad coordinates
    h_start = (h - target_h) // 2
    w_start = (w - target_w) // 2
    h_pad_start = (target_h - h) // 2
    w_pad_start = (target_w - w) // 2
    
    if h > target_h and w > target_w:
        # Crop
        output = img_slice[h_start:h_start + target_h, w_start:w_start + target_w]
    elif h <= target_h and w > target_w:
        # Crop width, pad height
        output[h_pad_start:h_pad_start + h, :] = img_slice[:, w_start:w_start + target_w]
    elif h > target_h and w <= target_w:
        # Crop height, pad width
        output[:, w_pad_start:w_pad_start + w] = img_slice[h_start:h_start + target_h, :]
    else:
        # Pad both dimensions
        output[h_pad_start:h_pad_start + h, w_pad_start:w_pad_start + w] = img_slice
    
    return output


def preprocess_volume(image, mask, pixel_size, target_resolution, target_size, 
                     is_mask=False):
    """
    Preprocess one 3D volume: resampling + crop/pad each slice.

    Args:
        image: 3D volume (H, W, D)
        mask: 3D mask (H, W, D) or None
        pixel_size: original spacing (x, y, z) in mm
        target_resolution: target in-plane spacing (x, y) in mm
        target_size: target slice size (H, W)
        is_mask: if True, we treat the input as a mask (nearest interp)

    Returns:
        processed_volume if mask is None
        or (processed_volume, processed_mask_volume) if mask is given
    """
    # Compute scale vector
    scale_vector = [
        pixel_size[0] / target_resolution[0],
        pixel_size[1] / target_resolution[1]
    ]
    
    # Process each slice
    processed_slices = []
    processed_mask_slices = []
    
    num_slices = image.shape[2]
    
    for slice_idx in range(num_slices):
        # Extract slice
        img_slice = image[:, :, slice_idx]
        
        # Resample
        if is_mask:
            # Nearest neighbor for masks
            resampled_slice = transform.rescale(
                img_slice, scale_vector, order=0,
                preserve_range=True, mode='constant'
            )
        else:
            # Bilinear for images
            resampled_slice = transform.rescale(
                img_slice, scale_vector, order=1,
                preserve_range=True, mode='constant'
            )
        
        # Crop or pad
        final_slice = crop_or_pad_slice(resampled_slice, target_size)
        processed_slices.append(final_slice)
        
        # Process mask if provided
        if mask is not None:
            mask_slice = mask[:, :, slice_idx]
            resampled_mask = transform.rescale(
                mask_slice, scale_vector, order=0,
                preserve_range=True, mode='constant'
            )
            final_mask = crop_or_pad_slice(resampled_mask, target_size)
            processed_mask_slices.append(final_mask)
    
    # Stack slices
    processed_volume = np.stack(processed_slices, axis=2)
    
    if mask is not None:
        processed_mask_volume = np.stack(processed_mask_slices, axis=2)
        return processed_volume, processed_mask_volume
    else:
        return processed_volume


def process_acdc_dataset(input_dir, output_dir_bias, output_dir_cropped, skip_bias_correction=False):
    """
    Run the full preprocessing on the ACDC dataset.

    For each patient we:
      - load the 3D image (and mask if it exists)
      - optionally apply N4 bias correction and save it
      - normalize, resample and crop/pad
      - save the final preprocessed image (and mask)

    Args:
        input_dir: folder with raw ACDC data (patientXXX subfolders)
        output_dir_bias: folder where we save bias-corrected volumes
        output_dir_cropped: folder where we save final preprocessed volumes
        skip_bias_correction: if True, we skip the N4 step
    """
    input_path = Path(input_dir)
    output_bias_path = Path(output_dir_bias)
    output_cropped_path = Path(output_dir_cropped)
    
    # Create output directories
    output_bias_path.mkdir(parents=True, exist_ok=True)
    output_cropped_path.mkdir(parents=True, exist_ok=True)
    
    # Get all patient folders
    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('patient')])
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Check if patient is already processed in bias_corr
        patient_output_bias = output_bias_path / patient_id
        patient_output_cropped = output_cropped_path / patient_id
        
        if (patient_output_bias.exists() and 
            (patient_output_bias / "img_bias_corr.nii.gz").exists()):
            print(f"\n  {patient_id} already processed, skipping...")
            continue
        
        print(f"\nProcessing {patient_id}...")
        
        # Find image and mask files (use frame01 like the original script)
        image_file = None
        mask_file = None
        
        # Look for frame01 files
        for f in patient_dir.glob("*frames01*.nii.gz"):
            if "_gt" not in f.name:
                image_file = f
            elif "_gt" in f.name:
                mask_file = f
        
        if image_file is None:
            print(f"  Warning: No image found for {patient_id}, skipping")
            continue
        
        try:
            # Load image
            nii_image = nib.load(str(image_file))
            image = nii_image.get_fdata()
            pixel_size = nii_image.header['pixdim'][1:4]
            affine = nii_image.affine
            
            # Load mask if available
            mask = None
            if mask_file and mask_file.exists():
                nii_mask = nib.load(str(mask_file))
                mask = nii_mask.get_fdata()
            
            # Step 1: Bias correction
            patient_output_bias = output_bias_path / patient_id
            patient_output_bias.mkdir(exist_ok=True)
            
            bias_corr_image = None
            if not skip_bias_correction:
                print(f"  Applying N4 bias correction...")
                bias_corr_image = n4_bias_correction(image)
                
                # Save bias-corrected image
                bias_corr_nii = nib.Nifti1Image(bias_corr_image, affine)
                bias_corr_path = patient_output_bias / "img_bias_corr.nii.gz"
                nib.save(bias_corr_nii, bias_corr_path)
                print(f"  Saved bias-corrected image to {bias_corr_path}")
            else:
                bias_corr_image = image
            
            # Step 2: Normalization
            print(f"  Normalizing...")
            normalized_image = min_max_normalize(bias_corr_image)
            
            # Step 3: Preprocess (resample + crop/pad)
            print(f"  Resampling and cropping...")
            if mask is not None:
                preprocessed_image, preprocessed_mask = preprocess_volume(
                    normalized_image, mask, pixel_size,
                    ACDC_TARGET_RESOLUTION, ACDC_IMAGE_SIZE, is_mask=False
                )
            else:
                preprocessed_image = preprocess_volume(
                    normalized_image, None, pixel_size,
                    ACDC_TARGET_RESOLUTION, ACDC_IMAGE_SIZE, is_mask=False
                )
                preprocessed_mask = None
            
            # Save preprocessed data
            patient_output_cropped = output_cropped_path / patient_id
            patient_output_cropped.mkdir(exist_ok=True)
            
            # Update affine for new resolution
            new_affine = affine.copy()
            new_affine[0, 0] = -ACDC_TARGET_RESOLUTION[0]
            new_affine[1, 1] = -ACDC_TARGET_RESOLUTION[1]
            
            # Save preprocessed image
            preprocessed_nii = nib.Nifti1Image(preprocessed_image, new_affine)
            img_cropped_path = patient_output_cropped / "img_cropped.nii.gz"
            nib.save(preprocessed_nii, img_cropped_path)
            print(f"  Saved preprocessed image to {img_cropped_path}")
            
            # Save mask if available
            if preprocessed_mask is not None:
                mask_cropped_path = patient_output_cropped / "mask_cropped.nii.gz"
                preprocessed_mask_nii = nib.Nifti1Image(
                    preprocessed_mask.astype(np.int16), new_affine
                )
                nib.save(preprocessed_mask_nii, mask_cropped_path)
                print(f"  Saved preprocessed mask to {mask_cropped_path}")
            
        except Exception as e:
            print(f"  Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="ACDC preprocessing: N4 bias correction + normalization + resampling + cropping."
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='database/msd_heart',
        help='Input directory containing raw ACDC data (patient folders)'
    )
    
    parser.add_argument(
        '--output_bias_dir',
        type=str,
        default='database/msd_heart_bias_corr',
        help='Output directory for bias-corrected images'
    )
    
    parser.add_argument(
        '--output_cropped_dir',
        type=str,
        default='database/msd_heart_bias_corr_cropped',
        help='Output directory for preprocessed images'
    )
    
    parser.add_argument(
        '--skip_bias',
        action='store_true',
        help='Skip N4 bias correction (if already done)'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run preprocessing
    print("="*60)
    print("COMPLETE ACDC PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Input: {args.input_dir}")
    print(f"Bias-corrected output: {args.output_bias_dir}")
    print(f"Preprocessed output: {args.output_cropped_dir}")
    if args.skip_bias:
        print("⚠️  Skipping N4 bias correction")
    print("="*60)
    
    process_acdc_dataset(
        args.input_dir,
        args.output_bias_dir,
        args.output_cropped_dir,
        skip_bias_correction=args.skip_bias
    )


if __name__ == '__main__':
    main()