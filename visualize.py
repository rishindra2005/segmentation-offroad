import cv2
import numpy as np
import os
from pathlib import Path
import random

# Input folders
mask_folder = "Offroad_Segmentation_Training_Dataset/train/Segmentation" 
image_folder = "Offroad_Segmentation_Training_Dataset/train/Color_Images"
# Output folder for comparison images
output_folder = "output_comparison"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all mask files
image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
mask_files = [f for f in Path(mask_folder).iterdir() 
               if f.is_file() and f.suffix.lower() in image_extensions]

print(f"Found {len(mask_files)} mask files in {mask_folder}")

# Take a sample of 40 images
sample_size = min(40, len(mask_files))
sampled_masks = random.sample(mask_files, sample_size)
print(f"Sampling {sample_size} images for processing")

# Dictionary to store color mappings (value -> color)
color_map = {}

# Process each file
for mask_file in sorted(sampled_masks):
    # Find corresponding color image
    img_file = Path(image_folder) / mask_file.name

    if not img_file.exists():
        print(f"  Skipped: Color image not found for {mask_file.name}")
        continue

    print(f"Processing: {mask_file.name}")

    # Read the mask and original image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)

    if mask is None or img is None:
        print(f"  Skipped: Could not read mask or image for {mask_file.name}")
        continue

    # Get unique values in mask
    u = np.unique(mask)

    # Create colorized mask (3 channels)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Assign colors to each unique value
    for v in u:
        if v not in color_map:
            # Generate new random color for this value
            color_map[v] = np.random.randint(0, 255, (3,), dtype=np.uint8)
        color_mask[mask == v] = color_map[v]

    # Ensure images are the same size for concatenation
    if img.shape[:2] != mask.shape[:2]:
        color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create overlaid image (3 channels)
    alpha = 0.6
    beta = 0.4
    overlaid = cv2.addWeighted(img, alpha, color_mask, beta, 0)

    # Concatenate Original, Colorized Mask, and Overlaid side-by-side
    comparison = np.hstack((img, color_mask, overlaid))

    # Save the comparison image
    output_path = os.path.join(output_folder, f"{mask_file.stem}_comparison.png")
    cv2.imwrite(output_path, comparison)

    # Print channel info for the first processed image
    if 'channels_reported' not in locals():
        print(f"  Single image shape: {img.shape}")
        print(f"  Combined image shape: {comparison.shape}")
        print(f"  Number of channels in output: {comparison.shape[2]}")
        channels_reported = True

print(f"\nProcessing complete! Comparison images saved to: {output_folder}")
print(f"Total unique mask values found: {len(color_map)}")