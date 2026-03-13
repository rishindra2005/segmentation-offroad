"""
Test Dataset Class Analyzer
Scans ground truth masks in the test directory to verify class presence.
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def main():
    test_mask_dir = "Offroad_Segmentation_testImages/Segmentation"
    
    if not os.path.exists(test_mask_dir):
        print(f"Error: Directory not found -> {test_mask_dir}")
        return

    mask_files = list(Path(test_mask_dir).glob("*.png"))
    print(f"Analyzing {len(mask_files)} masks in {test_mask_dir}...")

    # Tracking variables
    global_pixel_counts = np.zeros(len(class_names), dtype=np.int64)
    image_presence_counts = np.zeros(len(class_names), dtype=np.int64)

    for mask_path in tqdm(mask_files):
        mask = np.array(Image.open(mask_path))
        
        # Count unique values in this image
        unique_vals = np.unique(mask)
        
        # Check presence and pixels
        for raw_val, class_id in value_map.items():
            pixels = np.sum(mask == raw_val)
            if pixels > 0:
                global_pixel_counts[class_id] += pixels
                image_presence_counts[class_id] += 1

    # Print Results
    print("\n" + "="*60)
    print(f"{'Class Name':<18} | {'Images Contained':<15} | {'Total Pixels'}")
    print("-" * 60)
    
    total_pixels_all = np.sum(global_pixel_counts)
    
    for i in range(len(class_names)):
        presence = image_presence_counts[i]
        pixels = global_pixel_counts[i]
        percentage = (pixels / total_pixels_all) * 100 if total_pixels_all > 0 else 0
        
        status = "PRESENT" if presence > 0 else "MISSING !!!"
        
        print(f"{class_names[i]:<18} | {presence:<15} | {pixels:<12} ({percentage:.2f}%) -> {status}")

    print("-" * 60)
    print(f"Total Images: {len(mask_files)}")
    print("="*60)
    print("\nNOTE: If a class says 'MISSING', the 'nan' result in your test script is normal.")
    print("If a class says 'PRESENT' but your test script showed 'nan', your model is completely failing to predict it.")

if __name__ == "__main__":
    main()
