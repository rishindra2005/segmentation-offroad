import cv2
import numpy as np
from pathlib import Path

# Path to a sample mask
mask_path = "Offroad_Segmentation_Training_Dataset/train/Segmentation/cc0000012.png"

if not Path(mask_path).exists():
    print(f"Error: Mask not found at {mask_path}")
else:
    # Read the mask
    # IMREAD_UNCHANGED is crucial to get the raw 1-channel data
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # 1. Show Channels
    shape = mask.shape
    channels = 1 if len(shape) == 2 else shape[2]
    
    # 2. Show Classes
    unique_classes = np.unique(mask)
    
    print("-" * 30)
    print(f"FILE: {mask_path}")
    print("-" * 30)
    print(f"Image Shape:    {shape}")
    print(f"No. of Channels: {channels}")
    print(f"Unique Classes:  {unique_classes}")
    print(f"No. of Classes:  {len(unique_classes)}")
    print("-" * 30)
    print("\nTERMINOLOGY NOTE:")
    print("Channels: The 'layers' of the file (1 = Grayscale/ID mask, 3 = RGB color).")
    print("Classes:  The different integer values (0, 1, 2...) representing objects like Road, Grass, etc.")
