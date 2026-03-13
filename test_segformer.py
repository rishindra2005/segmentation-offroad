"""
SegFormer-B2 Inference/Test Script
Evaluates SegFormer weights on test images and saves visualizations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import random

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# Utils & Mapping
# ============================================================================

value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
n_classes = len(value_map)
color_palette = np.array([[0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43], [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]], dtype=np.uint8)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    img = img_tensor.cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = np.clip((img * std + mean), 0, 1)
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(gt_color); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title('Prediction'); axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform, self.mask_transform = transform, mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask_path = os.path.join(self.masks_dir, data_id)
        if os.path.exists(mask_path):
            mask = convert_mask(Image.open(mask_path))
        else:
            mask = Image.new('L', image.size, 0) # Dummy mask if missing
        
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        return image, mask, data_id

# ============================================================================
# Metrics
# ============================================================================

def get_metrics(outputs, labels):
    pred = torch.argmax(outputs, dim=1)
    pred_flat, target_flat = pred.view(-1), labels.view(-1)
    ious, accs = [], []
    for cls in range(n_classes):
        mask = (target_flat == cls)
        if mask.sum() == 0: ious.append(float('nan')); accs.append(float('nan'))
        else:
            inter = ((pred_flat == cls) & mask).sum().float()
            union = ((pred_flat == cls) | mask).sum().float()
            ious.append((inter/union).cpu().item())
            accs.append(((pred_flat[mask] == cls).sum().float() / mask.sum()).cpu().item())
    return ious, accs

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='segformer_v3_best.pth')
    parser.add_argument('--data_dir', type=str, default='Offroad_Segmentation_testImages')
    parser.add_argument('--output_dir', type=str, default='./predictions_segformer')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--tta', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['masks', 'masks_color', 'comparisons']: os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    w, h = 960, 540 
    transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)])

    testset = MaskDataset(args.data_dir, transform, mask_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Loading weights from {args.model_path}...")
    config = SegformerConfig.from_pretrained("nvidia/mit-b2", local_files_only=True)
    config.num_labels = n_classes
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", config=config, ignore_mismatched_sizes=True, use_safetensors=True, local_files_only=True).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_ious, all_accs = [], []
    print(f"Running inference on {len(testset)} images with Multiscale TTA (4 passes per image)...")
    
    # 1. Select 20 random indices for comparison saving
    num_samples_to_save = 20
    sample_indices = random.sample(range(len(testset)), min(num_samples_to_save, len(testset)))
    current_idx = 0

    # TTA Settings: 1.0x and 1.25x scales
    tta_scales = [1.0, 1.25]

    with torch.no_grad():
        for imgs, labels, data_ids in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            B, C, H, W = imgs.shape
            
            # Initialize accumulator for logits
            final_logits = torch.zeros((B, n_classes, H, W), device=device)
            
            if args.tta:
                for scale in tta_scales:
                    # Scale image if needed
                    if scale != 1.0:
                        scaled_h, scaled_w = int(H * scale), int(W * scale)
                        # Ensure dimensions are divisible by 14 for SegFormer
                        scaled_h = (scaled_h // 14) * 14
                        scaled_w = (scaled_w // 14) * 14
                        imgs_input = F.interpolate(imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                    else:
                        imgs_input = imgs
                    
                    # Pass 1: Normal
                    out = model(imgs_input).logits
                    final_logits += F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                    
                    # Pass 2: Horizontal Flip
                    out_f = model(torch.flip(imgs_input, dims=[3])).logits
                    out_f = torch.flip(out_f, dims=[3]) # Flip back
                    final_logits += F.interpolate(out_f, size=(H, W), mode='bilinear', align_corners=False)
                
                # Average the passes (2 scales * 2 flips = 4 passes)
                logits = final_logits / (len(tta_scales) * 2)
            else:
                # Standard single-pass inference
                logits = model(imgs).logits
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

            predicted_masks = torch.argmax(logits, dim=1)
            
            # Metrics
            ious, accs = get_metrics(logits, labels)
            all_ious.append(ious); all_accs.append(accs)

            # Save Results
            for i in range(imgs.shape[0]):
                d_id = data_ids[i]
                name = os.path.splitext(d_id)[0]
                pred_np = predicted_masks[i].cpu().numpy().astype(np.uint8)
                
                # 1. Raw mask (FAST)
                Image.fromarray(pred_np).save(os.path.join(args.output_dir, 'masks', f'{name}_pred.png'))
                # 2. Color mask (FAST)
                color_pred = mask_to_color(pred_np)
                cv2.imwrite(os.path.join(args.output_dir, 'masks_color', f'{name}_color.png'), cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
                
                # 3. Comparison (Side-by-side) - ONLY FOR SAMPLES (SLOW)
                if current_idx in sample_indices:
                    save_prediction_comparison(imgs[i], labels[i], predicted_masks[i], os.path.join(args.output_dir, 'comparisons', f'{name}_comp.png'), d_id)
                
                current_idx += 1

    # Summary
    mean_ious = np.nanmean(all_ious, axis=0)
    print("\n" + "="*30 + "\nTEST RESULTS\n" + "="*30)
    for i in range(n_classes):
        print(f"{class_names[i]:<18}: {mean_ious[i]:.4f}")
    print("-" * 30)
    print(f"MEAN IoU: {np.nanmean(mean_ious):.4f}")
    print("="*30)

if __name__ == "__main__": main()
