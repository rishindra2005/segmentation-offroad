"""
Segmentation Training Script - COMPREHENSIVE VERSION
Includes: Data Augmentation, Per-Class Metrics, Live Tables, Per-Epoch Samples, and Detailed Plotting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import torchvision
import random
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# Mask Conversion & Visualization Utils
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

# Class names for visualization
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
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
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform, self.mask_transform, self.augment = transform, mask_transform, augment
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))
        if self.augment:
            if random.random() > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)
        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255
        return image, mask, data_id

# ============================================================================
# Model & Metrics
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=7, padding=3), nn.GELU())
        self.block = nn.Sequential(nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128), nn.GELU(), nn.Conv2d(128, 128, kernel_size=1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

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

def evaluate_metrics(model, backbone, loader, device):
    all_ious, all_accs, losses = [], [], []
    model.eval()
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")
            labels_long = labels.squeeze(1).long()
            losses.append(F.cross_entropy(outputs, labels_long).item())
            ious, accs = get_metrics(outputs, labels_long)
            all_ious.append(ious); all_accs.append(accs)
    model.train()
    return np.nanmean(losses), np.nanmean(all_ious, axis=0), np.nanmean(all_accs, axis=0)

# ============================================================================
# Plotting & Logging
# ============================================================================

def save_all_results(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Mean plots
    plt.figure(figsize=(15, 10))
    for i, m in enumerate(['loss', 'iou', 'acc']):
        plt.subplot(2, 2, i+1); plt.plot(history[f'train_{m}'], label='Train'); plt.plot(history[f'val_{m}'], label='Val')
        plt.title(f'Mean {m.upper()}'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_metrics.png')); plt.close()
    
    # Per-class IoU plots
    class_dir = os.path.join(output_dir, 'classes')
    os.makedirs(class_dir, exist_ok=True)
    for c in range(n_classes):
        plt.figure(); plt.plot([h[c] for h in history['train_class_iou']], label='Train'); plt.plot([h[c] for h in history['val_class_iou']], label='Val')
        plt.title(f'IoU: {class_names[c]}'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(class_dir, f'class_{c}_{class_names[c]}.png')); plt.close()

    # Log file
    with open(os.path.join(output_dir, 'detailed_log.txt'), 'w') as f:
        for e in range(len(history['train_loss'])):
            f.write(f"EPOCH {e+1}\n" + "-"*30 + f"\nTrain Loss: {history['train_loss'][e]:.4f} | Val Loss: {history['val_loss'][e]:.4f}\n")
            f.write(f"{'Class':<18} | {'T-IoU':<8} | {'V-IoU':<8} | {'V-Acc':<8}\n")
            for c in range(n_classes):
                f.write(f"{class_names[c]:<18} | {history['train_class_iou'][e][c]:.3f} | {history['val_class_iou'][e][c]:.3f} | {history['val_class_acc'][e][c]:.3f}\n")
            f.write("="*50 + "\n\n")

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w= int(((960 / 4) // 14) * 14)
    h= int(((540 / 4) // 14) * 14)
    batch_size, w, h, lr, n_epochs = 16, w, h, 1e-4, 10
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    weights_dir = os.path.join(script_dir, 'weights')
    samples_dir = os.path.join(output_dir, 'samples')
    for d in [output_dir, weights_dir, samples_dir]: os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])

    train_loader = DataLoader(MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/train'), transform, mask_transform, True), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/val'), transform, mask_transform, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
    classifier = SegmentationHeadConvNeXt(384, n_classes, w//14, h//14).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    loss_fct = nn.CrossEntropyLoss()

    history = {'train_loss':[], 'val_loss':[], 'train_iou':[], 'val_iou':[], 'train_acc':[], 'val_acc':[], 'train_class_iou':[], 'val_class_iou':[], 'val_class_acc':[]}
    best_iou = 0.0

    for epoch in range(n_epochs):
        classifier.train()
        train_losses, train_ious, train_accs = [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
            
            logits = classifier(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")
            labels_long = labels.squeeze(1).long()
            
            loss = loss_fct(outputs, labels_long)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Calculate training metrics on the fly (for the progress bar/history)
            with torch.no_grad():
                batch_ious, batch_accs = get_metrics(outputs, labels_long)
                train_ious.append(batch_ious)
                train_accs.append(batch_accs)
                train_losses.append(loss.item())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{np.nanmean(batch_ious):.3f}")

        # Evaluation (Validation Only - Much faster than training set)
        print(f"\nEvaluating Validation Set...")
        v_loss, v_class_ious, v_class_accs = evaluate_metrics(classifier, backbone, val_loader, device)
        
        # Training metrics average
        t_loss = np.mean(train_losses)
        t_class_ious = np.nanmean(train_ious, axis=0)
        t_class_accs = np.nanmean(train_accs, axis=0)
        
        # Save 5 samples
        print(f"Saving visual samples...")
        os.makedirs(os.path.join(samples_dir, f"epoch_{epoch+1}"), exist_ok=True)
        for i, idx in enumerate(random.sample(range(len(val_dataset)), 5)):
            img, mask, d_id = val_dataset[idx]
            with torch.no_grad(): 
                out = F.interpolate(classifier(backbone.forward_features(img.unsqueeze(0).to(device))["x_norm_patchtokens"]), size=(h,w), mode="bilinear")
                pred = torch.argmax(out[0], 0)
            save_prediction_comparison(img, mask.squeeze(0).long(), pred, os.path.join(samples_dir, f"epoch_{epoch+1}", f"s_{i}_{d_id}"), d_id)

        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_iou'].append(np.nanmean(t_class_ious)); history['val_iou'].append(np.nanmean(v_class_ious))
        history['train_acc'].append(np.nanmean(t_class_accs)); history['val_acc'].append(np.nanmean(v_class_accs))
        history['train_class_iou'].append(t_class_ious); history['val_class_iou'].append(v_class_ious); history['val_class_acc'].append(v_class_accs)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"{'Class':<18} | {'T-IoU':<8} | {'V-IoU':<8} | {'V-Acc':<8}")
        for c in range(n_classes): print(f"{class_names[c]:<18} | {t_class_ious[c]:.3f} | {v_class_ious[c]:.3f} | {v_class_accs[c]:.3f}")
        print(f"MEAN {'':<13} | {np.nanmean(t_class_ious):.3f} | {np.nanmean(v_class_ious):.3f} | {np.nanmean(v_class_accs):.3f}")

        torch.save(classifier.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))
        if history['val_iou'][-1] > best_iou:
            best_iou = history['val_iou'][-1]
            torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head_best.pth"))
            print(f"*** New Best IoU: {best_iou:.4f} Saved ***")

    save_all_results(history, output_dir)
    print("\nTraining Complete! Check 'train_stats/' for all logs, graphs, and samples.")

if __name__ == "__main__": main()
