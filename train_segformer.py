"""
Advanced SegFormer-B2 Training Script - RESUME VERSION (Epoch 5-14)
Features: Dynamic Weight Tuning, Cosine LR Scheduler, Weighted OHEM, Destructive Augment Fix.
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
import random
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig, get_cosine_schedule_with_warmup

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# Mask Conversion & Visualization Utils
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
# Dynamic Weight Manager
# ============================================================================

class DynamicWeightManager:
    def __init__(self, n_classes, momentum=0.5):
        self.n_classes = n_classes
        self.momentum = momentum
        # Background, Trees, Lush, Grass, DryB, Clutter, Logs, Rocks, Land, Sky
        self.occupancy = np.array([0.25, 0.15, 0.10, 0.10, 0.05, 0.05, 0.02, 0.03, 0.20, 0.05])
        self.current_weights = torch.ones(n_classes).cuda()

    def update_weights(self, val_ious):
        safe_ious = np.nan_to_num(val_ious, nan=0.0)
        # Performance factor: 1 - IoU (Lower IoU -> Higher Weight)
        perf_factor = 1.0 - safe_ious + 0.1
        # Occupancy factor: 1 / sqrt(freq)
        occ_factor = 1.0 / np.sqrt(self.occupancy + 1e-6)
        
        new_weights = occ_factor * perf_factor
        new_weights = new_weights / new_weights.mean() # Normalize
        
        updated = (self.momentum * self.current_weights.cpu().numpy()) + ((1 - self.momentum) * new_weights)
        self.current_weights = torch.from_numpy(np.clip(updated, 0.1, 10.0)).float().cuda()
        print(f"\n[Dynamic Weights Updated] -> {np.round(self.current_weights.cpu().numpy(), 2)}")
        return self.current_weights

# ============================================================================
# Advanced Losses: Weighted Tversky + Weighted OHEM
# ============================================================================

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets, weights):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()
        tp = (probs * targets_one_hot).sum(dim=(2, 3))
        fp = (probs * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets_one_hot).sum(dim=(2, 3))
        tp = tp * weights.view(1, -1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()

class OHEMCrossEntropy(nn.Module):
    def __init__(self, thresh=0.7, min_kept=100000):
        super(OHEMCrossEntropy, self).__init__()
        self.thresh, self.min_kept = thresh, min_kept

    def forward(self, logits, targets, weights):
        loss = F.cross_entropy(logits, targets, weight=weights, reduction='none').view(-1)
        if self.min_kept < loss.numel():
            val, ind = loss.topk(int(self.thresh * loss.numel()))
            return val.mean()
        return loss.mean()

class AdvancedCombinedLoss(nn.Module):
    def __init__(self):
        super(AdvancedCombinedLoss, self).__init__()
        self.ohem, self.tversky = OHEMCrossEntropy(), TverskyLoss()

    def forward(self, logits, targets, weights):
        return 0.5 * self.ohem(logits, targets, weights) + 0.5 * self.tversky(logits, targets, weights)

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
            image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(image)
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

def evaluate_metrics_with_tta(model, loader, device, loss_fct, weights):
    all_ious, all_accs, losses = [], [], []
    model.eval()
    pbar = tqdm(loader, desc="Evaluating (TTA)", leave=False)
    with torch.no_grad():
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits1 = model(imgs).logits
            logits1 = F.interpolate(logits1, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            imgs_flipped = torch.flip(imgs, dims=[3])
            logits2 = model(imgs_flipped).logits
            logits2 = F.interpolate(logits2, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            logits2 = torch.flip(logits2, dims=[3])
            outputs = (logits1 + logits2) / 2.0
            losses.append(loss_fct(outputs, labels, weights).item())
            ious, accs = get_metrics(outputs, labels)
            all_ious.append(ious); all_accs.append(accs)
    model.train()
    return np.nanmean(losses), np.nanmean(all_ious, axis=0), np.nanmean(all_accs, axis=0)

def save_all_results(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 10))
    for i, m in enumerate(['loss', 'iou', 'acc']):
        plt.subplot(2, 2, i+1); plt.plot(history[f'train_{m}'], label='Train'); plt.plot(history[f'val_{m}'], label='Val')
        plt.title(f'Mean {m.upper()}'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mean_metrics.png')); plt.close()
    class_dir = os.path.join(output_dir, 'classes')
    os.makedirs(class_dir, exist_ok=True)
    for c in range(n_classes):
        plt.figure(); plt.plot([h[c] for h in history['train_class_iou']], label='Train'); plt.plot([h[c] for h in history['val_class_iou']], label='Val')
        plt.title(f'IoU: {class_names[c]}'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(class_dir, f'class_{c}_{class_names[c]}.png')); plt.close()
    with open(os.path.join(output_dir, 'detailed_log.txt'), 'w') as f:
        for e in range(len(history['train_loss'])):
            f.write(f"EPOCH {e+Start_E}\n" + "-"*30 + f"\nTrain Loss: {history['train_loss'][e]:.4f} | Val Loss: {history['val_loss'][e]:.4f}\n")
            f.write(f"{'Class':<18} | {'T-IoU':<8} | {'V-IoU':<8} | {'V-Acc':<8}\n")
            for c in range(n_classes):
                # Using explicit variables for clarity and safety
                ti = history['train_class_iou'][e][c]
                vi = history['val_class_iou'][e][c]
                va = history['val_class_acc'][e][c]
                f.write(f"{class_names[c]:<18} | {ti:.3f} | {vi:.3f} | {va:.3f}\n")
            f.write("="*50 + "\n\n")

# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    w, h = 960, 540 
    batch_size, acc_steps, base_lr = 1, 4, 1e-3
    start_epoch, end_epoch = 1, 11
    global Start_E; Start_E = start_epoch + 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_segformer_adv')
    weights_dir = os.path.join(script_dir, 'weights_segformer_adv')
    samples_dir = os.path.join(output_dir, 'samples')
    for d in [output_dir, weights_dir, samples_dir]: os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)])

    train_loader = DataLoader(MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/train'), transform, mask_transform, True), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/val'), transform, mask_transform, False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Config - Optimized for Local Caching
    try:
        config = SegformerConfig.from_pretrained("nvidia/mit-b2", local_files_only=True)
    except Exception:
        print("Config not found locally. Downloading from HuggingFace...")
        config = SegformerConfig.from_pretrained("nvidia/mit-b2", local_files_only=False)
        
    config.num_labels, config.hidden_dropout_prob, config.attention_probs_dropout_prob, config.classifier_dropout_prob = n_classes, 0.2, 0.2, 0.2
    # Initialize SegFormer B2 - Optimized for Local Caching
    try:
        # Try loading from local cache first (Skips internet check)
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", 
            config=config, 
            ignore_mismatched_sizes=True, 
            use_safetensors=True,
            local_files_only=True
        ).to(device)
        print("Loaded model from local cache.")
    except Exception:
        # Fallback to download if local files are missing
        print("Model not found locally. Downloading from HuggingFace...")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2", 
            config=config, 
            ignore_mismatched_sizes=True, 
            use_safetensors=True,
            local_files_only=False
        ).to(device)
    
    if start_epoch > 0:
        weight_path = os.path.join(weights_dir, f"epoch_{start_epoch}.pth")
        if os.path.exists(weight_path):
            print(f"Loading weights from {weight_path}...")
            model.load_state_dict(torch.load(weight_path))
        else:
            print(f"Warning: {weight_path} not found. Starting from scratch.")
    else:
        print("Starting from scratch (Epoch 0).")

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    
    # 4. Added Scheduler with Warmup
    num_training_steps = len(train_loader) * (end_epoch - start_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=num_training_steps)
    loss_fct = AdvancedCombinedLoss().to(device)
    weight_manager = DynamicWeightManager(n_classes)
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss':[], 'val_loss':[], 'train_iou':[], 'val_iou':[], 'train_acc':[], 'val_acc':[], 'train_class_iou':[], 'val_class_iou':[], 'val_class_acc':[]}
    best_iou = 0.0

    print(f"\nResuming Training from Epoch {start_epoch+1}...")
    for epoch in range(start_epoch, end_epoch):
        model.train(); train_losses, train_ious, train_accs = [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        # Get dynamic weights from previous epoch performance (or initial)
        current_weights = weight_manager.current_weights

        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = F.interpolate(model(imgs).logits, size=imgs.shape[2:], mode="bilinear")
                loss = loss_fct(outputs, labels, current_weights) / acc_steps
            scaler.scale(loss).backward()
            if (i + 1) % acc_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            with torch.no_grad():
                ious, accs = get_metrics(outputs, labels)
                train_ious.append(ious); train_accs.append(accs); train_losses.append(loss.item() * acc_steps)
            pbar.set_postfix(loss=f"{loss.item()*acc_steps:.4f}", iou=f"{np.nanmean(ious):.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        v_loss, v_class_ious, v_class_accs = evaluate_metrics_with_tta(model, val_loader, device, loss_fct, current_weights)
        # Update weights for NEXT epoch based on THIS validation
        current_weights = weight_manager.update_weights(v_class_ious)

        t_loss, t_class_ious, t_class_accs = np.mean(train_losses), np.nanmean(train_ious, axis=0), np.nanmean(train_accs, axis=0)
        os.makedirs(os.path.join(samples_dir, f"epoch_{epoch+1}"), exist_ok=True)
        for i, idx in enumerate(random.sample(range(len(val_dataset)), 5)):
            img, mask, d_id = val_dataset[idx]
            with torch.no_grad(): pred = torch.argmax(F.interpolate(model(img.unsqueeze(0).to(device)).logits, size=(h,w), mode="bilinear")[0], 0)
            save_prediction_comparison(img, mask, pred, os.path.join(samples_dir, f"epoch_{epoch+1}", f"s_{i}_{d_id}"), d_id)

        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_iou'].append(np.nanmean(t_class_ious)); history['val_iou'].append(np.nanmean(v_class_ious))
        history['train_acc'].append(np.nanmean(t_class_accs)); history['val_acc'].append(np.nanmean(v_class_accs))
        history['train_class_iou'].append(t_class_ious); history['val_class_iou'].append(v_class_ious); history['val_class_acc'].append(v_class_accs)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"{'Class':<18} | {'T-IoU':<8} | {'V-IoU':<8} | {'V-Acc':<8}")
        for c in range(n_classes): 
            print(f"{class_names[c]:<18} | {t_class_ious[c]:.3f} | {v_class_ious[c]:.3f} | {v_class_accs[c]:.3f}")
        
        # Calculate overall means
        mean_t_iou = np.nanmean(t_class_ious)
        mean_v_iou = np.nanmean(v_class_ious)
        mean_v_acc = np.nanmean(v_class_accs)
        
        print("-" * 50)
        print(f"MEAN {'':<13} | {mean_t_iou:.3f}    | {mean_v_iou:.3f}    | {mean_v_acc:.3f}")
        print("=" * 50)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))
        if history['val_iou'][-1] > best_iou:
            best_iou = history['val_iou'][-1]
            torch.save(model.state_dict(), os.path.join(script_dir, "segformer_adv_best.pth"))

    save_all_results(history, output_dir)

if __name__ == "__main__": main()
