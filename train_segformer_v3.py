"""
SegFormer-B2 Training Script V3 - SHOCK THERAPY (Scratch Run)
Focus: Lovasz-Softmax for direct IoU optimization, Extreme Augmentation, and Weighted Sampling.
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
# Radical Loss: Lovasz-Softmax (Direct IoU Optimization)
# ============================================================================

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] probabilities at each pixel
      labels: [P] ground truth labels
    """
    if probas.numel() == 0: return probas * 0.
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0: continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        gt_sorted = fg[perm]
        grad = lovasz_grad(gt_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.stack(losses).mean()

def boundary_loss(pred, target):
    target_float = target.unsqueeze(1).float()
    kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).cuda().view(1,1,3,3)
    with torch.no_grad():
        gt_edges = F.conv2d(target_float, kernel, padding=1).abs() > 0.1
    pred_edges = F.conv2d(F.softmax(pred, dim=1), kernel.repeat(n_classes, 1, 1, 1), padding=1, groups=n_classes).abs()
    return F.mse_loss(pred_edges, gt_edges.float().repeat(1, n_classes, 1, 1))

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-6):
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

class RadicalV3Loss(nn.Module):
    def __init__(self):
        super(RadicalV3Loss, self).__init__()
        self.tversky = TverskyLoss()

    def forward(self, logits, targets, weights):
        probas = F.softmax(logits, dim=1)
        # Flatten B,H,W into a single pixel dimension P
        P_probas = probas.permute(0, 2, 3, 1).reshape(-1, n_classes)
        P_targets = targets.view(-1)
        
        # 1. Lovasz Loss (Optimizes IoU directly)
        l_lovasz = lovasz_softmax_flat(P_probas, P_targets)
        # 2. Weighted Tversky
        l_tversky = self.tversky(logits, targets, weights)
        # 3. Boundary refinement
        l_bound = boundary_loss(logits, targets)
        return 0.5 * l_lovasz + 0.3 * l_tversky + 0.2 * l_bound

# ============================================================================
# Dynamic Weight Manager
# ============================================================================

class DynamicWeightManager:
    def __init__(self, n_classes, momentum=0.4):
        self.n_classes = n_classes
        self.momentum = momentum
        self.occupancy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.current_weights = torch.ones(n_classes).cuda()

    def update_weights(self, val_ious):
        safe_ious = np.nan_to_num(val_ious, nan=0.0)
        perf_factor = np.maximum(0.1, 0.8 - safe_ious)
        new_weights = perf_factor / perf_factor.mean()
        updated = (self.momentum * self.current_weights.cpu().numpy()) + ((1 - self.momentum) * new_weights)
        self.current_weights = torch.from_numpy(np.clip(updated, 0.1, 25.0)).float().cuda()
        print(f"\n[Radical Weights Updated] -> {np.round(self.current_weights.cpu().numpy(), 2)}")
        return self.current_weights

# ============================================================================
# Dataset & Main Logic
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False, crop_size=(532, 952)):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform, self.mask_transform, self.augment = transform, mask_transform, augment
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.crop_size = crop_size
        
        self.presence_cache = os.path.join(data_dir, "class_presence.npy")
        if os.path.exists(self.presence_cache):
            self.image_contents = np.load(self.presence_cache)
        else:
            print("Analyzing masks for weighted pipeline...")
            self.image_contents = np.zeros((len(self.data_ids), n_classes), dtype=np.float32)
            for i, d_id in enumerate(tqdm(self.data_ids)):
                m = np.array(Image.open(os.path.join(self.masks_dir, d_id)))
                for val in np.unique(m):
                    if val in value_map: self.image_contents[i, value_map[val]] = 1.0
            np.save(self.presence_cache, self.image_contents)

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))
        if self.augment:
            if random.random() < 0.2: image = TF.to_grayscale(image, num_output_channels=3)
            if random.random() > 0.5:
                angle = random.uniform(-40, 40)
                image, mask = TF.rotate(image, angle), TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.3, 1.0), ratio=(0.75, 1.33))
            image, mask = TF.resized_crop(image, i, j, h, w, size=self.crop_size), TF.resized_crop(mask, i, j, h, w, size=self.crop_size, interpolation=transforms.InterpolationMode.NEAREST)
            image = transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.15)(image)
            if random.random() > 0.5: image, mask = TF.hflip(image), TF.hflip(mask)
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(np.array(self.mask_transform(mask))).long()
        return image, mask, data_id

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
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Evaluating (TTA)", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits1 = F.interpolate(model(imgs).logits, size=imgs.shape[2:], mode="bilinear")
            imgs_f = torch.flip(imgs, dims=[3])
            logits2 = torch.flip(F.interpolate(model(imgs_f).logits, size=imgs.shape[2:], mode="bilinear"), dims=[3])
            outputs = (logits1 + logits2) / 2.0
            losses.append(loss_fct(outputs, labels, weights).item())
            ious, accs = get_metrics(outputs, labels)
            all_ious.append(ious); all_accs.append(accs)
    model.train()
    return np.nanmean(losses), np.nanmean(all_ious, axis=0), np.nanmean(all_accs, axis=0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    w, h = 952, 532 
    batch_size, acc_steps, base_lr = 1, 4, 1e-3
    n_epochs = 15

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_v3_pipe')
    weights_dir = os.path.join(script_dir, 'weights_v3_pipe')
    samples_dir = os.path.join(output_dir, 'samples')
    for d in [output_dir, weights_dir, samples_dir]: os.makedirs(d, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((h, w)), # Explicitly resize all inputs
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST) # Explicitly resize all masks
    ])

    train_dataset = MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/train'), transform, mask_transform, True, crop_size=(h,w))
    val_dataset = MaskDataset(os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset/val'), transform, mask_transform, False, crop_size=(h,w))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    config = SegformerConfig.from_pretrained("nvidia/mit-b2", local_files_only=True)
    config.num_labels, config.hidden_dropout_prob, config.attention_probs_dropout_prob, config.classifier_dropout_prob = n_classes, 0.2, 0.2, 0.2
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", config=config, ignore_mismatched_sizes=True, use_safetensors=True, local_files_only=True).to(device)
    
    # SCRATCH TRAINING
    print("Starting fresh SegFormer training from scratch...")

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    num_steps_per_epoch = 6000 // (batch_size * acc_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_steps_per_epoch, num_training_steps=num_steps_per_epoch*n_epochs)
    loss_fct = RadicalV3Loss().to(device)
    weight_manager = DynamicWeightManager(n_classes)
    scaler = torch.amp.GradScaler('cuda')

    current_weights = weight_manager.current_weights
    best_iou = 0.0

    for epoch in range(n_epochs):
        w_cpu = current_weights.cpu().numpy()
        boosted = w_cpu.copy()
        boosted[5]*=50; boosted[6]*=50; boosted[7]*=50 # Extreme Sampling
        image_weights = np.maximum(np.dot(train_dataset.image_contents, boosted), 0.1)
        sampler = torch.utils.data.WeightedRandomSampler(image_weights, num_samples=6000, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

        model.train(); train_ious = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for i, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = F.interpolate(model(imgs).logits, size=imgs.shape[2:], mode="bilinear")
                loss = loss_fct(outputs, labels, current_weights) / acc_steps
            scaler.scale(loss).backward()
            if (i + 1) % acc_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            with torch.no_grad():
                ious, _ = get_metrics(outputs, labels)
                train_ious.append(ious)
            pbar.set_postfix(loss=f"{loss.item()*acc_steps:.4f}", iou=f"{np.nanmean(ious):.3f}")

        v_loss, v_class_ious, v_class_accs = evaluate_metrics_with_tta(model, val_loader, device, loss_fct, current_weights)
        current_weights = weight_manager.update_weights(v_class_ious)

        # SAVE SAMPLES
        epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_samples_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for s_i, idx in enumerate(random.sample(range(len(val_dataset)), 5)):
                img, mask, d_id = val_dataset[idx]
                out = F.interpolate(model(img.unsqueeze(0).to(device)).logits, size=(h,w), mode="bilinear")[0]
                save_prediction_comparison(img, mask, torch.argmax(out, 0), os.path.join(epoch_samples_dir, f"s_{s_i}_{d_id}"), d_id)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        for c in range(n_classes): print(f"{class_names[c]:<18} | T-IoU: {np.nanmean(train_ious,0)[c]:.3f} | V-IoU: {v_class_ious[c]:.3f} | V-Acc: {v_class_accs[c]:.3f}")
        print(f"MEAN {'':<13} | {np.nanmean(train_ious):.3f}    | {np.nanmean(v_class_ious):.3f}\n" + "=" * 50)

        torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))
        if np.nanmean(v_class_ious) > best_iou:
            best_iou = np.nanmean(v_class_ious)
            torch.save(model.state_dict(), os.path.join(script_dir, "segformer_v3_best.pth"))

if __name__ == "__main__": main()
