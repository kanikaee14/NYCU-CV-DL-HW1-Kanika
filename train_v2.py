

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Config
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
PHASE1_EPOCHS   = 5       
PHASE2_EPOCHS   = 40      
MAX_LR          = 3e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE        = 12    
IMG_SIZE        = 224

# MixUp / CutMix probabilities
MIXUP_ALPHA  = 0.4
CUTMIX_ALPHA = 1.0
MIXUP_PROB   = 0.5     

# ─────────────────────────────────────────────────────────────────────────────
# 3. Transforms
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Datasets & Loaders
# ─────────────────────────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder("./train", transform=train_transform)
val_dataset   = datasets.ImageFolder("./val",   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

num_classes = len(train_dataset.classes)
print(f"Classes: {num_classes} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MixUp & CutMix helpers
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a random box from one image onto another."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Model — ResNet-50 + CBAM (Channel + Spatial Attention)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """CBAM channel attention gate."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        w   = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial attention gate."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAMBlock(nn.Module):
    """Full CBAM = channel attention → spatial attention."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResNetWithCBAM(nn.Module):
    """
    ResNet-50 backbone with CBAM attention injected after layer3 and layer4.
    Deeper MLP classifier head with stronger regularisation.
    Total params ~26.7M — well under 100M HW1 limit.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        weights  = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch
        self.layer2 = backbone.layer2   # 512 ch
        self.layer3 = backbone.layer3   # 1024 ch
        self.layer4 = backbone.layer4   # 2048 ch

        # CBAM after layer3 & layer4 — transformer-style attention within ResNet
        self.cbam3 = CBAMBlock(1024, reduction=16)
        self.cbam4 = CBAMBlock(2048, reduction=16)

        feat_dim = backbone.fc.in_features  # 2048
        self.gap  = nn.AdaptiveAvgPool2d(1)

        # Stronger head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam3(x)        # channel + spatial attention after layer3
        x = self.layer4(x)
        x = self.cbam4(x)        # channel + spatial attention after layer4
        x = self.gap(x)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for part in [self.stem, self.layer1, self.layer2,
                     self.layer3, self.layer4]:
            for p in part.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


model = ResNetWithCBAM(num_classes=num_classes).to(device)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total parameters: {total_params:.1f}M  (limit: 100M)")
assert total_params < 100, "Model exceeds 100M parameter limit!"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Loss
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_train(loader, mdl, opt, sched, crit, use_mix=True):
    mdl.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()

        # MixUp / CutMix with 50% probability per batch
        if use_mix and random.random() < MIXUP_PROB:
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)
            out  = mdl(imgs)
            loss = mixup_criterion(crit, out, y_a, y_b, lam)
        else:
            out  = mdl(imgs)
            loss = crit(out, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    sched.step()   # CosineAnnealingWarmRestarts steps per epoch
    return total_loss / len(loader), correct / total


def run_epoch_val(loader, mdl, crit):
    mdl.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = mdl(imgs)
            loss = crit(out, labels)
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 9. Phase 1 — Warmup: head + CBAM only (backbone frozen)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Head + CBAM warmup (backbone frozen)")
print("="*60)

model.freeze_backbone()
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable, lr=MAX_LR, weight_decay=WEIGHT_DECAY)
# Simple cosine for phase 1
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=PHASE1_EPOCHS, T_mult=1)

best_acc  = 0.0
for epoch in range(PHASE1_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch_train(train_loader, model, optimizer,
                                      scheduler, criterion, use_mix=False)
    vl_loss, vl_acc = run_epoch_val(val_loader, model, criterion)
    elapsed = time.time() - t0
    print(f"P1 [{epoch+1:02d}/{PHASE1_EPOCHS}] "
          f"TrainAcc={tr_acc:.4f} ValAcc={vl_acc:.4f} "
          f"Loss={tr_loss:.4f} ({elapsed:.0f}s)")
    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved (best={best_acc:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Phase 2 — Full fine-tune 
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: Full fine-tune + MixUp/CutMix + Warm Restarts")
print("="*60)

model.unfreeze_all()

optimizer = optim.AdamW([
    {"params": model.stem.parameters(),       "lr": MAX_LR * 0.01},
    {"params": model.layer1.parameters(),     "lr": MAX_LR * 0.02},
    {"params": model.layer2.parameters(),     "lr": MAX_LR * 0.05},
    {"params": model.layer3.parameters(),     "lr": MAX_LR * 0.1},
    {"params": model.cbam3.parameters(),      "lr": MAX_LR * 0.5},
    {"params": model.layer4.parameters(),     "lr": MAX_LR * 0.3},
    {"params": model.cbam4.parameters(),      "lr": MAX_LR},
    {"params": model.classifier.parameters(), "lr": MAX_LR},
], weight_decay=WEIGHT_DECAY)

# T_0=10: first restart at epoch 10, T_mult=2 → next at epoch 30, then 70
# This lets the model explore broadly early, then settle into the best minimum
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

no_improve = 0
for epoch in range(PHASE2_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch_train(train_loader, model, optimizer,
                                      scheduler, criterion, use_mix=True)
    vl_loss, vl_acc = run_epoch_val(val_loader, model, criterion)
    elapsed = time.time() - t0

    # Current LR of head group
    cur_lr = optimizer.param_groups[-1]["lr"]
    print(f"P2 [{epoch+1:02d}/{PHASE2_EPOCHS}] "
          f"TrainAcc={tr_acc:.4f} ValAcc={vl_acc:.4f} "
          f"Loss={tr_loss:.4f} LR={cur_lr:.2e} ({elapsed:.0f}s)")

    if vl_acc > best_acc:
        best_acc   = vl_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved (best={best_acc:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at P2 epoch {epoch+1}.")
            break

print(f"\n{'='*60}")
print(f"Training complete.  Best Val Accuracy: {best_acc:.4f}")
print(f"{'='*60}")
