
import os
import argparse
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

from torchvision.models import resnet50


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class ResNetWithCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone    = resnet50(weights=None)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam3  = CBAMBlock(1024, reduction=16)
        self.cbam4  = CBAMBlock(2048, reduction=16)
        feat_dim    = backbone.fc.in_features
        self.gap    = nn.AdaptiveAvgPool2d(1)
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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.gap(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Test Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.transform   = transform
        self.image_paths = []
        for root, _, files in os.walk(test_dir):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_paths.append(os.path.join(root, f))
        self.image_paths = sorted(self.image_paths)
        print(f"Found {len(self.image_paths)} test images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img  = Image.open(path).convert("RGB")
        return self.transform(img), os.path.basename(path)


# ─────────────────────────────────────────────────────────────────────────────
# 10-Crop TTA
# Center crop + 4 corner crops + horizontal flips of all 5 = 10 total
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

ten_crop_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.TenCrop(224),                         # returns 10 PIL images
    transforms.Lambda(
        lambda crops: torch.stack([
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
                transforms.ToTensor()(c)
            ) for c in crops
        ])
    ),
])


def predict_with_tta(model, test_dir, device):
    """Run 10-crop TTA and return {filename: probability_vector}."""
    dataset = TestDataset(test_dir, ten_crop_transform)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    all_probs = {}
    model.eval()
    with torch.no_grad():
        for batch_imgs, fnames in loader:
            # batch_imgs: (B, 10, C, H, W)
            B, n_crops, C, H, W = batch_imgs.shape
            imgs = batch_imgs.view(B * n_crops, C, H, W).to(device)

            logits = model(imgs)                    # (B*10, num_classes)
            probs  = F.softmax(logits, dim=1)
            probs  = probs.view(B, n_crops, -1).mean(dim=1)  # average 10 crops

            for fname, p in zip(fnames, probs):
                all_probs[fname] = p.cpu()

    return all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Class mapping from training folder
    ref_dataset  = ImageFolder("./train")
    num_classes  = len(ref_dataset.classes)
    idx_to_class = {v: k for k, v in ref_dataset.class_to_idx.items()}
    print(f"Number of classes: {num_classes}")

    def load_model(path):
        mdl = ResNetWithCBAM(num_classes=num_classes)
        mdl.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        mdl = mdl.to(device)
        mdl.eval()
        print(f"Loaded: {path}")
        return mdl

    # ── Model 1 (required) ────────────────────────────────────────────────
    model1 = load_model(args.model1)
    probs1 = predict_with_tta(model1, args.test_dir, device)
    print(f"Model 1 TTA done — {len(probs1)} images")

    # ── Model 2 (optional ensemble) ───────────────────────────────────────
    if args.model2 and os.path.exists(args.model2):
        model2 = load_model(args.model2)
        probs2 = predict_with_tta(model2, args.test_dir, device)
        print(f"Model 2 TTA done — ensembling")
        final_probs = {f: (probs1[f] + probs2[f]) / 2.0 for f in probs1}
    else:
        final_probs = probs1

    # ── Build prediction.csv ──────────────────────────────────────────────
    rows = []
    for fname, prob in final_probs.items():
        img_id     = os.path.splitext(fname)[0]
        class_name = idx_to_class[prob.argmax().item()]
        rows.append({"id": img_id, "prediction": class_name})

    df = pd.DataFrame(rows)
    df.to_csv("prediction.csv", index=False)
    print(f"\nSaved prediction.csv  ({len(df)} rows)")

    # ── Auto-zip for CodaBench ────────────────────────────────────────────
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("prediction.csv", arcname="prediction.csv")
    print("Created submission.zip  →  upload this to CodaBench")
    print("\nDone! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW1 Inference v2")
    parser.add_argument("--test_dir", type=str, default="./test")
    parser.add_argument("--model1",   type=str, default="./best_model.pth",
                        help="Primary model checkpoint")
    parser.add_argument("--model2",   type=str, default=None,
                        help="Optional second checkpoint for ensemble")
    args = parser.parse_args()
    run_inference(args)
