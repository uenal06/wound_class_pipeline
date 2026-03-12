import os
import time
import copy
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

import mlflow
import mlflow.pytorch

# ============================================================
# Basic Blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_ch_x, in_ch_g, inter_ch):
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2):
        super().__init__()
        self.up = UpConv(in_ch, out_ch)
        self.att = AttentionBlock(in_ch_x=skip_ch, in_ch_g=out_ch, inter_ch=max(out_ch // 2, 8))
        self.conv = nn.Sequential(
            ConvBlock(out_ch + skip_ch, out_ch),
            nn.Dropout2d(dropout),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# ============================================================
# Attention U-Net with EfficientNet-B0 Encoder
# ============================================================

class AttentionUNetEfficientNetB0(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        if in_channels != 3:
            first = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        return_nodes = {
            "features.0": "x0",  # H/2, 32
            "features.2": "x1",  # H/4, 24
            "features.3": "x2",  # H/8, 40
            "features.5": "x3",  # H/16, 112
            "features.8": "x4",  # H/32, 1280
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        self.decoder = nn.ModuleDict({
            "d4": DecoderBlock(1280, 112, 112),
            "d3": DecoderBlock(112, 40, 40),
            "d2": DecoderBlock(40, 24, 24),
            "d1": DecoderBlock(24, 32, 32),
        })

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        d4 = self.decoder["d4"](feats["x4"], feats["x3"])
        d3 = self.decoder["d3"](d4, feats["x2"])
        d2 = self.decoder["d2"](d3, feats["x1"])
        d1 = self.decoder["d1"](d2, feats["x0"])
        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out

# ============================================================
# Losses and Metrics
# ============================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def iou_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection + 1e-7
    return (intersection / union).item()

def dice_coef(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    intersection = (2.0 * (y_true * y_pred).sum())
    denom = y_true.sum() + y_pred.sum() + 1e-7
    return (intersection / denom).item()

class EarlyStopping:
    def __init__(self, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = float("-inf")
        self.best_state = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, model, epoch):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)

# ============================================================
# Data Loading
# ============================================================

def load_images_and_masks(img_dir, mask_dir, img_size):
    img_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])

    images, masks = [], []
    filenames = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []

    for fname in filenames:
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        img_path = os.path.join(img_dir, fname)
        basename, _ = os.path.splitext(fname)

        img = Image.open(img_path).convert("RGB")
        img = img_transform(img)

        mask_path_png = os.path.join(mask_dir, basename + ".png")
        mask_path_jpg = os.path.join(mask_dir, basename + ".jpg")

        if os.path.exists(mask_path_png):
            mask = Image.open(mask_path_png).convert("L")
        elif os.path.exists(mask_path_jpg):
            mask = Image.open(mask_path_jpg).convert("L")
        else:
            print(f"Warning: No mask found for {fname}")
            continue

        mask = mask_transform(mask)
        mask = (mask > 0.5).float()
        images.append(img)
        masks.append(mask)

    if not images:
        return torch.empty(0), torch.empty(0)
    return torch.stack(images), torch.stack(masks)

# ============================================================
# Main Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wound Segmentation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    cfg = config['segmentation']
    paths = config['paths']
    
    IMG_SIZE = tuple(cfg['img_size'])
    BATCH_SIZE = cfg['batch_size']
    LR = cfg['lr']
    NUM_EPOCHS = cfg['epochs']
    EARLYSTOPPING_PATIENCE = cfg['early_stopping_patience']
    
    torch.manual_seed(cfg['seed'])
    
    # Paths
    base_path = paths['seg_base_path']
    train_images_path = os.path.join(base_path, "train_images_aug")
    train_masks_path = os.path.join(base_path, "train_masks_aug")
    val_images_path = os.path.join(base_path, "val_images")
    val_masks_path = os.path.join(base_path, "val_masks")

    print("Loading data...")
    X_train, Y_train = load_images_and_masks(train_images_path, train_masks_path, IMG_SIZE)
    X_val, Y_val = load_images_and_masks(val_images_path, val_masks_path, IMG_SIZE)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("No training or validation data found! Check paths in config.yaml.")
        return

    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    g = torch.Generator()
    g.manual_seed(cfg['seed'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AttentionUNetEfficientNetB0(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        pretrained=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=EARLYSTOPPING_PATIENCE, restore_best_weights=True)

    # MLflow Setup
    mlflow_tracking_uri = paths['mlflow_tracking_uri']
    os.makedirs(mlflow_tracking_uri.replace("file:", ""), exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(cfg['mlflow_experiment_name'])

    run_name = f"seg_bs{BATCH_SIZE}_lr{LR}_{int(time.time())}"
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    output_dir = paths['seg_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_sum = 0.0

            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

            train_loss = train_loss_sum / len(train_loader)

            model.eval()
            val_loss_sum, val_dice_sum, val_iou_sum = 0.0, 0.0, 0.0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    probs = torch.sigmoid(logits)

                    val_loss_sum += loss.item()
                    val_dice_sum += dice_coef(masks, probs)
                    val_iou_sum += iou_score(masks, probs)

            val_loss = val_loss_sum / len(val_loader)
            val_dice = val_dice_sum / len(val_loader)
            val_iou = val_iou_sum / len(val_loader)

            scheduler.step(val_dice)
            early_stopping(val_dice, model, epoch + 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)

            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("val_dice", val_dice, step=epoch + 1)
            mlflow.log_metric("val_iou", val_iou, step=epoch + 1)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_dice {val_dice:.4f} | val_iou {val_iou:.4f}")

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        early_stopping.restore(model)
        
        model_path = os.path.join(output_dir, "wound_segmentation_model_best.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save history
        hist_df = pd.DataFrame(history)
        hist_df['epoch'] = range(1, len(history["train_loss"]) + 1)
        hist_path = os.path.join(output_dir, "history.csv")
        hist_df.to_csv(hist_path, index=False)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(hist_path)
        mlflow.pytorch.log_model(model, artifact_path="model")

    print("Training complete.")

if __name__ == "__main__":
    main()
