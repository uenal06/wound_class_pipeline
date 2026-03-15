import os
import time
import copy
import argparse
import yaml
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models

import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# Basic Setup & Seeds
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ============================================================
# Dataset
# ============================================================

class WoundDatasetWithMask(Dataset):
    def __init__(self, samples, mask_folder_root, classes, class_to_idx, ext_test_base_path_mask=None, transform=None, mask_transform=None):
        self.samples = samples
        self.mask_folder_root = Path(mask_folder_root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.mask_transform = mask_transform
        self.ext_test_base_path_mask = ext_test_base_path_mask

    def __len__(self):
        return len(self.samples)

    def _find_mask(self, image_path_str):
        image_path = Path(image_path_str)
        image_name = image_path.stem

        is_external_test = (str(self.mask_folder_root) == str(self.ext_test_base_path_mask))

        if is_external_test:
            wound_type = image_path.parent.name
            mask_search_dir = self.mask_folder_root / wound_type
        else:
            split_type = image_path.parent.name
            wound_type = image_path.parent.parent.name
            mask_search_dir = self.mask_folder_root / wound_type / split_type

        possible_mask_names = [f"{image_name}.png", f"{image_name}.jpg", f"{image_name}_mask.png", f"{image_name}_mask.jpg"]

        for mask_name in possible_mask_names:
            candidate = mask_search_dir / mask_name
            if candidate.exists():
                return str(candidate)
        return None

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        mask_path = self._find_mask(image_path)
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 255)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = T.Compose([T.Resize((256, 256)), T.ToTensor()])(mask)

        mask = (mask > 0.5).float()
        return image, mask, label

def collect_samples_train_with_fallback(base_path, train_split="train", aug_split="train_aug", use_augmented=True, other_splits=("val", "test"), exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif")):
    base = Path(base_path)
    if not base.exists():
        return [], {}, {"train": [], "val": [], "test": []}
        
    class_names = sorted([d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    samples = {train_split: [], "val": [], "test": []}

    def list_imgs(folder: Path):
        if not folder.is_dir(): return []
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

    for cls in class_names:
        cls_dir = base / cls
        label = class_to_idx[cls]

        chosen_train_dir = cls_dir / train_split
        if use_augmented:
            aug_dir = cls_dir / aug_split
            if aug_dir.exists() and len(list_imgs(aug_dir)) > 0:
                chosen_train_dir = aug_dir

        for p in list_imgs(chosen_train_dir):
            samples[train_split].append((str(p), label))

        for sp in other_splits:
            sp_dir = cls_dir / sp
            for p in list_imgs(sp_dir):
                samples[sp].append((str(p), label))

    return class_names, class_to_idx, samples

# ============================================================
# Classification Model
# ============================================================

class SoftGuidedAttentionEfficientNet(nn.Module):
    def __init__(self, num_classes, variant="efficientnet_b3", pretrained=True,
                 mask_resize_var=1.0, attention_strength=0.75, attention_power=1.15,
                 attention_threshold=0.0, use_residual=True, mask_alpha=0.38,
                 bg_floor=0.55, roi_boost=1.10, roi_pool_temperature=1.00, mid_idx=6, hi_idx=8):
        super().__init__()
        self.mask_resize_var = float(mask_resize_var)
        self.attention_strength = float(attention_strength)
        self.attention_power = float(attention_power)
        self.attention_threshold = float(attention_threshold)
        self.use_residual = bool(use_residual)
        self.mask_alpha = float(mask_alpha)
        self.bg_floor = float(bg_floor)
        self.roi_boost = float(roi_boost)
        self.roi_pool_temperature = float(roi_pool_temperature)
        
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.backbone_features = backbone.features
        self.mid_idx = int(mid_idx)
        self.hi_idx = int(hi_idx)

        mid_channels, hi_channels = self._infer_channels()

        self.attention_conv_mid = nn.Sequential(
            nn.Conv2d(mid_channels, max(32, mid_channels // 4), kernel_size=1),
            nn.BatchNorm2d(max(32, mid_channels // 4)), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(max(32, mid_channels // 4), max(16, mid_channels // 8), kernel_size=1),
            nn.BatchNorm2d(max(16, mid_channels // 8)), nn.ReLU(inplace=True),
            nn.Conv2d(max(16, mid_channels // 8), 1, kernel_size=1), nn.Sigmoid()
        )

        self.attention_conv_hi = nn.Sequential(
            nn.Conv2d(hi_channels, max(64, hi_channels // 4), kernel_size=1),
            nn.BatchNorm2d(max(64, hi_channels // 4)), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(max(64, hi_channels // 4), max(32, hi_channels // 8), kernel_size=1),
            nn.BatchNorm2d(max(32, hi_channels // 8)), nn.ReLU(inplace=True),
            nn.Conv2d(max(32, hi_channels // 8), 1, kernel_size=1), nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hi_channels * 2, num_classes)

    def _infer_channels(self):
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)
            for i, block in enumerate(self.backbone_features):
                x = block(x)
                if i == self.mid_idx: feat_mid = x
                if i == self.hi_idx: feat_hi = x
            return int(feat_mid.shape[1]), int(feat_hi.shape[1])

    @staticmethod
    def _ensure_mask_4d(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3: mask = mask.unsqueeze(1)
        return mask

    def _resize_mask(self, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        mask = SoftGuidedAttentionEfficientNet._ensure_mask_4d(mask)
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return (mask * self.mask_resize_var).clamp(0.0, 1.0)

    def _soft_mask_gate(self, m: torch.Tensor) -> torch.Tensor:
        return self.bg_floor + (self.roi_boost - self.bg_floor) * m

    def _combine_mask_and_attention(self, att: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mask_gate = self._soft_mask_gate(m)
        combined = self.mask_alpha * mask_gate + (1.0 - self.mask_alpha) * att
        if self.attention_power != 1.0: combined = torch.pow(combined + 1e-8, self.attention_power)
        if self.attention_threshold > 0.0: combined = torch.where(combined > self.attention_threshold, combined, torch.zeros_like(combined))
        return combined

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        feat_mid = None
        for i, block in enumerate(self.backbone_features):
            x = block(x)
            if i == self.mid_idx: feat_mid = x
            if i == self.hi_idx: feat_hi = x

        att_mid = self.attention_conv_mid(feat_mid)
        if mask is not None:
            gate_mid = self._combine_mask_and_attention(att_mid, self._resize_mask(mask, feat_mid.shape[2], feat_mid.shape[3]))
        else:
            gate_mid = torch.pow(att_mid + 1e-8, self.attention_power) if self.attention_power != 1.0 else att_mid
            
        attended_mid = feat_mid * gate_mid
        if self.use_residual: attended_mid = attended_mid + 0.1 * feat_mid
        feat_mid = self.attention_strength * attended_mid + (1.0 - self.attention_strength) * feat_mid

        x = feat_mid
        for i in range(self.mid_idx + 1, self.hi_idx + 1): x = self.backbone_features[i](x)
        features = x

        att_hi = self.attention_conv_hi(features)
        if mask is not None:
            m_hi = self._resize_mask(mask, features.shape[2], features.shape[3])
            gate_hi = self._combine_mask_and_attention(att_hi, m_hi)
            attended_hi = features * gate_hi
            if self.use_residual: attended_hi = attended_hi + 0.1 * features
            features = self.attention_strength * attended_hi + (1.0 - self.attention_strength) * features

            pooled_global = self.avgpool(features).flatten(1)
            roi_weight = self._soft_mask_gate(m_hi)
            if self.roi_pool_temperature != 1.0: roi_weight = torch.pow(roi_weight + 1e-8, self.roi_pool_temperature)
            pooled_roi = self.avgpool(features * roi_weight).flatten(1)
        else:
            gate_hi = torch.pow(att_hi + 1e-8, self.attention_power) if self.attention_power != 1.0 else att_hi
            attended_hi = features * gate_hi
            if self.use_residual: attended_hi = attended_hi + 0.1 * features
            features = self.attention_strength * attended_hi + (1.0 - self.attention_strength) * features
            pooled_global = self.avgpool(features).flatten(1)
            pooled_roi = pooled_global

        pooled = torch.cat([pooled_global, pooled_roi], dim=1)
        out = self.classifier(pooled)
        
        if getattr(self, 'export_gradcam', False):
            return out, features
            
        return out

class EarlyStopping:
    def __init__(self, patience=5, mode="max", min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = float(min_delta)
        self.restore_best_weights = restore_best_weights
        self.best_score = -math.inf if mode == "max" else math.inf
        self.best_state = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max": return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score, model, epoch):
        if self._is_improvement(float(score)):
            self.best_score = float(score)
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
# Main Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wound Classification Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    cfg = config['classification']
    paths = config['paths']
    SEED = cfg['seed']
    seed_everything(SEED)
    
    IMG_SIZE = tuple(cfg['img_size'])
    BATCH_SIZE = cfg['batch_size']
    LR = cfg['lr']
    NUM_EPOCHS = cfg['epochs']
    EARLYSTOPPING_PATIENCE = cfg['early_stopping_patience']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])

    base_path = paths['cls_base_path']
    mask_path = paths['cls_mask_path']
    ext_test_base_path_mask = paths.get('ext_test_base_path_mask')

    class_names, class_to_idx, samples = collect_samples_train_with_fallback(
        base_path=base_path,
        train_split="train",
        aug_split=cfg['aug_split_name'],
        use_augmented=cfg['use_augmented'],
        other_splits=("val", "test"),
    )

    if len(samples['train']) == 0:
        print("No training data found! Check config.yaml paths.")
        return

    train_dataset = WoundDatasetWithMask(samples["train"], mask_path, class_names, class_to_idx, ext_test_base_path_mask, transform, transform)
    val_dataset = WoundDatasetWithMask(samples["val"], mask_path, class_names, class_to_idx, ext_test_base_path_mask, transform, transform)

    g_loader = torch.Generator()
    g_loader.manual_seed(SEED)
    def seed_worker(worker_id):
        worker_seed = (SEED + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g_loader, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, pin_memory=True)

    model = SoftGuidedAttentionEfficientNet(num_classes=cfg['num_classes']).to(device)

    for p in model.parameters(): p.requires_grad = False
    for name, p in model.named_parameters():
        if ("attention" in name) or ("classifier" in name): p.requires_grad = True
    for i in range(len(model.backbone_features) - 3, len(model.backbone_features)):
        for p in model.backbone_features[i].parameters(): p.requires_grad = True

    backbone_params = []
    attention_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone_features" in name: backbone_params.append(p)
        elif "attention" in name: attention_params.append(p)
        else: head_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-5, "weight_decay": 1e-4},
        {"params": attention_params, "lr": 4e-4, "weight_decay": 1e-4},
        {"params": head_params, "lr": LR, "weight_decay": 1e-4},
    ])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    early_stopping = EarlyStopping(patience=EARLYSTOPPING_PATIENCE, mode="max", min_delta=1e-4, restore_best_weights=True)

    mlflow_tracking_uri = paths['mlflow_tracking_uri']
    os.makedirs(mlflow_tracking_uri.replace("file:", ""), exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(cfg['mlflow_experiment_name'])
    
    run_name = f"resnet_att_cls_bs{BATCH_SIZE}_lr{LR}_{int(time.time())}"
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    output_dir = paths['cls_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for inputs, masks, labels in train_loader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, masks, labels in val_loader:
                    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                    outputs = model(inputs, masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader.dataset)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            mlflow.log_metric("train_loss", train_loss, step=epoch+1)
            mlflow.log_metric("val_loss", val_loss, step=epoch+1)
            mlflow.log_metric("val_acc", val_acc, step=epoch+1)
            mlflow.log_metric("val_f1", val_f1, step=epoch+1)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

            early_stopping(val_f1, model, epoch + 1)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        early_stopping.restore(model)
        
        model_path = os.path.join(output_dir, "wound_classification_model_best.pth")
        torch.save(model.state_dict(), model_path)

        hist_df = pd.DataFrame(history)
        hist_df['epoch'] = range(1, len(history["train_loss"]) + 1)
        hist_path = os.path.join(output_dir, "history.csv")
        hist_df.to_csv(hist_path, index=False)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(hist_path)
        mlflow.pytorch.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()
