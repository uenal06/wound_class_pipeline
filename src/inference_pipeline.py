import os
import argparse
import yaml
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# Try importing the model classes from the training scripts.
# If they are in the same directory, this will work.
try:
    from train_segmentation import AttentionUNetEfficientNetB0
    from train_classification import SoftGuidedAttentionEfficientNet
except ImportError:
    print("Warning: Could not import model classes from train_segmentation.py and train_classification.py.")
    print("Please make sure they are in the same directory as inference_pipeline.py.")
    exit(1)

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def tensor_to_uint8_rgb(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Wound Inference Pipeline (Segmentation + Classification)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    cfg_seg = config['segmentation']
    cfg_cls = config['classification']
    cfg_inf = config['inference']
    paths = config['paths']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = paths['inference_input_dir']
    output_dir = paths['inference_output_dir']
    
    out_rgb_root = os.path.join(output_dir, "images_resized")
    out_mask_root = os.path.join(output_dir, "masks_pred")
    os.makedirs(out_rgb_root, exist_ok=True)
    os.makedirs(out_mask_root, exist_ok=True)

    # 1. Load Segmentation Model
    print(f"Loading Segmentation Model from {cfg_inf['seg_model_path']}...")
    seg_model = AttentionUNetEfficientNetB0(
        in_channels=cfg_seg['in_channels'],
        out_channels=cfg_seg['out_channels'],
        pretrained=False
    ).to(device)
    seg_model.load_state_dict(torch.load(cfg_inf['seg_model_path'], map_location=device))
    seg_model.eval()

    # 2. Load Classification Model
    print(f"Loading Classification Model from {cfg_inf['cls_model_path']}...")
    cls_model = SoftGuidedAttentionEfficientNet(
        num_classes=cfg_cls['num_classes']
    ).to(device)
    cls_model.load_state_dict(torch.load(cfg_inf['cls_model_path'], map_location=device))
    cls_model.eval()

    seg_img_size = tuple(cfg_seg['img_size'])
    resize_to_tensor_seg = T.Compose([
        T.Resize(seg_img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    normalize_seg = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cls_img_size = tuple(cfg_cls['img_size'])
    resize_to_tensor_cls = T.Compose([
        T.Resize(cls_img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

    results = []

    print("Starting Batch Inference...")
    count = 0
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith(VALID_EXTS):
                continue
            
            in_path = os.path.join(root, fn)
            rel_dir = os.path.relpath(root, input_dir)
            base = os.path.splitext(fn)[0]

            # Output paths for segment_batch.py compatibility
            out_rgb_path = os.path.join(out_rgb_root, rel_dir, f"{base}.png")
            out_mask_path = os.path.join(out_mask_root, rel_dir, f"{base}_mask.png")
            
            os.makedirs(os.path.dirname(out_rgb_path), exist_ok=True)
            os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

            try:
                img_pil = Image.open(in_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {in_path}: {e}")
                continue

            # --- SEGMENTATION ---
            x_resized = resize_to_tensor_seg(img_pil) # [3,H,W]
            x_norm = normalize_seg(x_resized).unsqueeze(0).to(device) # [1,3,H,W]
            
            seg_pred = seg_model(x_norm)
            seg_probs = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
            mask_np = (seg_probs >= cfg_inf['threshold']).astype(np.uint8)
            
            # Save segmentation outputs (like segment_batch.py)
            rgb_pil = tensor_to_uint8_rgb(x_resized)
            rgb_pil.save(out_rgb_path)
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(seg_img_size, resample=Image.NEAREST)
            mask_pil.save(out_mask_path)

            # --- CLASSIFICATION ---
            # Input image for cls
            img_cls = resize_to_tensor_cls(img_pil).unsqueeze(0).to(device)
            # Mask for cls
            mask_tensor_cls = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
            mask_tensor_cls = (mask_tensor_cls > 0.5).float()

            cls_pred = cls_model(img_cls, mask_tensor_cls)
            predicted_class = torch.argmax(cls_pred, dim=1).item()

            results.append({
                "image_path": in_path,
                "mask_path": out_mask_path,
                "predicted_class_idx": predicted_class
            })

            count += 1
            if count % 25 == 0:
                print(f"Processed {count} images...")

    print(f"\nInference completed for {count} images.")
    
    # Save results
    if count > 0:
        import pandas as pd
        df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "classification_results.csv")
        df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

if __name__ == "__main__":
    main()
