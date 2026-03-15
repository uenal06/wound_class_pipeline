import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Try importing the model classes from the training scripts.
# If they are in the same directory, this will work.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train_segmentation import AttentionUNetEfficientNetB0
    from train_classification import SoftGuidedAttentionEfficientNet
except ImportError:
    print("Warning: Could not import model classes from train_segmentation.py and train_classification.py.")
    print("Please make sure they are in the same directory as single_inference_plot.py.")
    exit(1)


class AttentionMaskGradCAM:
    def __init__(self, model, use_ts=False):
        self.model = model
        self.use_ts = use_ts
        self.activations = None
        self.gradients = None
        self.hook = None
        
        if not use_ts:
            self.layer = model.backbone_features[model.hi_idx]
            self.hook = self.layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        if not isinstance(out, torch.Tensor):
            if isinstance(out, tuple):
                out = out[0]
            else:
                return
        self.activations = out
        if not out.requires_grad:
            return
        def grad_hook(grad):
            self.gradients = grad
        out.register_hook(grad_hook)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()

    def __call__(self, img, mask, target_class):
        self.model.eval()
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            img = img.clone().detach().requires_grad_(True)
            out = self.model(img, mask)
            
            is_ts_with_features = False
            if isinstance(out, (tuple, list)):
                logits = out[0]
                features = out[1]
                is_ts_with_features = True
                features.retain_grad()
            else:
                logits = out
                features = None
                
            if self.use_ts and not is_ts_with_features:
                print("Warning: GradCAM is not supported on this .pt model. Export it using export_gradcam=True in your notebook. Returning empty CAM.")
                cam = np.zeros((img.shape[2], img.shape[3]), dtype=np.float32)
                return cam

            score = logits[0, target_class]
            score.backward()

        if is_ts_with_features:
            acts = features.clone().detach()
            grads = features.grad.clone().detach()
        else:
            if self.activations is None or self.gradients is None:
                # Fallback if hooks fail
                print("Warning: GradCAM hooks failed to capture activations/gradients, returning empty map.")
                cam = np.zeros((img.shape[2], img.shape[3]), dtype=np.float32)
                return cam
            acts = self.activations
            grads = self.gradients

        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        cam = cam[0,0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()


def tensor_to_img(img_tensor):
    img = img_tensor[0].cpu().permute(1,2,0).numpy()
    return np.clip(img,0,1)

def resize_mask(mask_numpy, shape):
    return cv2.resize(mask_numpy, shape, interpolation=cv2.INTER_NEAREST)

def overlay_mask(img, mask, alpha=0.6):
    h,w,_ = img.shape
    mask = resize_mask(mask,(w,h))

    green = np.zeros_like(img)
    green[:,:,1] = 1

    active = mask > 0.5

    out = img.copy()
    out[active] = img[active]*(1-alpha) + green[active]*alpha

    return out

def overlay_cam(img, cam):
    h,w,_ = img.shape
    cam = cv2.resize(cam,(w,h))

    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat,cv2.COLOR_BGR2RGB) / 255.0

    return img*0.6 + heat*0.4


def overlay_cam_mask(img, cam, mask):
    base = overlay_mask(img,mask)
    return overlay_cam(base,cam)

def main():
    parser = argparse.ArgumentParser(description="Wound Single Image Inference (Segmentation + Classification with GradCAM)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    cfg_seg = config['segmentation']
    cfg_cls = config['classification']
    cfg_inf = config['inference']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Segmentation Model
    print(f"Loading Segmentation Model from {cfg_inf['seg_model_path']}...")
    try:
        seg_model = torch.jit.load(cfg_inf['seg_model_path'], map_location=device)
        seg_model.eval()
        use_ts_seg = True
    except Exception as e:
        print(f"Could not load seg model as TorchScript: {e}. Trying state_dict...")
        use_ts_seg = False
        seg_model = AttentionUNetEfficientNetB0(
            in_channels=cfg_seg['in_channels'],
            out_channels=cfg_seg['out_channels'],
            pretrained=False
        ).to(device)
        try:
            seg_model.load_state_dict(torch.load(cfg_inf['seg_model_path'], map_location=device))
            seg_model.eval()
        except Exception as e2:
            print(f"Failed to load seg model: {e2}")
            return

    # 2. Load Classification Model
    print(f"Loading Classification Model from {cfg_inf['cls_model_path']}...")
    try:
        cls_model = torch.jit.load(cfg_inf['cls_model_path'], map_location=device)
        cls_model.eval()
        use_ts_cls = True
        print("Loaded TorchScript classification model.")
    except Exception as e:
        print(f"Failed to load classification model: {e}")
        return

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

    print(f"Processing image: {args.image}")
    try:
        img_pil = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    with torch.no_grad():
        # --- SEGMENTATION ---
        x_resized = resize_to_tensor_seg(img_pil) # [3,H,W]
        x_norm = normalize_seg(x_resized).unsqueeze(0).to(device) # [1,3,H,W]
        
        seg_pred = seg_model(x_norm)
        if isinstance(seg_pred, (tuple, list)): 
            seg_pred = seg_pred[0]
            
        # Segment batch handles logits via a flag, our models generally output logits
        # Let's assume logits by default for the trained models
        seg_pred = torch.sigmoid(seg_pred)

        if seg_pred.ndim == 4 and seg_pred.shape[1] == 1:
            seg_probs = seg_pred[0, 0].detach().cpu().numpy()
        else:
            seg_probs = seg_pred.squeeze().detach().cpu().numpy()
            
        mask_np = (seg_probs >= cfg_inf['threshold']).astype(np.uint8)
        
        # In segment_batch.py, mask_pil is created like this:
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(seg_img_size, resample=Image.NEAREST)

        # --- CLASSIFICATION ---
        # Input image for cls (the user wants exactly like segment_batch.py + classification notebook)
        # In segment_batch.py, the saved RGB image is tensor_to_uint8_rgb(x_resized).
        # Let's use the exact resized/normalized image from segmentation as input if possible, 
        # or load from the saved image to be perfectly identical to the pipeline.
        
        # To perfectly mimic loading from the saved image:
        rgb_pil_saved = tensor_to_img(x_resized.unsqueeze(0)) # Note: tensor_to_img outputs float 0-1
        rgb_pil_to_save = Image.fromarray((rgb_pil_saved * 255.0).astype(np.uint8))
        
        # Now apply classification transforms
        img_cls = resize_to_tensor_cls(rgb_pil_to_save).unsqueeze(0).to(device)
        
        # Mask for cls
        mask_tensor_cls = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
        mask_tensor_cls = (mask_tensor_cls > 0.5).float()

        cls_logits = cls_model(img_cls, mask_tensor_cls)
        probs = torch.softmax(cls_logits, 1)[0].detach().cpu().numpy()
        predicted_class = int(torch.argmax(cls_logits, dim=1).item())

    class_names = ["diabetic", "masd", "pressure", "venous"]
    pred_name = class_names[predicted_class]
    print(f"Predicted class: {pred_name}")

    # --- GRADCAM ---
    gradcam = AttentionMaskGradCAM(cls_model, use_ts=use_ts_cls)
    cam = gradcam(img_cls, mask_tensor_cls, predicted_class)
    gradcam.remove()

    # --- PLOTTING ---
    img_np = tensor_to_img(img_cls)
    mask_np_cls = mask_tensor_cls[0, 0].cpu().numpy()
    
    mask_overlay = overlay_mask(img_np, mask_np_cls)
    combined = overlay_cam_mask(img_np, cam, mask_np_cls)

    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 5, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mask_overlay)
    plt.title("Input Mask")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(cam, cmap="jet")
    plt.title("GradCAM")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(combined)
    plt.title(f"Combined\nPred: {pred_name}")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    y = np.arange(len(probs))
    bars = plt.barh(y, probs, color="gray")
    bars[predicted_class].set_color("red")
    plt.yticks(y, class_names)
    plt.xlim(0, 1)
    plt.title("Class Probabilities")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
