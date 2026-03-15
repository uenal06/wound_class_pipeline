import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import gradio as gr

# Ensure src is in PATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from single_inference_plot import AttentionMaskGradCAM, tensor_to_img, overlay_cam_mask, overlay_mask
from train_segmentation import AttentionUNetEfficientNetB0
from train_classification import SoftGuidedAttentionEfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
config_path = "src/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg_seg = config['segmentation']
cfg_cls = config['classification']
cfg_inf = config['inference']

class_names = ["diabetic", "masd", "pressure", "venous"]
friendly_names = {
    "diabetic": "Diabetic Foot Ulcer",
    "masd": "Moisture Associated Skin Damage (MASD)",
    "pressure": "Pressure Ulcer",
    "venous": "Venous Leg Ulcer"
}

# 1. Load Segmentation Model
print(f"Loading Segmentation Model from {cfg_inf['seg_model_path']}...")
try:
    seg_model = torch.jit.load(cfg_inf['seg_model_path'], map_location=device)
    seg_model.eval()
    use_ts_seg = True
except Exception as e:
    print(f"Could not load seg model as TorchScript. Trying state_dict...")
    use_ts_seg = False
    seg_model = AttentionUNetEfficientNetB0(
        in_channels=cfg_seg['in_channels'],
        out_channels=cfg_seg['out_channels'],
        pretrained=False
    ).to(device)
    seg_model.load_state_dict(torch.load(cfg_inf['seg_model_path'], map_location=device))
    seg_model.eval()

# 2. Load Classification Model
model_gradcam_path = "models/classification_runs/model_gradcam.pth"
print(f"Loading Classification Model from {model_gradcam_path}...")
cls_model = SoftGuidedAttentionEfficientNet(num_classes=4).to(device)
cls_model.load_state_dict(torch.load(model_gradcam_path, map_location=device))
cls_model.eval()
use_ts_cls = False

# Transforms
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

def analyze_wound(img_input):
    if img_input is None:
        return None, None, None, "Please upload an image."

    # Convert numpy array to PIL Image if necessary
    if isinstance(img_input, np.ndarray):
        img_pil = Image.fromarray(img_input).convert("RGB")
    else:
        img_pil = img_input.convert("RGB")

    with torch.no_grad():
        # --- SEGMENTATION ---
        x_resized = resize_to_tensor_seg(img_pil)
        x_norm = normalize_seg(x_resized).unsqueeze(0).to(device)
        
        seg_pred = seg_model(x_norm)
        if isinstance(seg_pred, (tuple, list)): 
            seg_pred = seg_pred[0]
            
        seg_pred = torch.sigmoid(seg_pred)

        if seg_pred.ndim == 4 and seg_pred.shape[1] == 1:
            seg_probs = seg_pred[0, 0].detach().cpu().numpy()
        else:
            seg_probs = seg_pred.squeeze().detach().cpu().numpy()
            
        mask_np = (seg_probs >= cfg_inf['threshold']).astype(np.uint8)
        
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(seg_img_size, resample=Image.NEAREST)

        # --- CLASSIFICATION ---
        rgb_pil_saved = tensor_to_img(x_resized.unsqueeze(0))
        rgb_pil_to_save = Image.fromarray((rgb_pil_saved * 255.0).astype(np.uint8))
        
        img_cls = resize_to_tensor_cls(rgb_pil_to_save).unsqueeze(0).to(device)
        
        mask_tensor_cls = T.ToTensor()(mask_pil).unsqueeze(0).to(device)
        mask_tensor_cls = (mask_tensor_cls > 0.5).float()

        cls_logits = cls_model(img_cls, mask_tensor_cls)
        # Note: if it returns a tuple, grab the first element
        if isinstance(cls_logits, (tuple, list)):
            cls_logits = cls_logits[0]
            
        probs = torch.softmax(cls_logits, 1)[0].detach().cpu().numpy()
        predicted_class = int(torch.argmax(cls_logits, dim=1).item())

    pred_name = class_names[predicted_class]
    confidence = float(probs[predicted_class])
    
    # Format probabilities for Gradio Label
    confidences_dict = {friendly_names[name]: float(prob) for name, prob in zip(class_names, probs)}

    # --- GRADCAM ---
    gradcam = AttentionMaskGradCAM(cls_model, use_ts=use_ts_cls)
    cam = gradcam(img_cls, mask_tensor_cls, predicted_class)
    gradcam.remove()

    # --- VISUALIZATION ---
    img_np = tensor_to_img(img_cls)
    mask_np_cls = mask_tensor_cls[0, 0].cpu().numpy()
    
    # 1. Raw mask image (scaled up to 255)
    mask_display = (mask_np_cls * 255).astype(np.uint8)
    # 2. Mask overlay on original image
    mask_overlay_img = overlay_mask(img_np, mask_np_cls)
    mask_overlay_display = (mask_overlay_img * 255).astype(np.uint8)
    # 3. Combined GradCAM + Mask overlay
    combined = overlay_cam_mask(img_np, cam, mask_np_cls)
    combined_display = (combined * 255).astype(np.uint8)

    # --- REPORT TEXT ---
    if confidence >= 0.8:
        severity_html = "<div style='padding:10px; border-left: 5px solid #28a745; background: #e8f5e9;'>🟢 <b>High Confidence</b></div>"
    elif confidence >= 0.5:
        severity_html = "<div style='padding:10px; border-left: 5px solid #ffc107; background: #fff8e1;'>🟡 <b>Moderate Confidence</b></div>"
    else:
        severity_html = "<div style='padding:10px; border-left: 5px solid #dc3545; background: #ffebee;'>🔴 <b>Low Confidence</b></div>"

    report_md = f"""
### AI WOUND CLASSIFICATION REPORT

**Predicted Type:** {friendly_names[pred_name]}
**Confidence Score:** {confidence:.2f}

**Models:** 
- Segmentation: Attention U-Net (EfficientNet-B0)
- Classification: Soft-Guided Attention Network (EfficientNet-B3)

*Note: This AI system is for demonstration and research purposes only.*
    """

    return mask_display, mask_overlay_display, combined_display, confidences_dict, severity_html, report_md


# --- GRADIO UI ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="AI Wound Classifier") as demo:
    gr.Markdown("# Wound Classification Pipeline")
    gr.Markdown("Upload a wound image and the AI model will automatically segment and classify the wound type.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Wound Image")
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
            gr.Markdown("### Image Analysis Visualizations")
            with gr.Row():
                mask_output = gr.Image(label="Segmentation Mask", interactive=False)
                mask_overlay_output = gr.Image(label="Mask Overlay", interactive=False)
                gradcam_output = gr.Image(label="GradCAM Overlay", interactive=False)
            
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=4, label="Top AI Predictions")
            
            gr.Markdown("### Severity Indicator")
            severity_output = gr.HTML()
            
            gr.Markdown("### AI Diagnosis Report")
            report_output = gr.Markdown()
            
    analyze_btn.click(
        fn=analyze_wound,
        inputs=[image_input],
        outputs=[mask_output, mask_overlay_output, gradcam_output, label_output, severity_output, report_output]
    )
    
app = demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
