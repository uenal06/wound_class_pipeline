import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import streamlit as st

# Ensure src is in PATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from single_inference_plot import AttentionMaskGradCAM, tensor_to_img, overlay_cam_mask, overlay_mask
from train_segmentation import AttentionUNetEfficientNetB0
from train_classification import SoftGuidedAttentionEfficientNet

# Page configuration
st.set_page_config(page_title="AI Wound Classifier", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.severity-high { padding:10px; border-left: 5px solid #28a745; background: #e8f5e9; border-radius: 4px; }
.severity-moderate { padding:10px; border-left: 5px solid #ffc107; background: #fff8e1; border-radius: 4px; }
.severity-low { padding:10px; border-left: 5px solid #dc3545; background: #ffebee; border-radius: 4px; }
.report-box { padding:20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = "src/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg_seg = config['segmentation']
    cfg_cls = config['classification']
    cfg_inf = config['inference']

    # 1. Load Segmentation Model
    try:
        seg_model = torch.jit.load(cfg_inf['seg_model_path'], map_location=device)
        seg_model.eval()
    except Exception:
        seg_model = AttentionUNetEfficientNetB0(
            in_channels=cfg_seg['in_channels'],
            out_channels=cfg_seg['out_channels'],
            pretrained=False
        ).to(device)
        seg_model.load_state_dict(torch.load(cfg_inf['seg_model_path'], map_location=device))
        seg_model.eval()

    # 2. Load Classification Model
    model_gradcam_path = "models/classification_runs/model_gradcam.pth"
    cls_model = SoftGuidedAttentionEfficientNet(num_classes=4).to(device)
    cls_model.load_state_dict(torch.load(model_gradcam_path, map_location=device))
    cls_model.eval()

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
    
    return device, config, seg_model, cls_model, resize_to_tensor_seg, normalize_seg, resize_to_tensor_cls

device, config, seg_model, cls_model, resize_to_tensor_seg, normalize_seg, resize_to_tensor_cls = load_models()
cfg_inf = config['inference']
seg_img_size = tuple(config['segmentation']['img_size'])

class_names = ["diabetic", "masd", "pressure", "venous"]
friendly_names = {
    "diabetic": "Diabetic Foot Ulcer",
    "masd": "Moisture Associated Skin Damage (MASD)",
    "pressure": "Pressure Ulcer",
    "venous": "Venous Leg Ulcer"
}

def analyze_wound(img_input):
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
    
    # Format probabilities
    confidences_dict = {friendly_names[name]: float(prob) for name, prob in zip(class_names, probs)}

    # --- GRADCAM ---
    gradcam = AttentionMaskGradCAM(cls_model, use_ts=False)
    cam = gradcam(img_cls, mask_tensor_cls, predicted_class)
    gradcam.remove()

    # --- VISUALIZATION ---
    img_np = tensor_to_img(img_cls)
    mask_np_cls = mask_tensor_cls[0, 0].cpu().numpy()
    
    # 1. Raw mask image
    mask_display = Image.fromarray((mask_np_cls * 255).astype(np.uint8)).convert("RGB")
    # 2. Mask overlay on original image
    mask_overlay_img = overlay_mask(img_np, mask_np_cls)
    mask_overlay_display = Image.fromarray((mask_overlay_img * 255).astype(np.uint8))
    # 3. Combined GradCAM + Mask overlay
    combined = overlay_cam_mask(img_np, cam, mask_np_cls)
    combined_display = Image.fromarray((combined * 255).astype(np.uint8))

    return mask_display, mask_overlay_display, combined_display, confidences_dict, confidence, pred_name


# --- STREAMLIT UI ---
st.title("AI Wound Classifier")
st.markdown("Upload a wound image and the AI model will automatically segment and classify the wound type.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Wound Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

if uploaded_file is not None and analyze_button:
    with st.spinner('Analyzing...'):
        mask_display, mask_overlay_display, combined_display, confidences_dict, confidence, pred_name = analyze_wound(image)

        with col1:
            st.markdown("### Image Analysis Visualizations")
            vis1, vis2, vis3 = st.columns(3)
            with vis1:
                st.image(mask_display, caption="Segmentation Mask", use_container_width=True)
            with vis2:
                st.image(mask_overlay_display, caption="Mask Overlay", use_container_width=True)
            with vis3:
                st.image(combined_display, caption="GradCAM Overlay", use_container_width=True)

        with col2:
            st.markdown("### Top AI Predictions")
            
            # Create a dataframe for the bar chart
            df = pd.DataFrame(list(confidences_dict.items()), columns=['Class', 'Probability'])
            df = df.sort_values(by='Probability', ascending=False)
            
            # Display progress bars for each class
            for _, row in df.iterrows():
                st.write(f"**{row['Class']}**: {row['Probability']*100:.1f}%")
                st.progress(row['Probability'])
            
            st.markdown("### Severity Indicator")
            if confidence >= 0.8:
                st.markdown("<div class='severity-high'>🟢 <b>High Confidence</b></div>", unsafe_allow_html=True)
            elif confidence >= 0.5:
                st.markdown("<div class='severity-moderate'>🟡 <b>Moderate Confidence</b></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='severity-low'>🔴 <b>Low Confidence</b></div>", unsafe_allow_html=True)
            
            st.markdown("### AI Diagnosis Report")
            st.markdown(f"""
            <div class='report-box'>
            <h4>AI WOUND CLASSIFICATION REPORT</h4>
            <br/>
            <b>Predicted Type:</b> {friendly_names[pred_name]}<br/>
            <b>Confidence Score:</b> {confidence:.2f}<br/>
            <br/>
            <b>Models:</b><br/>
            - Segmentation: Attention U-Net (EfficientNet-B0)<br/>
            - Classification: Soft-Guided Attention Network (EfficientNet-B3)<br/>
            <br/>
            <i>Note: This AI system is for demonstration and research purposes only.</i>
            </div>
            """, unsafe_allow_html=True)
