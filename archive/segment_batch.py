import os
import zipfile
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def maybe_unzip(zip_or_pt_path: str, extract_dir: str) -> str:
    if zip_or_pt_path.lower().endswith(".zip") and os.path.isfile(zip_or_pt_path):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_or_pt_path, "r") as z:
            z.extractall(extract_dir)

        for root, _, files in os.walk(extract_dir):
            for fn in files:
                if fn.lower().endswith(".pt"):
                    return os.path.join(root, fn)

        raise FileNotFoundError("Zip extracted but no .pt TorchScript model found.")
    return zip_or_pt_path


def load_torchscript_model(pt_path: str, device: torch.device) -> torch.jit.ScriptModule:
    m = torch.jit.load(pt_path, map_location=device)
    m.eval()
    return m


def build_transforms(img_size: Tuple[int, int]):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    resize_to_tensor = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])

    normalize = T.Normalize(mean=mean, std=std)
    return resize_to_tensor, normalize


def tensor_to_uint8_rgb(x: torch.Tensor) -> Image.Image:
    """
    x: float tensor [3,H,W] in 0..1
    """
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def segment_one(
    model: torch.jit.ScriptModule,
    img_pil: Image.Image,
    device: torch.device,
    img_size: Tuple[int, int],
    threshold: float,
    model_outputs_logits: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
      x_resized: [3,H,W] float in 0..1 (NOT normalized)
      mask_2d: uint8 numpy [H,W] 0 or 1
    """
    resize_to_tensor, normalize = build_transforms(img_size)

    x_resized = resize_to_tensor(img_pil)            # [3,H,W] in 0..1
    x_norm = normalize(x_resized).unsqueeze(0).to(device)  # [1,3,H,W]

    pred = model(x_norm)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    if model_outputs_logits:
        pred = torch.sigmoid(pred)

    if pred.ndim == 4 and pred.shape[1] == 1:
        prob = pred[0, 0].detach().cpu().numpy()
    else:
        prob = pred.squeeze().detach().cpu().numpy()

    mask = (prob >= threshold).astype(np.uint8)
    return x_resized, mask


def save_outputs(
    out_rgb_path: str,
    out_mask_path: str,
    x_resized: torch.Tensor,
    mask: np.ndarray,
    save_tensor_pt: bool,
    out_tensor_path: Optional[str],
    normalize_tensor: bool,
    img_size: Tuple[int, int],
):
    os.makedirs(os.path.dirname(out_rgb_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

    rgb_pil = tensor_to_uint8_rgb(x_resized)
    rgb_pil.save(out_rgb_path)

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((img_size[0], img_size[1]), resample=Image.NEAREST)
    mask_pil.save(out_mask_path)

    if save_tensor_pt:
        if out_tensor_path is None:
            raise ValueError("out_tensor_path must be provided when save_tensor_pt is True")

        os.makedirs(os.path.dirname(out_tensor_path), exist_ok=True)

        tensor_to_save = x_resized
        if normalize_tensor:
            _, normalize = build_transforms(img_size)
            tensor_to_save = normalize(x_resized)

        torch.save(tensor_to_save, out_tensor_path)


def main():
    parser = argparse.ArgumentParser(description="TorchScript segmentation + export resized images and masks.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument("--model_path", type=str, required=True, help="TorchScript .pt or zipped .pt")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256], help="width height")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_outputs_logits", action="store_true")

    parser.add_argument("--save_tensor_pt", action="store_true", help="Also save tensor per image as .pt")
    parser.add_argument("--normalize_tensor", action="store_true", help="If saving .pt, save normalized tensor")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = (int(args.img_size[0]), int(args.img_size[1]))

    model_path = maybe_unzip(args.model_path, extract_dir="./tmp_seg_model_ts")
    model = load_torchscript_model(model_path, device)

    out_rgb_root = os.path.join(args.output_dir, "images_resized")
    out_mask_root = os.path.join(args.output_dir, "masks_pred")
    out_tensor_root = os.path.join(args.output_dir, "tensors_pt")

    count = 0
    for root, _, files in os.walk(args.input_dir):
        for fn in files:
            if not fn.lower().endswith(VALID_EXTS):
                continue

            in_path = os.path.join(root, fn)
            rel_dir = os.path.relpath(root, args.input_dir)
            base = os.path.splitext(fn)[0]

            out_rgb_path = os.path.join(out_rgb_root, rel_dir, f"{base}.png")
            out_mask_path = os.path.join(out_mask_root, rel_dir, f"{base}_mask.png")
            out_tensor_path = os.path.join(out_tensor_root, rel_dir, f"{base}.pt") if args.save_tensor_pt else None

            try:
                img = Image.open(in_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {in_path}: {e}")
                continue

            x_resized, mask = segment_one(
                model=model,
                img_pil=img,
                device=device,
                img_size=img_size,
                threshold=float(args.threshold),
                model_outputs_logits=bool(args.model_outputs_logits),
            )

            save_outputs(
                out_rgb_path=out_rgb_path,
                out_mask_path=out_mask_path,
                x_resized=x_resized,
                mask=mask,
                save_tensor_pt=bool(args.save_tensor_pt),
                out_tensor_path=out_tensor_path,
                normalize_tensor=bool(args.normalize_tensor),
                img_size=img_size,
            )

            count += 1
            if count % 25 == 0:
                print(f"Processed {count} images...")

    print(f"Done. Processed {count} images.")
    print(f"Resized images: {out_rgb_root}")
    print(f"Pred masks: {out_mask_root}")
    if args.save_tensor_pt:
        print(f"Tensors: {out_tensor_root}")


if __name__ == "__main__":
    main()