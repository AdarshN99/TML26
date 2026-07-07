import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.signal
import scipy.ndimage

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")

TEMP_OUT_DIR = Path("submission_temp")
OUTPUT_ZIP = "outputs/submission_best.zip"

clean_dir = DATASET_DIR / "clean_targets"

# Beta: Overall watermark strength for the copy insertion
BETA = 12.0 
# Alpha NVF: 1.0 concentrates watermark on edges/textures, 0.0 on flat areas
NVF_ALPHA = 1.0 

TEMP_OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# MAPPING
# ============================================================

CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

# ============================================================
# HELPERS
# ============================================================

def load_image(path):
    return np.array(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    )


def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def wiener_filter_rgb(img, size=5):
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = scipy.signal.wiener(img[:, :, c], (size, size))
    return out


def compute_nvf_weight(target_img, alpha_nvf):
    # Calculate luminance in [0, 1] range
    lum = np.dot(target_img, [0.2989, 0.5870, 0.1140]) / 255.0
    
    # Compute local variance over a 5x5 window
    mean_sq = scipy.ndimage.uniform_filter(lum**2, size=5)
    sq_mean = scipy.ndimage.uniform_filter(lum, size=5)**2
    local_var = np.maximum(mean_sq - sq_mean, 0)
    
    # Tuning parameter theta
    sigma_max_sq = np.max(local_var)
    theta = 100.0 / (sigma_max_sq + 1e-8)
    
    # Non-stationary Gaussian NVF
    nvf = 1.0 / (1.0 + local_var * theta)
    
    # Final Weight
    W = ((1.0 - nvf) * alpha_nvf + nvf * (1.0 - alpha_nvf)) * lum
    
    return np.expand_dims(W, axis=-1)


def predict_watermark_batch(image_paths):
    image_paths = list(image_paths)
    
    if len(image_paths) == 0:
        raise RuntimeError("No images found.")
        
    first = load_image(image_paths[0])
    w_hat_accumulator = np.zeros_like(first)
    
    for p in image_paths:
        y = load_image(p)
        if y.shape != first.shape:
            raise RuntimeError(f"Shape mismatch: {image_paths[0]} -> {first.shape} | {p} -> {y.shape}")
            
        # Estimate the cover image (x_hat) using denoising
        x_hat = wiener_filter_rgb(y, size=5)
        
        # Predict watermark
        w_hat = y - x_hat
        w_hat_accumulator += w_hat
        
    return w_hat_accumulator / len(image_paths)

# ============================================================
# EXTRACT WATERMARK PATTERNS
# ============================================================

print("\nExtracting watermark patterns via Wiener Denoising...")

watermark_patterns = {}

for wm_name, start_idx, end_idx in CATEGORIES:

    print(f"\nProcessing {wm_name}")
    wm_dir = DATASET_DIR / "watermarked_sources" / wm_name
    wm_paths = sorted(wm_dir.glob("*.png"))

    print(f"Watermarked images: {len(wm_paths)}")
    
    # Predict the watermark solely from the stego images (no clean targets used here)
    w_hat = predict_watermark_batch(wm_paths)
    watermark_patterns[wm_name] = w_hat

    print(f"Pattern shape: {w_hat.shape}")

# ============================================================
# FORGE TARGET IMAGES
# ============================================================

print("\nForging images with NVF Masking...")

processed = 0

for wm_name, start_idx, end_idx in CATEGORIES:

    w_hat = watermark_patterns[wm_name]

    print(f"{wm_name} -> {start_idx}.png to {end_idx}.png")

    for img_idx in range(start_idx, end_idx + 1):

        target_path = clean_dir / f"{img_idx}.png"
        target_img = load_image(target_path)

        if target_img.shape != w_hat.shape:
            raise RuntimeError(f"Target shape mismatch\n{target_path}\ntarget={target_img.shape}\ndelta={w_hat.shape}")

        # Compute NVF weighting matrix for the target image
        W = compute_nvf_weight(target_img, NVF_ALPHA)
        
        # Copy Insertion
        forged = target_img + BETA * W * np.sign(w_hat)
        
        forged = np.clip(forged, 0, 255)

        out_path = TEMP_OUT_DIR / f"{img_idx}.png"
        save_image(forged, out_path)
        
        processed += 1

print(f"\nSuccessfully forged {processed} images.")

# ============================================================
# PACKAGE ZIP
# ============================================================

print("\nCreating submission zip...")

with zipfile.ZipFile(
    OUTPUT_ZIP,
    "w",
    zipfile.ZIP_DEFLATED
) as zf:

    for img_path in sorted(
        TEMP_OUT_DIR.glob("*.png"),
        key=lambda p: int(p.stem)
    ):
        zf.write(img_path, arcname=img_path.name)

print(f"\nSaved submission: {OUTPUT_ZIP}")