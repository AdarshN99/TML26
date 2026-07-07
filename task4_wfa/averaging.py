import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")

TEMP_OUT_DIR = Path("submission_temp")
OUTPUT_ZIP = "outputs/submission_averaging.zip"

clean_dir = DATASET_DIR / "clean_targets"

ALPHA = 0.10

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


def average_images(image_paths):

    image_paths = list(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found."
        )

    first = load_image(image_paths[0])

    accumulator = np.zeros_like(first)

    for p in image_paths:

        img = load_image(p)

        if img.shape != first.shape:
            raise RuntimeError(
                f"Shape mismatch inside batch:\n"
                f"{image_paths[0]} -> {first.shape}\n"
                f"{p} -> {img.shape}"
            )

        accumulator += img

    accumulator /= len(image_paths)

    return accumulator


# ============================================================
# EXTRACT WATERMARK PATTERNS
# ============================================================

print("\nExtracting watermark patterns...")

watermark_patterns = {}

for wm_name, start_idx, end_idx in CATEGORIES:

    print(f"\nProcessing {wm_name}")

    wm_dir = DATASET_DIR / "watermarked_sources" / wm_name

    wm_paths = sorted(wm_dir.glob("*.png"))

    clean_paths = [
        clean_dir / f"{i}.png"
        for i in range(start_idx, end_idx + 1)
    ]

    print(f"Watermarked images: {len(wm_paths)}")
    print(f"Clean images: {len(clean_paths)}")

    wm_avg = average_images(wm_paths)
    clean_avg = average_images(clean_paths)

    delta = wm_avg - clean_avg

    watermark_patterns[wm_name] = delta

    print(
        f"Pattern shape: {delta.shape}"
    )

# ============================================================
# FORGE TARGET IMAGES
# ============================================================

print("\nForging images...")

processed = 0

for wm_name, start_idx, end_idx in CATEGORIES:

    delta = watermark_patterns[wm_name]

    print(
        f"{wm_name} -> "
        f"{start_idx}.png to {end_idx}.png"
    )

    for img_idx in range(start_idx, end_idx + 1):

        target_path = clean_dir / f"{img_idx}.png"

        target_img = load_image(target_path)

        if target_img.shape != delta.shape:

            raise RuntimeError(
                f"Target shape mismatch\n"
                f"{target_path}\n"
                f"target={target_img.shape}\n"
                f"delta={delta.shape}"
            )

        forged = target_img + ALPHA * delta

        forged = np.clip(
            forged,
            0,
            255
        )

        out_path = TEMP_OUT_DIR / f"{img_idx}.png"

        save_image(
            forged,
            out_path
        )

        processed += 1

print(
    f"\nSuccessfully forged "
    f"{processed} images."
)

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

        zf.write(
            img_path,
            arcname=img_path.name
        )

print(f"\nSaved submission: {OUTPUT_ZIP}")