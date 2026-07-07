import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")

TEMP_OUT_DIR = Path("submission_temp")
OUTPUT_ZIP = "outputs/submission_residual_wm.zip"


ALPHA = 0.5

# PCA 
N_COMPONENTS = 3

TEMP_OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# DATASET MAPPING
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

clean_dir = DATASET_DIR / "clean_targets"

# ============================================================
# HELPERS
# ============================================================

def load_image(path):
    return np.array(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    )


def save_image(arr, path):

    arr = np.clip(arr, 0, 255)

    Image.fromarray(
        arr.astype(np.uint8)
    ).save(path)


# ============================================================
# PCA WATERMARK EXTRACTION
# ============================================================

print("\nExtracting PCA watermark patterns...")

watermark_patterns = {}

for wm_name, start_idx, end_idx in CATEGORIES:

    print(f"\nProcessing {wm_name}")

    wm_dir = DATASET_DIR / "watermarked_sources" / wm_name

    wm_paths = sorted(
        wm_dir.glob("*.png")
    )

    residuals = []

    reference_shape = None

    for wm_path in wm_paths:

        wm = load_image(wm_path)
        
        if reference_shape is None:
            reference_shape = wm.shape

        residual = wm - gaussian_filter(wm, sigma=5)

        residuals.append(
        	residual.reshape(-1)
    	)
    residuals = np.stack(
        residuals,
        axis=0
    )

    print(
        "Residual matrix:",
        residuals.shape
    )

    pca = PCA(
        n_components=min(
            N_COMPONENTS,
            residuals.shape[0]
        )
    )

    pca.fit(residuals)

    principal_component = (
        pca.components_[0]
        .reshape(reference_shape)
    )

    mean_residual = (
        residuals.mean(axis=0)
        .reshape(reference_shape)
    )

    watermark = (
        0.5 * mean_residual +
        0.5 * principal_component
    )

    watermark_patterns[wm_name] = watermark

    print(
        "Explained variance:",
        pca.explained_variance_ratio_[0]
    )

    vis = watermark.copy()

    vis -= vis.min()

    if vis.max() > 0:
        vis /= vis.max()

    vis *= 255

    Image.fromarray(
        vis.astype(np.uint8)
    ).save(
        f"{wm_name}_pca_watermark.png"
    )

# ============================================================
# FORGE IMAGES
# ============================================================

print("\nForging images...")

processed = 0

for wm_name, start_idx, end_idx in CATEGORIES:

    watermark = watermark_patterns[wm_name]

    for img_idx in range(start_idx, end_idx + 1):

        target_path = (
            clean_dir /
            f"{img_idx}.png"
        )

        clean = load_image(target_path)

        if clean.shape != watermark.shape:

            raise RuntimeError(
                f"Shape mismatch during forging\n"
                f"{target_path}\n"
                f"clean={clean.shape}\n"
                f"watermark={watermark.shape}"
            )

        forged = (
            clean
            + ALPHA * watermark
        )

        forged = np.clip(
            forged,
            0,
            255
        )

        out_path = (
            TEMP_OUT_DIR /
            f"{img_idx}.png"
        )

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

print(
    f"\nSaved submission to "
    f"{OUTPUT_ZIP}"
)
