import zipfile
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DDIMScheduler,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")
MODEL_DIR = Path("wm_models")
OUTPUT_DIR = Path("inference_outputs_sweep")

IMAGE_SIZE = 128
NUM_DIFFUSION_STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {
    "t5": {
        "timestep": 5,
        "residual_alpha": None,
    },

    "t10": {
        "timestep": 10,
        "residual_alpha": None,
    },

    "t20": {
        "timestep": 20,
        "residual_alpha": None,
    },

    "t10_residual_alpha01": {
        "timestep": 10,
        "residual_alpha": 0.1,
    },
}


# ============================================================
# WATERMARK / TARGET MAPPING
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
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ),
])


# ============================================================
# LOAD TARGET
# ============================================================

def load_target(image_path):

    image = Image.open(image_path).convert("RGB")

    original_size = image.size

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE), original_size


# ============================================================
# TENSOR TO IMAGE
# ============================================================

def tensor_to_image(tensor, original_size):

    tensor = tensor.detach().cpu().squeeze(0)

    tensor = tensor * 0.5 + 0.5

    tensor = tensor.clamp(0, 1)

    array = tensor.permute(1, 2, 0).numpy()

    array = (array * 255).astype(np.uint8)

    image = Image.fromarray(array)

    image = image.resize(
        original_size,
        Image.Resampling.LANCZOS,
    )

    return image


# ============================================================
# SHALLOW DIFFUSION
# ============================================================

@torch.no_grad()
def diffusion_reconstruction(
    target,
    model,
    ddpm_scheduler,
    ddim_scheduler,
    shallow_timestep,
):

    noise = torch.randn_like(target)

    timestep = torch.tensor(
        [shallow_timestep],
        device=DEVICE,
        dtype=torch.long,
    )

    noisy_target = ddpm_scheduler.add_noise(
        target,
        noise,
        timestep,
    )

    sample = noisy_target

    ddim_scheduler.set_timesteps(
        NUM_DIFFUSION_STEPS,
        device=DEVICE,
    )

    inference_timesteps = [
        t
        for t in ddim_scheduler.timesteps
        if t <= shallow_timestep
    ]

    for t in inference_timesteps:

        model_output = model(
            sample,
            t,
        ).sample

        sample = ddim_scheduler.step(
            model_output,
            t,
            sample,
        ).prev_sample

    return sample


# ============================================================
# APPLY EXPERIMENT
# ============================================================

def apply_experiment(
    target,
    diffusion_output,
    residual_alpha,
):

    if residual_alpha is None:

        return diffusion_output

    residual = diffusion_output - target

    forged = (
        target
        + residual_alpha * residual
    )

    return forged


# ============================================================
# PROCESS WATERMARK MODEL
# ============================================================

def process_watermark(
    wm_name,
    target_start,
    target_end,
):

    print("\n")
    print("=" * 60)
    print(f"PROCESSING {wm_name}")
    print("=" * 60)

    model_path = MODEL_DIR / wm_name

    model = UNet2DModel.from_pretrained(
        model_path
    )

    model = model.to(DEVICE)

    model.eval()

    scheduler_path = model_path / "scheduler"

    ddpm_scheduler = DDPMScheduler.from_pretrained(
        scheduler_path
    )

    ddim_scheduler = DDIMScheduler.from_config(
        ddpm_scheduler.config
    )

    target_dir = DATASET_DIR / "clean_targets"

    for image_number in range(
        target_start,
        target_end + 1,
    ):

        target_path = (
            target_dir
            / f"{image_number}.png"
        )

        target, original_size = load_target(
            target_path
        )

        for experiment_name, config in EXPERIMENTS.items():

            shallow_timestep = config["timestep"]

            residual_alpha = config["residual_alpha"]

            diffusion_output = diffusion_reconstruction(
                target,
                model,
                ddpm_scheduler,
                ddim_scheduler,
                shallow_timestep,
            )

            forged_tensor = apply_experiment(
                target,
                diffusion_output,
                residual_alpha,
            )

            experiment_dir = (
                OUTPUT_DIR
                / experiment_name
            )

            experiment_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                experiment_dir
                / f"{image_number}.png"
            )

            forged_image = tensor_to_image(
                forged_tensor,
                original_size,
            )

            forged_image.save(output_path)

        print(
            f"{wm_name}: saved variants for "
            f"{image_number}.png"
        )

    del model

    torch.cuda.empty_cache()


# ============================================================
# CREATE ZIP FILES
# ============================================================

def create_submission_zips():

    print("\nCreating submission ZIP files...")

    for experiment_name in EXPERIMENTS:

        experiment_dir = (
            OUTPUT_DIR
            / experiment_name
        )

        image_paths = sorted(
            experiment_dir.glob("*.png"),
            key=lambda path: int(path.stem),
        )

        print(
            f"{experiment_name}: "
            f"{len(image_paths)} images"
        )

        if len(image_paths) != 200:

            raise RuntimeError(
                f"{experiment_name} does not "
                f"contain exactly 200 images."
            )

        zip_path = (
            OUTPUT_DIR
            / f"submission_{experiment_name}.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zip_file:

            for image_path in image_paths:

                zip_file.write(
                    image_path,
                    arcname=image_path.name,
                )

        print(
            f"Created {zip_path}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        wm_name,
        target_start,
        target_end,
    ) in CATEGORIES:

        process_watermark(
            wm_name,
            target_start,
            target_end,
        )

    create_submission_zips()

    print("\nSWEEP COMPLETE")