import os
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import UNet2DModel, DDPMScheduler


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")
MODEL_DIR = Path("outputs_aug/wm_models")

IMAGE_SIZE = 128

BATCH_SIZE = 4

EPOCHS = 20

LEARNING_RATE = 2e-5

NUM_DIFFUSION_STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUGMENTATIONS_PER_IMAGE = 20

# ============================================================
# DATASET
# ============================================================

class WatermarkDataset(Dataset):

    def __init__(self, image_dir):

        self.image_paths = sorted(
            Path(image_dir).glob("*.png")
        )

        self.transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05,
            hue=0.0,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])


    def __len__(self):

        return (
            len(self.image_paths)
            * AUGMENTATIONS_PER_IMAGE
        )


    def __getitem__(self, index):

        original_index = (
            index % len(self.image_paths)
        )

        image_path = self.image_paths[
            original_index
        ]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        return image


# ============================================================
# TRAIN ONE WATERMARK MODEL
# ============================================================

def train_watermark_model(wm_name):

    print("\n")
    print("=" * 60)

    print(
        f"TRAINING MODEL FOR {wm_name}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    image_dir = (
        DATASET_DIR
        / "watermarked_sources"
        / wm_name
    )

    dataset = WatermarkDataset(
        image_dir
    )

    print(
        f"Found {len(dataset)} images"
    )


    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    # --------------------------------------------------------
    # Diffusion model
    # --------------------------------------------------------

    model = UNet2DModel(

        sample_size=IMAGE_SIZE,

        in_channels=3,

        out_channels=3,

        layers_per_block=2,

        block_out_channels=(
            64,
            128,
            256,
            256,
        ),

        down_block_types=(

            "DownBlock2D",

            "DownBlock2D",

            "AttnDownBlock2D",

            "DownBlock2D",

        ),

        up_block_types=(

            "UpBlock2D",

            "AttnUpBlock2D",

            "UpBlock2D",

            "UpBlock2D",

        ),

    ).to(DEVICE)


    # --------------------------------------------------------
    # Noise scheduler
    # --------------------------------------------------------

    noise_scheduler = DDPMScheduler(

        num_train_timesteps=NUM_DIFFUSION_STEPS

    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()


    for epoch in range(EPOCHS):

        epoch_loss = 0.0


        for clean_images in dataloader:

            clean_images = clean_images.to(
                DEVICE
            )


            # -----------------------------------------------
            # Generate Gaussian noise
            # -----------------------------------------------

            noise = torch.randn_like(
                clean_images
            )


            # -----------------------------------------------
            # Random diffusion timestep
            # -----------------------------------------------

            timesteps = torch.randint(

                0,

                NUM_DIFFUSION_STEPS,

                (
                    clean_images.shape[0],
                ),

                device=DEVICE,

            ).long()


            # -----------------------------------------------
            # Add noise to watermarked images
            # -----------------------------------------------

            noisy_images = (
                noise_scheduler.add_noise(

                    clean_images,

                    noise,

                    timesteps,

                )
            )


            # -----------------------------------------------
            # Predict noise
            # -----------------------------------------------

            noise_prediction = model(

                noisy_images,

                timesteps,

            ).sample


            # -----------------------------------------------
            # Diffusion loss
            # -----------------------------------------------

            loss = F.mse_loss(

                noise_prediction,

                noise,

            )


            # -----------------------------------------------
            # Backpropagation
            # -----------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0,

            )

            optimizer.step()


            epoch_loss += loss.item()


        average_loss = (

            epoch_loss
            / len(dataloader)

        )


        if epoch % 10 == 0:

            print(

                f"{wm_name} | "

                f"Epoch {epoch:03d} | "

                f"Loss {average_loss:.6f}"

            )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    output_dir = (

        MODEL_DIR
        / wm_name

    )


    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )


    model.save_pretrained(

        output_dir

    )


    noise_scheduler.save_pretrained(

        output_dir
        / "scheduler"

    )


    print(

        f"Saved {wm_name} model to {output_dir}"

    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        f"Using device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(

            "GPU:",

            torch.cuda.get_device_name(0),

        )


    MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    WATERMARKS = [

        "WM_1",

        "WM_2",

        "WM_3",

        "WM_4",

        "WM_5",

        "WM_6",

        "WM_7",

        "WM_8",

    ]


    for wm_name in WATERMARKS:

        train_watermark_model(
            wm_name
        )


    print("\nALL MODELS TRAINED")