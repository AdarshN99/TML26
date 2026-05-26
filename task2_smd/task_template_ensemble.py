import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).parent
DATA_ROOT = BASE / "data"
TARGET_MODEL_PATH = BASE / "target_model/weights.safetensors" 
SUSPECT_MODELS_DIR = BASE /  "suspect_models"
OUTPUT_CSV = BASE / "outputs/submission_ensemble.csv"

# ============================================================
# MODEL
# ============================================================

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    model = make_model()
    state_dict = load_file(path, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
])

dataset = datasets.CIFAR100(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform,
)

BATCH_SIZE = 128
NUM_WORKERS = 4

NUM_PROBE_BATCHES = 20
EPSILON = 2 / 255
FGSM_ALPHA = 1 / 255

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.hooks = []
        target_layers = {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }

        for name, layer in target_layers.items():
            hook = layer.register_forward_hook(
                self.save_output(name)
            )
            self.hooks.append(hook)

    def save_output(self, name):
        def fn(_, __, output):
            self.features[name] = output.detach()
        return fn

    def forward(self, x):
        logits = self.model(x)
        return logits, self.features


# ============================================================
# ADVERSARIAL GENERATION
# ============================================================

def generate_fgsm(model, x, y):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + FGSM_ALPHA * grad.sign()
    x_adv = torch.clamp(
        x_adv,
        min=-3,
        max=3,
    )

    return x_adv.detach(), grad.detach()


# ============================================================
# SIMILARITY METRICS
# ============================================================

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()

'''
def normalized_l2(a, b):
    a = a.flatten()
    b = b.flatten() 
    dist = torch.norm(a - b)
    denom = (torch.norm(a) + torch.norm(b) + 1e-8)
    score = 1.0 - (dist / denom)
    return score.item()
'''

def feature_similarity(f1, f2):
    sims = []
    layer_weights = {
        "layer1": 0.35,
        "layer2": 0.30,
        "layer3": 0.20,
        "layer4": 0.15,
    }

    for layer in layer_weights:
        x1 = F.adaptive_avg_pool2d(f1[layer], 1).flatten(1)
        x2 = F.adaptive_avg_pool2d(f2[layer],1).flatten(1)
        sim = F.cosine_similarity(x1, x2, dim=1).mean()
        sims.append(sim.item() * layer_weights[layer])
    return sum(sims)


def logit_similarity(z1, z2): 
    z1 = z1 - z1.mean(dim=1, keepdim=True)
    z2 = z2 - z2.mean(dim=1, keepdim=True)
    sim = F.cosine_similarity(z1, z2, dim=1).mean()
    return sim.item()


def gradient_similarity(g1, g2):
    g1 = g1.flatten(1)
    g2 = g2.flatten(1)
    sim = F.cosine_similarity(g1, g2, dim=1).mean()
    return sim.item()


# ============================================================
# WEIGHT SIMILARITY
# ============================================================

def model_weight_similarity(target_model, suspect_model):
    scores = []
    target_sd = target_model.state_dict()
    suspect_sd = suspect_model.state_dict()

    for k in target_sd:
        if "weight" not in k:
            continue
        if target_sd[k].shape != suspect_sd[k].shape:
            continue
        s = cosine_similarity(target_sd[k].float(), suspect_sd[k].float())
        scores.append(s)
    return np.mean(scores)


# ============================================================
# MAIN EVALUATION
# ============================================================

target_model = load_model(TARGET_MODEL_PATH)

target_extractor = FeatureExtractor(target_model)

suspect_paths = sorted(
    glob.glob(
        os.path.join(
            SUSPECT_MODELS_DIR,
            "*.safetensors"
        )
    )
)

results = []

# ------------------------------------------------------------
# PRECOMPUTE TARGET FEATURES
# ------------------------------------------------------------

target_batches = []

print("Precomputing target fingerprints...")

for batch_idx, (x, y) in enumerate(loader):
    if batch_idx >= NUM_PROBE_BATCHES:
        break
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    with torch.no_grad():
        target_logits, target_feats = target_extractor(x)

    x_adv, target_grad = generate_fgsm(target_model, x, y)

    with torch.no_grad():
        adv_logits, adv_feats = target_extractor(x_adv)

    target_batches.append({
        "clean_x": x,
        "adv_x": x_adv,
        "target_logits": target_logits.detach(),
        "adv_logits": adv_logits.detach(),
        "target_feats": {
            k: v.detach()
            for k, v in target_feats.items()
        },
        "adv_feats": {
            k: v.detach()
            for k, v in adv_feats.items()
        },
        "target_grad": target_grad.detach(),
    })

# ------------------------------------------------------------
# EVALUATE SUSPECTS
# ------------------------------------------------------------

for idx, suspect_path in enumerate(suspect_paths):

    print(f"[{idx+1}/{len(suspect_paths)}]")

    suspect_model = load_model(suspect_path)

    suspect_extractor = FeatureExtractor(suspect_model)

    weight_scores = []
    feature_scores = []
    logit_scores = []
    grad_scores = []
    adv_logit_scores = []

    for batch in target_batches:
        x = batch["clean_x"]
        x_adv = batch["adv_x"]

        # ----------------------------------------------------
        # CLEAN INPUTS
        # ----------------------------------------------------

        with torch.no_grad():
            suspect_logits, suspect_feats = (suspect_extractor(x))

        # ----------------------------------------------------
        # ADVERSARIAL INPUTS
        # ----------------------------------------------------

        x_adv_var = x_adv.clone().detach().requires_grad_(True)

        suspect_adv_logits, suspect_adv_feats = (suspect_extractor(x_adv_var))

        loss = suspect_adv_logits.mean()
        suspect_model.zero_grad()
        loss.backward()
        suspect_grad = (x_adv_var.grad.detach())

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        fs = feature_similarity(
            batch["target_feats"],
            suspect_feats,
        )

        ls = logit_similarity(
            batch["target_logits"],
            suspect_logits,
        )

        als = logit_similarity(
            batch["adv_logits"],
            suspect_adv_logits,
        )

        gs = gradient_similarity(
            batch["target_grad"],
            suspect_grad,
        )

        feature_scores.append(fs)
        logit_scores.append(ls)
        adv_logit_scores.append(als)
        grad_scores.append(gs)

    # --------------------------------------------------------
    # WEIGHT SCORE
    # --------------------------------------------------------

    ws = model_weight_similarity(target_model, suspect_model)

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    feature_score = np.mean(feature_scores)
    logit_score = np.mean(logit_scores)
    adv_logit_score = np.mean(adv_logit_scores)
    grad_score = np.mean(grad_scores)

    final_score = (
        0.15 * ws +
        0.30 * feature_score +
        0.20 * logit_score +
        0.20 * adv_logit_score +
        0.15 * grad_score
    )

    final_score = float(np.clip(final_score, 0.0, 1.0))

    results.append({
        "id": idx,
        "score": final_score,
    })

    del suspect_model
    torch.cuda.empty_cache()

# ============================================================
# SCORE SHARPENING
# ============================================================

scores = np.array(
    [r["score"] for r in results]
)

scores = (scores - scores.min()) / (
    scores.max() - scores.min() + 1e-8
)

scores = scores ** 2

for i in range(len(results)):
    results[i]["score"] = float(scores[i])

# ============================================================
# SAVE CSV
# ============================================================

submission_df = pd.DataFrame(results)

submission_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(f"Saved to: {OUTPUT_CSV}")