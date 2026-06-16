import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from torchvision.models import resnet50
from torch.utils.data import TensorDataset, DataLoader, random_split

# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).parent

SEED = 42

NUM_CLASSES = 9

BATCH_SIZE = 128
EPOCHS = 50

STOP = 10

LR = 0.02

EPSILON = 8.0 / 255.0
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

PGD_ALPHA = 2.0 / 255.0
PGD_STEPS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

data = np.load("train.npz")

images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Classes:", torch.unique(labels))

dataset = TensorDataset(images, labels)

# ============================================================
# TRAIN - VALIDATION SPLIT
# ============================================================

train_size = 45000
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================


model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(
    model,
    x,
    y,
    epsilon,
    alpha,
    steps
):

    x_adv = x.detach().clone()

    x_adv += torch.empty_like(x_adv).uniform_(
        -epsilon,
        epsilon
    )

    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):

        x_adv.requires_grad_(True)

        logits = model(x_adv)

        loss = criterion(logits, y)

        grad = torch.autograd.grad(
            loss,
            x_adv,
            retain_graph=False,
            create_graph=False
        )[0]

        with torch.no_grad():

            x_adv = x_adv + alpha * grad.sign()

            delta = torch.clamp(
                x_adv - x,
                min=-epsilon,
                max=epsilon
            )

            x_adv = torch.clamp(
                x + delta,
                0.0,
                1.0
            )

    return x_adv.detach()

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_clean(model, loader):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


def evaluate_adv(model, loader):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        x_adv = pgd_attack(
            model,
            x,
            y,
            EPSILON,
            PGD_ALPHA,
            PGD_STEPS
            )

        with torch.no_grad():
            logits = model(x_adv)

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total

# ============================================================
# TRAINING
# ============================================================

best_adv_acc = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        x_adv = pgd_attack(
            model,
            x,
            y,
            EPSILON,
            PGD_ALPHA,
            PGD_STEPS
        )

        optimizer.zero_grad()

        logits = model(x_adv)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    clean_acc = evaluate_clean(
        model,
        val_loader
    )

    adv_acc = evaluate_adv(
        model,
        val_loader
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Clean Accuracy: {clean_acc:.2f}% "
        f"Adversarial Accuracy: {adv_acc:.2f}%"
    )

    if clean_acc >= 50 and adv_acc > best_adv_acc:

        best_adv_acc = adv_acc
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BASE / "pgd_model.pt"
        )

        print(
            f"Saved best model "
            f"(adversarial accuracy = {adv_acc:.2f}%)"
        )
    else:
        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

    if epochs_without_improvement >= STOP:

        print(
            f"Early stopped "
            f"after {epoch + 1} epochs."
        )

        break

print("Training finished. Best adversarial accuracy:", best_adv_acc)