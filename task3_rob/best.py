import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from torchvision.models import models
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import os, time, copy

# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).parent

DATA_PATH   = BASE / "train.npz"          
SAVE_PATH   = "best.pt"

ARCH        = "resnet50" 

NUM_CLASSES = 9

IMG_SIZE    = 32
BATCH_SIZE  = 128
EPOCHS      = 100

EPS         = 8 / 255      
PGD_ALPHA   = 2 / 255      
PGD_STEPS   = 10          
PGD_RAND    = True        

LR          = 0.1
MOMENTUM    = 0.9
WEIGHT_DECAY= 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATA
# ============================================================

class NpzDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images      
        self.labels    = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]          
        if img.shape[0] == 3:           
            img = img.transpose(1, 2, 0)
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_loaders(path):
    data   = np.load(path)
    images = data["data"] if "data" in data else data["images"]
    labels = data["labels"] if "labels" in data else data["label"]

    # Augmentations for training
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),                       
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])

    full = NpzDataset(images, labels, transform=None)
    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    train_idx, val_idx = random_split(
        range(len(full)), [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = NpzDataset(images[list(train_idx)], labels[list(train_idx)], train_tf)
    val_ds   = NpzDataset(images[list(val_idx)],   labels[list(val_idx)],   val_tf)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,   
        batch_size=256,        
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    return train_loader, val_loader

# ============================================================
# MODEL
# ============================================================

def build_model(arch=ARCH):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")


    model.fc      = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS, rand_init=PGD_RAND):
    x_adv = x.detach().clone()

    if rand_init:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            loss   = nn.CrossEntropyLoss()(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    return x_adv

# ============================================================
# FGSM WITH RANDOM START
# ============================================================

def fgsm_attack(model, x, y, eps=EPS):
    x_adv = x.detach().clone() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv.requires_grad_(True)
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
    grad  = torch.autograd.grad(loss, x_adv)[0]
    x_adv = torch.clamp(x_adv.detach() + eps * grad.sign(), 0.0, 1.0)
    return x_adv.detach()

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


def evaluate_robust(model, loader, eps=EPS, steps=20):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y, eps=eps, alpha=eps/4, steps=steps)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total

# ============================================================
# TRAINING
# ============================================================

def train(model, train_loader, val_loader):
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

    scaler    = GradScaler()          
    criterion = nn.CrossEntropyLoss()

    best_score  = 0.0
    best_state  = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        # FGSM for first 10 epochs, then switch to PGD
        use_pgd = epoch > 10

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Generate adversarial examples
            model.eval()          # BN in eval mode during attack 
            with torch.no_grad():
                pass
            if use_pgd:
                x_adv = pgd_attack(model, x, y)
            else:
                x_adv = fgsm_attack(model, x, y)
            model.train()

            optimizer.zero_grad()
            with autocast():
                logits = model(x_adv)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            correct    += (logits.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()

        train_loss = total_loss / total
        train_acc  = correct / total

        # Validation for every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            clean_acc  = evaluate_clean(model, val_loader)
            robust_acc = evaluate_robust(model, val_loader, steps=20)
            score      = 0.5 * clean_acc + 0.5 * robust_acc

            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"Loss: {train_loss:.4f} "
                  f"Clean Accuracy: {clean_acc:.3f}  "
                  f"Adversarial Accuracy: {robust_acc:.3f}  "
            )

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, SAVE_PATH)
                print(f"Saved best model (score={best_score:.4f})")
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"Loss={train_loss:.4f} "
            )
    return best_state


if __name__ == "__main__":
    train_loader, val_loader = get_loaders(DATA_PATH)
    model = build_model(ARCH)

    best_state = train(model, train_loader, val_loader)

    # Final check on full val set
    model.load_state_dict(best_state)
    clean  = evaluate_clean(model, val_loader)
    robust = evaluate_robust(model, val_loader, steps=50)  
    print(f"\nTraining Finished. Final "
          f"Clean Accuracy: {clean:.4f}  "
          f"Adversial Accuracy: {robust:.4f}  "
    )