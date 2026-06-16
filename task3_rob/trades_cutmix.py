import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import copy, time, os

# ============================================================
#  CONFIG 
# ============================================================

DATA_PATH    = "train.npz"
SAVE_PATH    = "trades_cutmix.pt"
ARCH         = "resnet50"    

NUM_CLASSES  = 9
BATCH        = 128
EPOCHS       = 100

EPS          = 8 / 255
ALPHA        = 2 / 255
PGD_STEPS    = 10
BETA         = 6.0             

LR           = 0.01             
WD           = 5e-4
LABEL_SMOOTH = 0.1

VAL_FRAC     = 0.1
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  Arch: {ARCH}")


# ============================================================
#  CUTMIX AUGMENTATION
# ============================================================

def cutmix_data(x, y, alpha=1.0):
    lam   = np.random.beta(alpha, alpha)
    B     = x.size(0)
    idx   = torch.randperm(B, device=x.device)
    _, _, H, W = x.shape

    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx    = np.random.randint(W)
    cy    = np.random.randint(H)
    x1    = max(cx - cut_w // 2, 0)
    x2    = min(cx + cut_w // 2, W)
    y1    = max(cy - cut_h // 2, 0)
    y2    = min(cy + cut_h // 2, H)

    x_mix         = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj       = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x_mix, y, y[idx], lam_adj


# ============================================================
#  LOAD DATA
# ============================================================

class NpzDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels.astype(np.int64)
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)           
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_loaders():
    data   = np.load(DATA_PATH)
    keys   = list(data.keys())
    images = data.get("data",   data.get("images", data.get("X",      data[keys[0]])))
    labels = data.get("labels", data.get("label",  data.get("y",      data[keys[1]])))
    print(f"Loaded {len(labels)} samples | keys={keys} | image shape={images[0].shape}")

    rng   = np.random.default_rng(SEED)
    idx   = np.arange(len(labels))
    rng.shuffle(idx)
    n_val = int(VAL_FRAC * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Training: crop + flip (CutMix applied manually in train loop)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])

    tl = DataLoader(NpzDataset(images[train_idx], labels[train_idx], train_tf),
                    batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    vl = DataLoader(NpzDataset(images[val_idx],   labels[val_idx],   val_tf),
                    batch_size=256,  shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    return tl, vl


# ============================================================
#  MODEL
# ============================================================

def build_model(arch=ARCH, pretrained=True):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    model.fc      = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


# ============================================================
#  TRADES LOSS
# ============================================================

def trades_loss(model, x_nat, y, beta=BETA):
    model.eval()                         # freeze BN during inner attack
    x_adv = (x_nat + 0.001 * torch.randn_like(x_nat)).clamp(0, 1).detach()

    with torch.no_grad():
        p_nat = F.softmax(model(x_nat), dim=1)

    for _ in range(PGD_STEPS):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                          p_nat, reduction="batchmean")
        grad  = torch.autograd.grad(kl, x_adv)[0]
        x_adv = (x_nat + torch.clamp(
            x_adv.detach() + ALPHA * grad.sign() - x_nat, -EPS, EPS
        )).clamp(0, 1).detach()

    model.train()

    logits_nat = model(x_nat)
    logits_adv = model(x_adv)

    # Label smoothing on natural loss
    loss_nat = F.cross_entropy(logits_nat, y, label_smoothing=LABEL_SMOOTH)
    loss_rob = F.kl_div(F.log_softmax(logits_adv, dim=1),
                        F.softmax(logits_nat, dim=1),
                        reduction="batchmean")
    return loss_nat + beta * loss_rob, logits_adv


# ============================================================
#  EVALUATION
# ============================================================

def eval_both(model, loader, pgd_steps=20):
    model.eval()
    c_clean = c_rob = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            c_clean += (model(x).argmax(1) == y).sum().item()

        x_adv = (x + torch.zeros_like(x).uniform_(-EPS, EPS)).clamp(0, 1).detach()
        for _ in range(pgd_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), y)
            grad  = torch.autograd.grad(loss, x_adv)[0]
            x_adv = (x + torch.clamp(
                x_adv.detach() + ALPHA * grad.sign() - x, -EPS, EPS
            )).clamp(0, 1).detach()

        with torch.no_grad():
            c_rob += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)

    return c_clean / total, c_rob / total


# ============================================================
#  TRAINING 
# ============================================================

def train():
    train_loader, val_loader = get_loaders()
    model     = build_model(ARCH, pretrained=True)

    optimizer = optim.SGD(
        model.parameters(), 
        lr=LR,
        momentum=0.9, 
        weight_decay=WD, 
        nesterov=True
    )

    # Drop LR at 50% and 75% of training
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(EPOCHS * 0.50), int(EPOCHS * 0.75)],
        gamma=0.1
    )
    scaler = GradScaler()

    best_score, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Apply CutMix with probability 0.5
            if np.random.random() < 0.5:
                x, y_a, y_b, lam = cutmix_data(x, y)
                optimizer.zero_grad()
                with autocast():
                    loss_a, logits_adv = trades_loss(model, x, y_a)
                    loss_b, _          = trades_loss(model, x, y_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
            else:
                optimizer.zero_grad()
                with autocast():
                    loss, logits_adv = trades_loss(model, x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            correct    += (logits_adv.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()

        # Validate for every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            clean, robust = eval_both(model, val_loader, pgd_steps=20)
            score = 0.5 * clean + 0.5 * robust
            print(f"Epoch {epoch:3d}/{EPOCHS} loss={total_loss/total:.4f} "
                  f"adv_tr={correct/total:.3f}  "
                  f"clean={clean:.3f} robust={robust:.3f} score={score:.3f} "
            )

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, SAVE_PATH)
                print(f"New best score={best_score:.4f}")
        else:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss={total_loss/total:.4f} "
                  f"adv_tr={correct/total:.3f} "
            )

    # Final evaluation
    model.load_state_dict(best_state)
    clean, robust = eval_both(model, val_loader, pgd_steps=50)
    print(f"\n{'='*60}")
    print(f"FINAL clean: {clean:.4f}  robust: {robust:.4f}  "
          f"score={0.5*clean+0.5*robust:.4f}")

    return best_score

if __name__ == "__main__":
    score = train()
    print(f"\nBest unified score: {score:.4f}")