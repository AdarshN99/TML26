import torch
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import norm
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission_lira.csv"

# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return index, id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        idx, id_, img, label = super().__getitem__(index)
        return idx, id_, img, label, self.membership[index]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

priv_ds.__class__ = TaskDataset

# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform

pub_loader = DataLoader(pub_ds, batch_size=64, shuffle=False)
priv_loader = DataLoader(priv_ds, batch_size=64, shuffle=False)


# Model 
def get_model():
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    return model

print("Loading target model...")
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Score
def compute_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            # Unpack only the first 4 elements: (idxs, ids, imgs, labels)
            idxs, ids, imgs, labels = batch[:4] 
            
            logits = model(imgs.to(device))
            labels = labels.to(device)
            
            # LiRA Scaled Score: log(p / (1-p))
            y_logit = logits.gather(1, labels.unsqueeze(1)).squeeze()
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, labels.unsqueeze(1), False)
            other_logits = logits[mask].view(logits.size(0), -1)
            max_other = torch.logsumexp(other_logits, dim=1)
            
            s = y_logit - max_other
            scores.extend(zip(idxs.tolist(), s.cpu().numpy()))
    return scores

# Create Shadow Splits
def make_splits(n, K, train_frac=0.5, seed=42):
    rng = np.random.RandomState(seed)
    splits = []

    for _ in range(K):
        perm = rng.permutation(n)
        split = int(train_frac * n)
        train_idx = perm[:split]
        holdout_idx = perm[split:]
        splits.append((set(train_idx), set(holdout_idx)))

    return splits

# Shadow Model Training
def train_shadow(model, loader, device, epochs=6):
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for _, _, imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

    return model


# Training
N = len(pub_ds)
N_priv = len(priv_ds)
K = 32 

in_scores = [[] for _ in range(N)]
out_scores = [[] for _ in range(N)]
priv_shadow_scores = [[] for _ in range(N_priv)] 

splits = make_splits(N, K=K)

for k, (train_idx, holdout_idx) in enumerate(splits):
    print(f"Shadow {k+1}/{K}")
    train_subset = Subset(pub_ds, list(train_idx))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    
    model_k = train_shadow(get_model(), train_loader, device)

    # 1. Collect Public Scores
    full_pub_loader = DataLoader(pub_ds, batch_size=64, shuffle=False)
    scores = compute_scores(model_k, full_pub_loader, device)
    for idx, s in scores:
        if idx in train_idx:
            in_scores[idx].append(s)
        else:
            out_scores[idx].append(s)
            
    # 2. Collect Private Scores (Shadow models serve as the "OUT" distribution)
    full_priv_loader = DataLoader(priv_ds, batch_size=64, shuffle=False)
    p_scores = compute_scores(model_k, full_priv_loader, device)
    for idx, s in p_scores:
        priv_shadow_scores[idx].append(s)
        
    del model_k
    torch.cuda.empty_cache()

# We only need mu and sigma for samples when they are NOT in the training set
params_out = []
for i in range(N):
    mu_out = np.mean(out_scores[i])
    sigma_out = np.std(out_scores[i])
    params_out.append((mu_out, max(sigma_out, 1e-6)))

priv_params_out = []
for i in range(N_priv):
    mu_out = np.mean(priv_shadow_scores[i])
    sigma_out = np.std(priv_shadow_scores[i])
    priv_params_out.append((mu_out, max(sigma_out, 1e-6)))


# LiRA
priv_ids = []
priv_probs = []

# Get target model scores for private set
target_priv_scores = compute_scores(model, DataLoader(priv_ds, batch_size=64), device)
# Ensure they are sorted by index to match priv_shadow_scores
target_priv_scores.sort(key=lambda x: x[0]) 

for i in range(N_priv):
    s_target = target_priv_scores[i][1]
    
    # Calculate stats for this specific image from shadow models
    mu_out = np.mean(priv_shadow_scores[i])
    sigma_out = np.std(priv_shadow_scores[i])
    
    # Use Gaussian CDF for the LLR-based probability
    prob = norm.cdf(s_target, loc=mu_out, scale=max(sigma_out, 1e-6))
    
    priv_ids.append(priv_ds.ids[i])
    priv_probs.append(prob)


# create submission
print("Creating submission...")

df = pd.DataFrame({
    "id": priv_ids,
    "score": priv_probs
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)

target_pub_scores = compute_scores(model, DataLoader(pub_ds, batch_size=64), device)
target_pub_scores.sort()

pub_lira_probs = []
for i in range(N):
    s_target = target_pub_scores[i][1]
    mu, sigma = params_out[i]
    pub_lira_probs.append(norm.cdf(s_target, loc=mu, scale=sigma))

membership_truth = np.array(pub_ds.membership)
pub_lira_probs = np.array(pub_lira_probs)

scores_in = pub_lira_probs[membership_truth == 1]
scores_out = pub_lira_probs[membership_truth == 0]

threshold = np.percentile(scores_out, 95)
tpr = np.mean(scores_in >= threshold)
print(f"Corrected TPR@5%FPR: {tpr:.4f}")