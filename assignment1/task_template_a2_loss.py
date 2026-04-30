import torch
import pandas as pd

from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission_loss.csv"

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
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


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

# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# approach:2 Negative Loss
all_ids = []
all_scores = []

criterion = torch.nn.CrossEntropyLoss(reduction="none")

with torch.no_grad():
    for id_, imgs, labels in priv_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels).cpu().numpy()
        scores = -loss

        all_ids.extend(id_.tolist())
        all_scores.extend(scores.tolist())
    

# create submission
print("Creating submission...")

df = pd.DataFrame({
    "id": all_ids,
    "score": all_scores
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)