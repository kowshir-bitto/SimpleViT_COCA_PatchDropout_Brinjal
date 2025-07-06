import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

# ---------------- Configuration ----------------
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
DATA_DIR = "/kaggle/input/brinjal-fruit/New folder"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CLAHE Preprocessing ----------------
def apply_clahe(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(clahe_img)

# ---------------- Custom Dataset ----------------
class BrinjalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(set([s[1] for s in samples])))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = apply_clahe(img)
        if self.transform:
            img = self.transform(img)
        label_idx = self.class_to_idx[label]
        return img, label_idx

# ---------------- Oversample Function ----------------
def load_oversampled_data(data_dir):
    data = []
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                data.append((os.path.join(cls_path, fname), cls))

    df = {}
    for cls in sorted(set([label for _, label in data])):
        df[cls] = [s for s in data if s[1] == cls]

    max_len = max(len(v) for v in df.values())
    oversampled = []
    for cls, items in df.items():
        items_resampled = resample(items, replace=True, n_samples=max_len, random_state=42)
        oversampled.extend(items_resampled)

    return oversampled

# ---------------- Prepare Data ----------------
samples = load_oversampled_data(DATA_DIR)
np.random.shuffle(samples)

# Plot class distribution
df_vis = pd.DataFrame(samples, columns=["path", "label"])
plt.figure(figsize=(10, 5))
sns.countplot(data=df_vis, x="label")
plt.title("Balanced Class Distribution")
plt.xticks(rotation=45)
plt.show()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ---------------- Dataset & Dataloaders ----------------
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

train_dataset = BrinjalDataset(train_samples, transform=transform)
val_dataset = BrinjalDataset(val_samples, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- ViT Model ----------------
vit = SimpleViT(
    image_size=IMAGE_SIZE,
    patch_size=16,
    num_classes=len(train_dataset.class_to_idx),
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    patch_dropout=0.5
)

vit = Extractor(vit, return_embeddings_only=True, detach=False)

class CoCa(nn.Module):
    def __init__(self, img_encoder, num_classes):
        super(CoCa, self).__init__()
        self.img_encoder = img_encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        embeddings = self.img_encoder(x)
        pooled = embeddings.mean(dim=1)
        return self.fc(pooled)

model = CoCa(vit, len(train_dataset.class_to_idx)).to(DEVICE)

# ---------------- Training Setup ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_acc, val_acc, train_loss, val_loss = [], [], [], []

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / len(train_dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    model.eval()
    val_loss_epoch, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss_epoch += loss.item() * inputs.size(0)
            val_correct += torch.sum(preds == labels).item()

    val_epoch_loss = val_loss_epoch / len(val_dataset)
    val_epoch_acc = val_correct / len(val_dataset)
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} => Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}")

# ---------------- Plot ----------------
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()

# ---------------- Evaluation ----------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_dataset.idx_to_class.values()))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=train_dataset.idx_to_class.values(),
            yticklabels=train_dataset.idx_to_class.values(),
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
