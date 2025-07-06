import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

# Configuration
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
DATA_DIR = "/kaggle/input/eggplant-and-brinjal/Brinjal"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load Full Dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)

# -------------------- Undersampling --------------------
# Group image indices by class
label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_dataset):
    label_to_indices[label].append(idx)

# Minimum class count
min_class_count = min(len(indices) for indices in label_to_indices.values())

# Undersample each class
balanced_indices = []
for label, indices in label_to_indices.items():
    sampled = np.random.choice(indices, min_class_count, replace=False)
    balanced_indices.extend(sampled)

np.random.shuffle(balanced_indices)
balanced_dataset = Subset(full_dataset, balanced_indices)

# Visualize balanced distribution
balanced_labels = [full_dataset.targets[i] for i in balanced_indices]
class_counts = Counter(balanced_labels)
plt.figure(figsize=(10, 5))
sns.barplot(x=[class_names[i] for i in class_counts.keys()], y=list(class_counts.values()))
plt.title('Balanced Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split Balanced Dataset
train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- SimpleViT + CoCa --------------------
vit = SimpleViT(
    image_size=IMAGE_SIZE,
    patch_size=16,
    num_classes=num_classes,
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
        embeddings = self.img_encoder(x)  # shape: (batch_size, num_patches, 512)
        pooled = embeddings.mean(dim=1)
        return self.fc(pooled)

model = CoCa(vit, num_classes).to(DEVICE)

# -------------------- Training --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_acc, val_acc, train_loss, val_loss = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc.item())

    model.eval()
    val_running_loss, val_running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / val_size
    val_epoch_acc = val_running_corrects.double() / val_size
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc.item())

    print(f"Epoch {epoch+1}/{EPOCHS} => Train Acc: {epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f}")

# -------------------- Plotting --------------------
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

# -------------------- Evaluation --------------------
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
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()