# resnet_model_light.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import os

# === Настройки ===
DATA_PATH = "./datasets/fer2013/train"
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. Датасет ===
class FERDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.samples = []
        for label in os.listdir(folder):
            for f in os.listdir(os.path.join(folder, label)):
                if f.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(folder, label, f), label))
        self.labels = sorted(list(set(l for _, l in self.samples)))
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), self.label2id[label]


# === 2. Модель ===
def build_resnet(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# === 3. Тренировка ===
if __name__ == "__main__":
    ds = FERDataset(DATA_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = build_resnet(num_classes=len(ds.labels)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Обучение ResNet на {DEVICE} ({len(ds)} изображений)")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for x, y in tqdm(dl, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Средний loss: {running_loss/len(dl):.4f}")

    torch.save(model.state_dict(), "resnet_emotion_light.pth")
    print("✅ Модель сохранена: resnet_emotion_light.pth")
