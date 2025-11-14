import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Используем устройство: {DEVICE}")

# === Загрузка данных ===
data = torch.load("fusion_dataset.pt", map_location=DEVICE)
X_img, X_aud, X_txt, y = data["img"], data["aud"], data["txt"], data["label"]

# === Разделение на train / val ===
n = int(0.8 * len(y))
train = {k: v[:n] for k, v in data.items()}
val   = {k: v[n:] for k, v in data.items()}

# === Модель ===
class FusionModel(nn.Module):
    def __init__(self, dim_img=512, dim_aud=768, dim_txt=768, hidden=512, num_classes=7):
        super().__init__()
        self.img_fc = nn.Linear(dim_img, hidden)
        self.aud_fc = nn.Linear(dim_aud, hidden)
        self.txt_fc = nn.Linear(dim_txt, hidden)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, aud, txt):
        img = torch.relu(self.img_fc(img))
        aud = torch.relu(self.aud_fc(aud))
        txt = torch.relu(self.txt_fc(txt))
        fused = torch.cat([img, aud, txt], dim=1)
        return self.fc_out(fused)

model = FusionModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# === Обучение ===
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(train["img"], train["aud"], train["txt"])
    loss = criterion(outputs, train["label"].long())
    loss.backward()
    optimizer.step()

    # валидация
    model.eval()
    with torch.no_grad():
        val_outputs = model(val["img"], val["aud"], val["txt"])
        val_loss = criterion(val_outputs, val["label"].long())
        preds = torch.argmax(val_outputs, dim=1)
        acc = (preds == val["label"]).float().mean().item()

    print(f"Эпоха {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc*100:.2f}%")

torch.save(model.state_dict(), "fusion_model.pth")
print("✅ Мультимодальная модель обучена и сохранена!")
