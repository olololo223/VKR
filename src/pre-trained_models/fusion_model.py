# fusion_model_light.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FusionEmotionNet(nn.Module):
    def __init__(self, img_dim=512, aud_dim=768, hidden=256, num_classes=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + aud_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, img_emb, aud_emb):
        fused = torch.cat([img_emb, aud_emb], dim=1)
        return self.fc(fused)

if __name__ == "__main__":
    model = FusionEmotionNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Заглушка (пример эмбеддингов)
    img_emb = torch.randn(200, 512).to(DEVICE)
    aud_emb = torch.randn(200, 768).to(DEVICE)
    labels = torch.randint(0, 7, (200,)).to(DEVICE)

    dl = torch.utils.data.DataLoader(list(zip(img_emb, aud_emb, labels)), batch_size=BATCH_SIZE, shuffle=True)

    print(f"🔗 Обучение FusionNet на {DEVICE}")
    for epoch in range(EPOCHS):
        running_loss = 0
        for img, aud, y in tqdm(dl, desc=f"Эпоха {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            out = model(img, aud)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Средний loss: {running_loss/len(dl):.4f}")

    torch.save(model.state_dict(), "fusion_emotion_light.pth")
    print("✅ Модель сохранена: fusion_emotion_light.pth")
