# wav2vec_model_light.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import os

DATA_PATH = "datasets/ravdess"
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RAVDESSDataset(torch.utils.data.Dataset):
    def __init__(self, folder, processor):
        self.samples = []
        self.processor = processor
        for label in os.listdir(folder):
            for f in os.listdir(os.path.join(folder, label)):
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(folder, label, f), label))
        self.labels = sorted(list(set(l for _, l in self.samples)))
        self.label2id = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)  # mono
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), self.label2id[label]

class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for p in self.wav2vec.parameters():
            p.requires_grad = False  # заморозим encoder
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.wav2vec(x).last_hidden_state.mean(dim=1)
        return self.fc(features)

if __name__ == "__main__":
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    ds = RAVDESSDataset(DATA_PATH, processor)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = Wav2VecClassifier(num_classes=len(ds.labels)).to(DEVICE)
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"🎤 Обучение Wav2Vec2 на {DEVICE} ({len(ds)} аудио)")
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

    torch.save(model.state_dict(), "wav2vec_emotion_light.pth")
    print("✅ Модель сохранена: wav2vec_emotion_light.pth")
