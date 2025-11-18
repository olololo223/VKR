import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import json
import os

DATA_PATH = "datasets/ravdess"
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4
DEVICE = "cpu"

class RAVDESSDataset(torch.utils.data.Dataset):
    def __init__(self, folder, processor, emotion_map):
        with open('config/model_config.json', 'r', encoding='utf-8') as file:
            self.config = json.load(file)
        self.samples = []
        self.processor = processor
        self.emotion_map = self.config['ravdess']['emotion_map']

        wav_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        for path in wav_files:
            filename = os.path.basename(path)
            info = self.parse_filename(filename)

            if info is None:
                continue

            emotion_code = info["emotion"]

            if emotion_code not in emotion_map:
                continue

            class_id = emotion_map[emotion_code]

            self.samples.append((path, class_id))
        self.labels = sorted(list(set(l for _, l in self.samples)))
        self.label2id = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id = self.samples[idx]
        waveform, sr = torchaudio.load(path)

        waveform = waveform.mean(dim=0)

        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        return inputs.input_values.squeeze(0), class_id

    def parse_filename(self, filename):
        parts = filename.split('-')

        if len(parts) >= 7:
            return {
                'modality': parts[0],
                'vocal_channel': parts[1],
                'emotion': parts[2],
                'intensity': parts[3],
                'statement': parts[4],
                'repetition': parts[5],
                'actor': parts[6].split('.')[0]
            }
        return None


class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        for p in self.wav2vec.parameters():
            p.requires_grad = False
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
    torch.save(model.state_dict(), "wav2vec_emotion_light.pth")
