"""
fusion_demo.py
Демо: камера + микрофон -> мультимодальная модель -> эмоция

Подстройте пути к моделям в переменных ниже при необходимости.
"""
import os
import time
import tempfile
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
CAMERA_INDEX = 0
MODELS_DIR = "models"

RESNET_PTH = os.path.join(MODELS_DIR, "resnet_emotion_light.pth")
WAV2VEC_DIR = os.path.join(MODELS_DIR, "wav2vec2")
RUBERT_DIR = os.path.join(MODELS_DIR, "rubert_emotion_model")
FUSION_PTH = os.path.join(MODELS_DIR, "fusion_model.pth")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def check_file(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} не найден: {path}")

check_file(RESNET_PTH, "ResNet weights")
check_file(WAV2VEC_DIR, "Wav2Vec2 folder")
check_file(RUBERT_DIR, "RuBERT folder")
check_file(FUSION_PTH, "Fusion model weights")

print(f"Device: {DEVICE}")
from torchvision import models

resnet = models.resnet18()
resnet.fc = nn.Identity()
state = torch.load(RESNET_PTH, map_location=DEVICE)
resnet.load_state_dict(state, strict=False)
resnet = resnet.to(DEVICE).eval()

img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

wav2proc = Wav2Vec2Processor.from_pretrained(WAV2VEC_DIR)
wav2model = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR).to(DEVICE).eval()

tokenizer = AutoTokenizer.from_pretrained(RUBERT_DIR)
rubert = AutoModel.from_pretrained(RUBERT_DIR).to(DEVICE).eval()

try:
    whisper_model = whisper.load_model("tiny", device=DEVICE)
except Exception as e:
    print("⚠️ Ошибка при загрузке whisper:", e)
    whisper_model = None

class FusionModel(nn.Module):
    def __init__(self, dim_img=512, dim_aud=768, dim_txt=768, hidden=512, num_classes=7):
        super().__init__()
        self.img_fc = nn.Linear(dim_img, hidden)
        self.aud_fc = nn.Linear(dim_aud, hidden)
        self.txt_fc = nn.Linear(dim_txt, hidden)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden*3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, img, aud, txt):
        img = torch.relu(self.img_fc(img))
        aud = torch.relu(self.aud_fc(aud))
        txt = torch.relu(self.txt_fc(txt))
        cat = torch.cat([img, aud, txt], dim=1)
        return self.head(cat)

fusion = FusionModel(dim_img=512, dim_aud=768, dim_txt=768, num_classes=len(EMOTIONS))
fusion.load_state_dict(torch.load(FUSION_PTH, map_location=DEVICE), strict=False)
fusion = fusion.to(DEVICE).eval()

def capture_image_from_cam():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Не удалось открыть камеру. Проверь индекс камеры.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось получить кадр с камеры.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    x = img_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(x)
    return emb

def record_audio(seconds=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"⏺️ Запись {seconds} секунд...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio, sr

def audio_to_wav_bytes(audio_np, sr):
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, sr)
    tmp.close()
    return tmp.name

def get_audio_embedding_from_np(audio_np, sr):
    inputs = wav2proc(audio_np, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        last = wav2model(**inputs.to(DEVICE)).last_hidden_state.mean(dim=1)
    return last

def transcribe_with_whisper(path):
    if whisper_model is None:
        return ""
    try:
        res = whisper_model.transcribe(path, language=None)
        return res.get("text","").strip()
    except Exception as e:
        print("⚠️ Whisper error:", e)
        return ""

def get_text_embedding_from_text(text):
    if not text:
        return torch.zeros((1,768), device=DEVICE)
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = rubert(**inputs).last_hidden_state.mean(dim=1)
    return emb

def pretty_probs(probs):
    lines = []
    for e, p in zip(EMOTIONS, probs):
        lines.append(f"{e:9s} → {p*100:5.1f}%")
    return "\n".join(lines)

def run_once():
    print("\n📸 Снимем кадр с камеры...")
    img_emb = capture_image_from_cam()

    audio_np, sr = record_audio()
    wav_path = audio_to_wav_bytes(audio_np, sr)

    print("🔊 Извлекаем аудио-эмбеддинг (Wav2Vec2)...")
    try:
        aud_emb = get_audio_embedding_from_np(audio_np, sr)
    except Exception as e:
        print("⚠️ Ошибка при извлечении аудио-эмбеддинга:", e)
        aud_emb = torch.zeros((1,768), device=DEVICE)

    print("📝 Транскрибируем речь (Whisper)...")
    text = transcribe_with_whisper(wav_path)
    if text:
        print("🗣  Распознано:", text)
    else:
        print("ℹ️  Текст не распознан (пусто). Можно ввести вручную.")
        text = input("Введи текст (или enter, чтобы пропустить): ").strip()

    txt_emb = get_text_embedding_from_text(text)

    with torch.no_grad():
        logits = fusion(img_emb.to(DEVICE), aud_emb.to(DEVICE), txt_emb.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = EMOTIONS[pred_idx]

    print("\n" + "="*32)
    print(f"🎭 Предсказанная эмоция: {pred_label.upper()}  (индекс {pred_idx})")
    print(pretty_probs(probs))
    print("="*32 + "\n")

    try:
        os.remove(wav_path)
    except Exception:
        pass

if __name__ == "__main__":
    print("=== Fusion demo ===")
    print("Примечание: при первом запуске Whisper и модели HuggingFace могут скачиваться (нужно соединение).")
    print("Нажми Enter, чтобы начать одну итерацию, Ctrl+C чтобы выйти.")
    try:
        while True:
            input("→ Нажми Enter для записи (камера + микрофон)...")
            run_once()
    except KeyboardInterrupt:
        print("\nВыход. Удачи!")
