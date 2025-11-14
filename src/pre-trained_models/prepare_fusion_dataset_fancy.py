import os
import torch
import torchaudio
from torchvision import models, transforms
from PIL import Image
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    AutoTokenizer, AutoModel, pipeline
)
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
import whisper

# === Пути ===
DATASET_DIR = "datasets"
FER_DIR = os.path.join(DATASET_DIR, "fer2013plus", "train")
RAVDESS_DIR = os.path.join(DATASET_DIR, "ravdess")
TEXT_SAVE_DIR = os.path.join(DATASET_DIR, "texts")
SAVE_PATH = "fusion_dataset.pt"

# === Устройство ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Эмоции ===
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
emotion_to_id = {e: i for i, e in enumerate(EMOTIONS)}

# === Карта эмоций RAVDESS ===
emotion_map = {
    "01": "neutral",
    "02": "neutral",  # calm → neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# === Модели ===
print("🔧 Загрузка моделей...")

# ResNet (визуальная)
resnet = models.resnet18()
resnet.fc = torch.nn.Identity()
resnet.load_state_dict(
    torch.load("models/resnet_emotion_light.pth", map_location=DEVICE),
    strict=False
)
resnet.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Wav2Vec2 (аудио)
wav2proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2model = Wav2Vec2Model.from_pretrained("models/wav2vec2").to(DEVICE)
wav2model.eval()

# RuBERT (русский)
rubert_tok = AutoTokenizer.from_pretrained("models/rubert_emotion_model")
rubert = AutoModel.from_pretrained("models/rubert_emotion_model").to(DEVICE)
rubert.eval()

# English GoEmotions (английский)
eng_model = "SamLowe/roberta-base-go_emotions"
eng_analyzer = pipeline("feature-extraction", model=eng_model, tokenizer=eng_model, device=0 if DEVICE=="cuda" else -1)

# Whisper (Speech-to-Text)
whisper_model = whisper.load_model("base", device=DEVICE)

# === Функции ===

def get_img_emb(path):
    try:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = resnet(x).cpu()
        return emb.squeeze(0)
    except Exception:
        return torch.zeros(512)

def get_audio_emb(path):
    try:
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        inputs = wav2proc(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = wav2model(**inputs.to(DEVICE)).last_hidden_state.mean(dim=1).cpu()
        return emb.squeeze(0)
    except Exception:
        return torch.zeros(768)

def transcribe_audio(audio_path, txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        lang = "ru" if any("а" <= c <= "я" for c in text.lower()) else "en"
        return text, lang

    result = whisper_model.transcribe(audio_path)
    text = result["text"].strip()
    lang = result["language"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, lang

def get_text_emb(audio_path, emotion, base):
    txt_path = os.path.join(TEXT_SAVE_DIR, emotion, f"{base}.txt")
    text, lang = transcribe_audio(audio_path, txt_path)

    if not text:
        return torch.zeros(768)

    if lang.startswith("ru"):
        inputs = rubert_tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            emb = rubert(**inputs).last_hidden_state.mean(dim=1).cpu()
        return emb.squeeze(0)
    else:
        with torch.no_grad():
            outputs = eng_analyzer(text)
        emb = torch.tensor(outputs[0]).mean(dim=0)
        return emb

# === Основной цикл ===
data = {"img": [], "aud": [], "txt": [], "label": []}

print("\n🚀 Генерация Fusion-эмбеддингов...")

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Обработка RAVDESS...", total=24)

    for actor in os.listdir(RAVDESS_DIR):
        actor_dir = os.path.join(RAVDESS_DIR, actor)
        if not os.path.isdir(actor_dir):
            continue

        for fname in os.listdir(actor_dir):
            if not fname.endswith(".wav"):
                continue

            parts = fname.split("-")
            if len(parts) < 3:
                continue

            emotion_id = parts[2]
            emotion = emotion_map.get(emotion_id)
            if emotion not in EMOTIONS:
                continue

            base = os.path.splitext(fname)[0]
            aud_path = os.path.join(actor_dir, fname)

            # для fer2013+ используем просто ту же эмоцию
            img_dir = os.path.join(FER_DIR, emotion)
            if not os.path.exists(img_dir):
                continue
            img_files = os.listdir(img_dir)
            if not img_files:
                continue
            img_path = os.path.join(img_dir, img_files[0])  # берём случайное изображение

            img_emb = get_img_emb(img_path)
            aud_emb = get_audio_emb(aud_path)
            txt_emb = get_text_emb(aud_path, emotion, base)

            data["img"].append(img_emb)
            data["aud"].append(aud_emb)
            data["txt"].append(txt_emb)
            data["label"].append(emotion_to_id[emotion])

        progress.advance(task)
print(data)
# === Сохраняем всё ===
# было:
# for k in data.keys():
#     data[k] = torch.stack(data[k])

# исправить на:
for k in data.keys():
    if k == "label":
        data[k] = torch.tensor(data[k])  # список чисел → один тензор
    else:
        data[k] = torch.stack(data[k])   # список тензоров → батч


torch.save(data, SAVE_PATH)
print(f"\n✅ Fusion датасет сохранён: {SAVE_PATH}")
print(f"📦 Размер: {len(data['label'])} примеров")
