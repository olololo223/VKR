"""
fusion_gui.py
Tkinter GUI real-time fusion demo (camera + microphone -> emotion)

Сохрани рядом с папкой models/, содержащей:
 - resnet_emotion_light.pth
 - wav2vec2/  (preprocessor_config.json, config.json, pytorch_model.bin и т.д.)
 - rubert_emotion_model/
 - fusion_model.pth

Запуск:
    python fusion_gui.py
"""
import os
import queue
import tempfile
import threading
import time
import traceback

import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import ttk, messagebox

from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import whisper
import soundfile as sf

# ----------------------------
# Настройки
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "models"
RESNET_PTH = os.path.join(MODELS_DIR, "resnet_emotion_light.pth")
WAV2VEC_DIR = os.path.join(MODELS_DIR, "wav2vec2")
RUBERT_DIR = os.path.join(MODELS_DIR, "rubert_emotion_model")
FUSION_PTH = os.path.join(MODELS_DIR, "fusion_model.pth")
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0   # окно для анализа (длина аудио фрагмента)
STEP_SECONDS = 1.0     # через сколько секунд берем новое предсказание (накладывающиеся окна)
CAMERA_INDEX = 0

# ----------------------------
# Простая проверка файлов
# ----------------------------
def file_ok(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} не найден: {path}")

try:
    file_ok(RESNET_PTH, "ResNet weights")
    file_ok(WAV2VEC_DIR, "Wav2Vec2 folder")
    file_ok(RUBERT_DIR, "RuBERT folder")
    file_ok(FUSION_PTH, "Fusion model weights")
except Exception as e:
    messagebox.showerror("Missing files", str(e))
    raise

# ----------------------------
# Модель/процессоры
# ----------------------------
print("Загрузка моделей на", DEVICE, " — это может занять время...")
from torchvision import models
# ResNet backbone
resnet = models.resnet18()
resnet.fc = nn.Identity()
state = torch.load(RESNET_PTH, map_location=DEVICE)
resnet.load_state_dict(state, strict=False)
resnet.to(DEVICE).eval()
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Wav2Vec2 processor + model
wav2proc = Wav2Vec2Processor.from_pretrained(WAV2VEC_DIR)
wav2model = Wav2Vec2Model.from_pretrained(WAV2VEC_DIR).to(DEVICE).eval()

# RuBERT tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(RUBERT_DIR)
rubert = AutoModel.from_pretrained(RUBERT_DIR).to(DEVICE).eval()

# Whisper (tiny для скорости)
try:
    whisper_model = whisper.load_model("tiny", device=DEVICE)
except Exception as e:
    print("Whisper загрузить не получилось:", e)
    whisper_model = None

# Fusion model (структура должна совпадать с обученной)
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
fusion.to(DEVICE).eval()

print("Модели загружены.")

# ----------------------------
# Аудио поток (звук) — используем callback для InputStream
# ----------------------------
class AudioRingBuffer:
    def __init__(self, maxlen):
        self.maxlen = int(maxlen)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.pos = 0
        self.lock = threading.Lock()

    def push(self, data):
        with self.lock:
            n = data.shape[0]
            if n >= self.maxlen:
                # keep last maxlen
                self.buf[:] = data[-self.maxlen:]
                self.pos = 0
            else:
                end = (self.pos + n) % self.maxlen
                if self.pos + n <= self.maxlen:
                    self.buf[self.pos:self.pos+n] = data
                else:
                    part = self.maxlen - self.pos
                    self.buf[self.pos:] = data[:part]
                    self.buf[:n-part] = data[part:]
                self.pos = end

    def get_last(self, length):
        length = int(length)
        with self.lock:
            if length >= self.maxlen:
                return self.buf.copy()
            # compute start index of last `length` samples
            # buffer stores continuous last maxlen with wrap at self.pos
            start = (self.pos - length) % self.maxlen
            if start + length <= self.maxlen:
                return self.buf[start:start+length].copy()
            else:
                part = self.maxlen - start
                return np.concatenate([self.buf[start:], self.buf[:length-part]])

# Инициализация кольцевого буфера
MAX_SECONDS = 10
ring = AudioRingBuffer(int(SAMPLE_RATE * MAX_SECONDS))

def sd_callback(indata, frames, time_info, status):
    if status:
        print("sounddevice status:", status)
    samples = indata[:,0].astype(np.float32)
    ring.push(samples)

# ----------------------------
# Функции извлечения эмбеддингов
# ----------------------------
def img_to_emb(frame_bgr):
    # frame_bgr — OpenCV BGR image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    x = img_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(x)
    return emb  # [1,512]

def audio_np_to_wavfile(audio_np, sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, sr)
    tmp.close()
    return tmp.name

def audio_to_audemb(audio_np, sr):
    try:
        inputs = wav2proc(audio_np, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            last = wav2model(**inputs.to(DEVICE)).last_hidden_state.mean(dim=1)
        return last
    except Exception as e:
        print("Ошибка Wav2Vec2:", e)
        return torch.zeros((1,768), device=DEVICE)

def text_to_txtemb(text):
    if not text:
        return torch.zeros((1,768), device=DEVICE)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        emb = rubert(**inputs).last_hidden_state.mean(dim=1)
    return emb

def transcribe_whisper(path):
    if whisper_model is None:
        return ""
    try:
        res = whisper_model.transcribe(path, language=None)
        return res.get("text","").strip()
    except Exception as e:
        print("Whisper error:", e)
        return ""

# ----------------------------
# GUI
# ----------------------------
class FusionApp:
    def __init__(self, root):
        self.root = root
        root.title("Fusion Real-time Demo")
        self.input_devices = {}
        self.running = False

        # Camera frame
        self.cam_label = ttk.Label(root, text="Камера:")
        self.cam_label.grid(row=0, column=0, sticky="w")
        self.canvas = tk.Label(root)
        self.canvas.grid(row=1, column=0, rowspan=6, padx=5, pady=5)

        # Microphone selection
        self.mic_label = ttk.Label(root, text="Микрофон:")
        self.mic_label.grid(row=0, column=1, sticky="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(root, textvariable=self.device_var, width=40)
        self.device_combo.grid(row=1, column=1, padx=5, pady=2)

        # Controls
        self.btn_frame = ttk.Frame(root)
        self.btn_frame.grid(row=2, column=1, sticky="n")
        self.start_btn = ttk.Button(self.btn_frame, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        self.stop_btn = ttk.Button(self.btn_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)

        # Emotion display
        self.emotion_var = tk.StringVar(value="—")
        self.emotion_label = ttk.Label(root, textvariable=self.emotion_var, font=("Helvetica", 18))
        self.emotion_label.grid(row=3, column=1, sticky="w", padx=5, pady=10)

        # Probabilities text
        self.probs_text = tk.Text(root, width=30, height=10, state="disabled")
        self.probs_text.grid(row=4, column=1, padx=5, pady=5)

        # Status
        self.status_var = tk.StringVar(value="Готов")
        self.status_label = ttk.Label(root, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=2, sticky="we")

        # camera capture
        self.cap = None
        self.cam_thread = None
        self.audio_stream = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self._load_devices()

    def _load_devices(self):
        # список устройств
        devices = sd.query_devices()

        input_names = []
        default_in = None

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = f"{i}: {dev['name']}"
                input_names.append(name)
                if i == sd.default.device[0]:  # текущее устройство по умолчанию
                    default_in = name

        # обновляем выпадающий список правильного combobox
        self.device_combo["values"] = input_names

        # устанавливаем правильную переменную device_var
        if default_in:
            self.device_var.set(default_in)
        elif input_names:
            self.device_var.set(input_names[0])




    def start(self):
        # start camera preview and audio stream + worker
        try:
            sel = self.device_combo.get()
            if not sel:
                messagebox.showerror("Error", "Выберите микрофон.")
                return
            dev_idx = int(sel.split(":")[0])
        except Exception:
            messagebox.showerror("Error", "Неверный выбор микрофона.")
            return

        # open camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Не удалось открыть камеру.")
            return

        # start audio stream
        try:
            self.audio_stream = sd.InputStream(device=dev_idx, channels=1, samplerate=SAMPLE_RATE, callback=sd_callback)
            self.audio_stream.start()
        except Exception as e:
            messagebox.showerror("Error", f"Не удалось запустить аудио вход: {e}")
            if self.cap:
                self.cap.release()
            return

        # threads
        self.stop_event.clear()
        self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self.cam_thread.start()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        self.start_btn['state'] = 'disabled'
        self.stop_btn['state'] = 'normal'
        self.status_var.set("Запущено (реальное время)")

    def stop(self):
        self.stop_event.set()
        time.sleep(0.2)
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        self.start_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'
        self.status_var.set("Остановлено")

    def _cam_loop(self):
        try:
            while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # resize for display
                disp = cv2.resize(frame, (320, 240))
                img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im_pil)
                # update Tk label
                self.canvas.imgtk = imgtk
                self.canvas.configure(image=imgtk)
                time.sleep(0.03)
        except Exception as e:
            print("cam loop error:", e)
            traceback.print_exc()

    def _update_ui_result(self, emotion, probs):
        self.emotion_var.set(emotion.upper())
        self.probs_text.configure(state="normal")
        self.probs_text.delete("1.0", tk.END)
        for e, p in zip(EMOTIONS, probs):
            self.probs_text.insert(tk.END, f"{e:10s} → {p*100:5.1f}%\n")
        self.probs_text.configure(state="disabled")

    def _worker_loop(self):
        # основной цикл: каждые STEP_SECONDS берем последний WINDOW_SECONDS аудио и прогоняем
        try:
            while not self.stop_event.is_set():
                # get last WINDOW_SECONDS from ring
                samples_needed = int(WINDOW_SECONDS * SAMPLE_RATE)
                audio_np = ring.get_last(samples_needed)
                if audio_np is None or len(audio_np) < int(0.5*SAMPLE_RATE):
                    # ещё не накопилось достаточно данных
                    time.sleep(0.2)
                    continue

                # 1) capture current webcam frame for image embedding
                frame = None
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                if frame is None:
                    # create blank
                    frame = np.zeros((240,320,3), dtype=np.uint8)

                # 2) get embeddings (image, audio, text)
                try:
                    img_emb = img_to_emb(frame)  # [1,512]
                except Exception as e:
                    print("img embedding error:", e)
                    img_emb = torch.zeros((1,512), device=DEVICE)

                # audio -> save temporary wav for whisper, and process via wav2vec
                try:
                    # normalize audio to -1..1 if needed
                    if audio_np.dtype != np.float32:
                        audio_np = audio_np.astype(np.float32)
                    # wav2vec embedding
                    aud_emb = audio_to_audemb(audio_np, SAMPLE_RATE)  # [1,768]
                except Exception as e:
                    print("audio embedding error:", e)
                    aud_emb = torch.zeros((1,768), device=DEVICE)

                # whisper transcription (non-blocking-ish — but could be heavy)
                txt = ""
                try:
                    wavfile = audio_np_to_wavfile(audio_np, SAMPLE_RATE)
                    if whisper_model is not None:
                        txt = transcribe_whisper(wavfile)
                    # cleanup
                    try:
                        os.remove(wavfile)
                    except Exception:
                        pass
                except Exception as e:
                    print("whisper pipeline error:", e)
                    txt = ""

                # text -> rubert embedding
                try:
                    txt_emb = text_to_txtemb(txt)
                except Exception as e:
                    print("text embedding error:", e)
                    txt_emb = torch.zeros((1,768), device=DEVICE)

                # fusion forward
                try:
                    with torch.no_grad():
                        logits = fusion(img_emb.to(DEVICE), aud_emb.to(DEVICE), txt_emb.to(DEVICE))
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        pred_label = EMOTIONS[pred_idx]
                except Exception as e:
                    print("fusion forward error:", e)
                    probs = np.zeros(len(EMOTIONS))
                    pred_label = "unknown"

                # update UI in main thread
                self.root.after(0, self._update_ui_result, pred_label, probs)
                time.sleep(STEP_SECONDS)
        except Exception as e:
            print("worker_loop exception:", e)
            traceback.print_exc()

# ----------------------------
# Запуск приложения
# ----------------------------
def main():
    root = tk.Tk()
    app = FusionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
