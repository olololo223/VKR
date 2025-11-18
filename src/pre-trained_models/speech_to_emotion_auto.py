import torch
import whisper
from transformers import pipeline

def transcribe_audio(audio_path: str, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    lang = result["language"]
    return text, lang

def analyze_emotion_text(text: str, lang: str):
    if lang.startswith("ru"):
        emotion_analyzer = pipeline("text-classification", model="./emotion_rubert_ruemocorpus", top_k=3)
    else:
        emotion_analyzer = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=3
        )

    emotions = emotion_analyzer(text)
    best = emotions[0][0]
    return best

def speech_to_emotion(audio_path: str, model_size="small"):
    text, lang = transcribe_audio(audio_path, model_size=model_size)
    emotion = analyze_emotion_text(text, lang)
    return emotion

if __name__ == "__main__":
    audio_file = "example.wav"
    result = speech_to_emotion(audio_file, model_size="tiny")

