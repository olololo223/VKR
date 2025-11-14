import os
import time
from datetime import datetime

FILE_PATH = "C:/Users/User/Desktop/VKR/src/pre-trained_models/resnet_emotion_light.pth"  # путь к файлу
CHECK_INTERVAL = 60 * 60  # 1 час в секундах

print(f"🔍 Мониторинг файла: {FILE_PATH}")
print("Скрипт будет проверять наличие файла каждый час.\n")

while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(FILE_PATH):
        print(f"[{now}] ✅ Файл найден!")
    else:
        print(f"[{now}] ⏳ Файл пока отсутствует.")
    time.sleep(CHECK_INTERVAL)
