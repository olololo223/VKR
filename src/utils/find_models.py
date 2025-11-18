import os
import glob

def find_latest_models():
    """Поиск последних обученных моделей"""

    visual_pattern = "logs/training/visual_*"
    visual_dirs = glob.glob(visual_pattern)
    visual_dirs.sort(key=os.path.getmtime, reverse=True)

    visual_model = None
    if visual_dirs:
        latest_visual_dir = visual_dirs[0]
        visual_model = os.path.join(latest_visual_dir, "final_model.h5")
        if os.path.exists(visual_model):
            print(f"Найдена визуальная модель: {visual_model}")
        else:
            visual_model = os.path.join(latest_visual_dir, "best_model.h5")
            if os.path.exists(visual_model):
                print(f"Найдена визуальная модель: {visual_model}")
            else:
                print("Визуальная модель не найдена")
                visual_model = None
    else:
        print("Директории с визуальными моделями не найдены")

    audio_pattern = "logs/training/audio_*"
    audio_dirs = glob.glob(audio_pattern)
    audio_dirs.sort(key=os.path.getmtime, reverse=True)

    audio_model = None
    if audio_dirs:
        latest_audio_dir = audio_dirs[0]
        audio_model = os.path.join(latest_audio_dir, "final_audio_model.h5")
        if os.path.exists(audio_model):
            print(f"Найдена аудио модель: {audio_model}")
        else:
            audio_model = os.path.join(latest_audio_dir, "best_audio_model.h5")
            if os.path.exists(audio_model):
                print(f"Найдена аудио модель: {audio_model}")
            else:
                print("Аудио модель не найдена")
                audio_model = None
    else:
        print("Директории с аудио моделями не найдены")

    return visual_model, audio_model

if __name__ == "__main__":
    visual, audio = find_latest_models()
    print("\nКоманда для запуска мультимодального обучения:")
    if visual and audio:
        print(f'python main.py --mode train_multimodal --visual_model_path "{visual}" --audio_model_path "{audio}"')
    else:
        print("Не все модели найдены!")
