"""
Скрипт для запуска GUI демо приложения распознавания эмоций
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.demo_gui import main

if __name__ == "__main__":
    print("=" * 50)
    print("Запуск GUI демо приложения")
    print("=" * 50)
    print("\nУбедитесь, что:")
    print("1. Модели обучены и находятся в logs/training/")
    print("2. Камера и микрофон доступны")
    print("\nЗапуск...")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nПриложение остановлено пользователем")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()

