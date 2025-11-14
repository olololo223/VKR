import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "training_log.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    print("Найденные столбцы:", df.columns.tolist())

    # проверяем доступные поля
    has_acc = "accuracy" in df.columns and "val_accuracy" in df.columns
    has_loss = "loss" in df.columns and "val_loss" in df.columns

    if not has_acc or not has_loss:
        print("В CSV нет подходящих столбцов (loss/val_loss или accuracy/val_accuracy).")
        return

    # ==============================
    # ГРАФИК LOSS
    # ==============================
    plt.figure(figsize=(10, 5))
    plt.plot(df["loss"], label="Train Loss")
    #plt.plot(df["val_loss"], label="Val Loss")
    plt.plot(df["accuracy"], label="Train Accuracy")
    #plt.plot(df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.title("Accuracy and loss curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_loss_plot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
