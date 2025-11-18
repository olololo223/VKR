import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

dataset = load_dataset("seara/ru_go_emotions", "simplified")

def simplify_labels(example):
    mapping = {
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "happy",
        "sadness": "sad",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    labels = example["labels"]
    if isinstance(labels, list):
        labels = labels[0] if labels else "neutral"
    example["labels"] = mapping.get(labels, "neutral")
    return example

dataset = dataset.map(simplify_labels)


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

label_list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

dataset = dataset.map(lambda e: {"label": label2id[e["labels"]]}, batched=False)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./emotion_rubert_ru_go_emotions",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs_rubert",
    save_steps=500,
    logging_steps=100,
    report_to=[] 
)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./rubert_emotion_model")
tokenizer.save_pretrained("./rubert_emotion_model")
