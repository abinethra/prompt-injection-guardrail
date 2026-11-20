from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Disable tokenizers fork warning on Linux when using DataLoader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "transformer"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
RANDOM_SEED = 42


class PromptInjectionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(filename: str) -> tuple[List[str], List[int]]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run scripts/prepare_data.py first.")
    df = pd.read_csv(path)
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def evaluate_val_loss(
    model: AutoModelForSequenceClassification,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(outputs.loss.item()) * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def freeze_early_layers(model: AutoModelForSequenceClassification, freeze_blocks: int = 3) -> None:
    if not hasattr(model, "distilbert"):
        return

    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False

    for idx, layer in enumerate(model.distilbert.transformer.layer):
        if idx < freeze_blocks:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

def main() -> None:
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {MODEL_NAME} on device: {device} | max_length={MAX_LENGTH} | epochs={EPOCHS} | batch_size={BATCH_SIZE}")

    train_texts, train_labels = load_split("train.csv")
    val_texts, val_labels = load_split("val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    freeze_early_layers(model, freeze_blocks=3)

    train_ds = PromptInjectionDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = PromptInjectionDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())

            if step % 10 == 0 or step == num_batches:
                pct = (step / num_batches) * 100.0
                avg_step_loss = running_loss / step
                print(
                    f"Epoch [{epoch}/{EPOCHS}] | Step [{step}/{num_batches}] ({pct:5.1f}%) | "
                    f"train_loss={avg_step_loss:.4f}",
                    flush=True,
                )

        val_loss, val_acc = evaluate_val_loss(model, val_loader, device)
        print(f"--> End of Epoch {epoch}/{EPOCHS}: val_loss={val_loss:.4f} | val_acc={val_acc:.4f}", flush=True)

    model.save_pretrained(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)
    print(f"Saved fine-tuned transformer model and tokenizer to: {MODELS_DIR}")


if __name__ == "__main__":
    main()