import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CHECKPOINTS_DIR, EPOCHS, LR, MODEL_NAME, OUTPUTS_DIR, SEED, WEIGHT_DECAY
from dataset import get_loader
from model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += logits.argmax(1).tolist()
            targets += y.tolist()

    n = len(targets)
    acc = sum(p == t for p, t in zip(preds, targets)) / n
    f1 = f1_score(targets, preds, average="macro")
    return total_loss / n, acc, f1


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="Epoca")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set(title="Top-1 Accuracy", xlabel="Epoca")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = get_loader("train"), get_loader("val")
    model = build_model(MODEL_NAME).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    history = {k: [] for k in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]}
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for prefix, (loss, acc, f1) in [("train", tr), ("val", va)]:
            history[f"{prefix}_loss"].append(loss)
            history[f"{prefix}_acc"].append(acc)
            history[f"{prefix}_f1"].append(f1)

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train loss {tr[0]:.4f} acc {tr[1]:.4f} f1 {tr[2]:.4f} | "
              f"val loss {va[0]:.4f} acc {va[1]:.4f} f1 {va[2]:.4f}")

        if va[2] > best_f1:
            best_f1 = va[2]
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{MODEL_NAME}_best.pt")
            print(f"  -> model salvat (val F1 = {best_f1:.4f})")

    with open(OUTPUTS_DIR / f"{MODEL_NAME}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, OUTPUTS_DIR / f"{MODEL_NAME}_curves.png")


if __name__ == "__main__":
    main()
