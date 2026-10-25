"""Evalueaza cel mai bun checkpoint pe setul de test."""
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import CHECKPOINTS_DIR, CLASSES, MODEL_NAME, OUTPUTS_DIR
from dataset import get_loader
from model import build_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        preds += probs.argmax(1).tolist()
        targets += y.tolist()
    return targets, preds


def plot_confusion(cm, path, normalize=False):
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predictie")
    plt.ylabel("Adevar")
    plt.title("Matrice de confuzie (test)" + (" - normalizata" if normalize else ""))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(MODEL_NAME, pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / f"{MODEL_NAME}_best.pt", map_location=device))

    y_true, y_pred = predict(model, get_loader("test"), device)

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    summary = (f"Top-1 Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
               f"Macro F1:       {f1_score(y_true, y_pred, average='macro'):.4f}\n\n{report}")
    print(summary)
    (OUTPUTS_DIR / f"{MODEL_NAME}_test_report.txt").write_text(summary)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, OUTPUTS_DIR / f"{MODEL_NAME}_confusion.png")
    plot_confusion(cm, OUTPUTS_DIR / f"{MODEL_NAME}_confusion_norm.png", normalize=True)


if __name__ == "__main__":
    main()
