import torch.nn as nn
from torchvision import models

from config import CLASSES


def build_model(name: str, num_classes: int = len(CLASSES), pretrained: bool = True):
    """Model preantrenat pe ImageNet cu ultimul strat inlocuit (transfer learning).

    Modelul returneaza logits; Softmax se aplica la inferenta / evaluare
    (CrossEntropyLoss include deja log-softmax).
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model necunoscut: {name}")
    return model
