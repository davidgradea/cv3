from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "PBC_dataset_normal_DIB"
SPLITS_DIR = ROOT / "data" / "splits"
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


CLASSES = ["neutrophil", "eosinophil", "lymphocyte", "monocyte", "ig", "erythroblast"]

IMAGES_PER_CLASS = 750
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42

IMG_SIZE = 224

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
MODEL_NAME = "resnet50"  #  "efficientnet_b0"
