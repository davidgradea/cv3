import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, IMG_SIZE, MEAN, NUM_WORKERS, RAW_DIR, SPLITS_DIR, STD


def get_transforms(train: bool):
    if train:
       
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class BloodCellDataset(Dataset):
    def __init__(self, split: str):
        self.df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        self.transform = get_transforms(train=(split == "train"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(RAW_DIR / row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def get_loader(split: str, batch_size: int = BATCH_SIZE):
    return DataLoader(
        BloodCellDataset(split),
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
