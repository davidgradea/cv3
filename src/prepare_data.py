
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASSES, IMAGES_PER_CLASS, RAW_DIR, SEED, SPLIT_RATIOS, SPLITS_DIR


def collect_images():
    rng = random.Random(SEED)
    rows = []
    for label, cls in enumerate(CLASSES):
        # Ignoram fisierele "._*" (metadate macOS)
        files = sorted(p for p in (RAW_DIR / cls).glob("*.jpg") if not p.name.startswith("._"))
        if len(files) < IMAGES_PER_CLASS:
            raise ValueError(f"Clasa {cls} are doar {len(files)} imagini")
        for p in rng.sample(files, IMAGES_PER_CLASS):
            rows.append({"path": p.relative_to(RAW_DIR).as_posix(), "class": cls, "label": label})
    return pd.DataFrame(rows)


def main():
    df = collect_images()
    train_r, val_r, test_r = SPLIT_RATIOS

    train_df, rest_df = train_test_split(
        df, train_size=train_r, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=test_r / (val_r + test_r), stratify=rest_df["label"], random_state=SEED
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.sort_values(["label", "path"]).to_csv(SPLITS_DIR / f"{name}.csv", index=False)

    summary = pd.DataFrame({
        "train": train_df["class"].value_counts(),
        "val": val_df["class"].value_counts(),
        "test": test_df["class"].value_counts(),
    }).loc[CLASSES]
    summary.loc["TOTAL"] = summary.sum()
    print(summary)


if __name__ == "__main__":
    main()
