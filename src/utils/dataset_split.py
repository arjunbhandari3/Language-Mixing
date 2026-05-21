import math
import random
from pathlib import Path

from utils.data_processing import write_csv


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, value in (("train", train_ratio), ("validation", val_ratio), ("test", test_ratio)):
        if value < 0:
            raise ValueError(f"{name.title()} ratio must be non-negative.")

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received train={train_ratio}, validation={val_ratio}, test={test_ratio}."
        )


def split_rows_by_paragraph(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    paragraph_ids = sorted({row["Paragraph"] for row in rows}, key=lambda x: int(x))
    if not paragraph_ids:
        return {"train": [], "validation": [], "test": []}

    randomizer = random.Random(seed)
    shuffled = paragraph_ids[:]
    randomizer.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count : train_count + val_count + test_count])

    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        paragraph = row["Paragraph"]
        if paragraph in train_ids:
            splits["train"].append(row)
        elif paragraph in val_ids:
            splits["validation"].append(row)
        else:
            splits["test"].append(row)

    return splits


def write_split_csvs(stem: str, out_dir: Path, split_rows: dict[str, list[dict[str, str]]]) -> dict[str, Path]:
    paths = {
        "train": out_dir / f"{stem}_train.csv",
        "validation": out_dir / f"{stem}_validation.csv",
        "test": out_dir / f"{stem}_test.csv",
    }
    for split_name, csv_path in paths.items():
        write_csv(split_rows[split_name], csv_path)
    return paths
