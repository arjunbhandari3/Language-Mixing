import csv
import re
from pathlib import Path

TOKEN_RE = re.compile(r"[\u0900-\u097Fa-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def parse_paragraphs_from_text(raw_text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", raw_text.strip())
    paragraphs = [chunk.strip() for chunk in chunks if chunk.strip()]
    return paragraphs


def read_input_paragraphs(input_path: Path) -> list[str]:
    raw_text = input_path.read_text(encoding="utf-8")
    paragraphs = parse_paragraphs_from_text(raw_text)
    if not paragraphs:
        raise ValueError("No non-empty paragraphs found in text input.")
    return paragraphs


def write_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Paragraph", "Token", "Language", "Borrowing", "POS", "Explanation"],
        )
        writer.writeheader()
        writer.writerows(rows)
