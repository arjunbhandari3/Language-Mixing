import argparse
import json
from pathlib import Path
from typing import Any

from utils.data_processing import read_input_paragraphs, tokenize, write_csv
from utils.dataset_split import split_rows_by_paragraph, write_split_csvs
from utils.llm_client import build_llm_prompt, call_lmstudio_chat, load_llm_config, parse_llm_json
from utils.pos import normalize_pos_tag
from utils.summaries import summarize, summarize_by_paragraph


def analyze_paragraph_with_llm(
    text: str,
    paragraph_id: int,
    llm_config: dict[str, str],
) -> list[dict[str, str]]:
    tokens = tokenize(text)
    if not tokens:
        return []

    prompt = build_llm_prompt(tokens, text)
    raw = call_lmstudio_chat(
        prompt=prompt,
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key=llm_config.get("api_key", "lm-studio"),
    )
    payload = parse_llm_json(raw)

    items = payload.get("items")
    if not isinstance(items, list):
        print(f"  Warning: paragraph {paragraph_id} skipped - LLM response missing 'items' list.")
        return []
    if len(items) != len(tokens):
        print(
            f"  Warning: paragraph {paragraph_id} skipped - "
            f"LLM returned {len(items)} items for {len(tokens)} tokens."
        )
        return []

    rows: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each LLM item must be an object.")

        language = str(item.get("language", "OTHER")).upper()
        borrowing = str(item.get("borrowing", "UNKNOWN")).upper()
        pos = normalize_pos_tag(str(item.get("pos", "X")))
        explanation = str(item.get("explanation", "No explanation provided."))
        token_value = str(item.get("token", tokens[index]))

        if language not in {"NE", "EN", "OTHER"}:
            language = "OTHER"
        if borrowing not in {"NATIVE", "BORROWED", "CODE-MIXED", "UNKNOWN"}:
            borrowing = "UNKNOWN"

        rows.append(
            {
                "Paragraph": str(paragraph_id),
                "Token": token_value,
                "Language": language,
                "Borrowing": borrowing,
                "POS": pos,
                "Explanation": explanation,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze extracted translation paragraphs into token-level code-mixing categories."
    )
    parser.add_argument(
        "input_file",
        help="Path to extracted translation text (.txt); paragraphs separated by blank lines",
    )
    parser.add_argument(
        "--out",
        default="src/output",
        help="Directory where analysis files will be written",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting (default: 42)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.parent
    config_path = project_root / "src" / "config" / "llm.json"
    llm_config = load_llm_config(config_path)

    paragraphs = read_input_paragraphs(input_path)
    total = len(paragraphs)
    print(f"Analyzing {total} paragraph(s) using model '{llm_config['model']}'...")

    rows: list[dict[str, str]] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        print(f"  [{index}/{total}] Processing paragraph {index}...")
        rows.extend(analyze_paragraph_with_llm(paragraph, index, llm_config=llm_config))
    print("Analysis complete.")

    summary = summarize(rows)
    paragraph_summary = summarize_by_paragraph(rows)
    split_rows = split_rows_by_paragraph(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    stem = input_path.stem
    table_path = out_dir / f"{stem}_analysis.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    paragraph_summary_path = out_dir / f"{stem}_paragraph_summary.json"
    split_paths = write_split_csvs(stem, out_dir, split_rows)

    write_csv(rows, table_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paragraph_summary_path.write_text(
        json.dumps(paragraph_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved token analysis table: {table_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved paragraph summary metrics: {paragraph_summary_path}")
    print(f"Saved training split: {split_paths['train']}")
    print(f"Saved validation split: {split_paths['validation']}")
    print(f"Saved test split: {split_paths['test']}")


if __name__ == "__main__":
    main()
