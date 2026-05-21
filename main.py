import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze extracted translation paragraphs from a text file."
    )
    parser.add_argument(
        "input_file",
        help="Path to paragraph text file (.txt). Separate paragraphs with blank lines.",
    )
    parser.add_argument("--out", default="src/output")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input text file not found: {input_path}")

    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))
    from analyze_translations import (
        load_llm_config,
        read_input_paragraphs,
        analyze_paragraph_with_llm,
        summarize,
        summarize_by_paragraph,
        split_rows_by_paragraph,
        write_split_csvs,
        write_csv,
    )
    import json

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "src" / "config" / "llm.json"
    llm_config = load_llm_config(config_path)

    paragraphs = read_input_paragraphs(input_path)
    total = len(paragraphs)
    print(f"Analyzing {total} paragraph(s) using model '{llm_config['model']}'...")
    rows = []
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
        json.dumps(paragraph_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved token analysis table: {table_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved paragraph summary metrics: {paragraph_summary_path}")
    print(f"Saved training split: {split_paths['train']}")
    print(f"Saved validation split: {split_paths['validation']}")
    print(f"Saved test split: {split_paths['test']}")


if __name__ == "__main__":
    main()

