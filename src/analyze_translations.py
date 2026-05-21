import argparse
import csv
import json
import math
import random
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[\u0900-\u097Fa-zA-Z']+")

def load_llm_config(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"LLM config file not found: {config_path}. "
            "Create src/config/llm.json with model, base_url, and api_key."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LLM config JSON must be an object.")

    model = str(payload.get("model", "")).strip()
    base_url = str(payload.get("base_url", "http://127.0.0.1:1234/v1")).strip()
    api_key = str(payload.get("api_key", "lm-studio")).strip()

    if not model:
        raise ValueError("LLM config must include a non-empty 'model' key.")

    return {"model": model, "base_url": base_url, "api_key": api_key}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_llm_prompt(tokens: list[str], paragraph_text: str) -> str:
    token_json = json.dumps(tokens, ensure_ascii=False)
    return (
        "You are a linguistics annotator for Nepali-English code-mixing. "
        "Classify each token in order and return STRICT JSON only.\n\n"
        "Allowed labels:\n"
        "- Language: NE | EN | OTHER\n"
        "- Borrowing: NATIVE | BORROWED | CODE-MIXED | UNKNOWN\n\n"
        "Input paragraph:\n"
        f"{paragraph_text}\n\n"
        "Tokens (ordered list):\n"
        f"{token_json}\n\n"
        "Return this exact schema and preserve token order/length:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "token": "<string>",\n'
        '      "language": "NE|EN|OTHER",\n'
        '      "borrowing": "NATIVE|BORROWED|CODE-MIXED|UNKNOWN",\n'
        '      "explanation": "<short reason>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def parse_llm_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM output was not valid JSON.")
        return json.loads(text[start : end + 1])


def call_lmstudio_chat(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: int = 120,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    def send_chat(request_payload: dict[str, Any]) -> str:
        data = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key or 'lm-studio'}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.read().decode("utf-8")

    try:
        body = send_chat(payload)
    except urllib.error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "response_format" in response_text:
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            try:
                body = send_chat(fallback_payload)
            except urllib.error.HTTPError as fallback_exc:
                fallback_text = fallback_exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    "LM Studio rejected the chat request "
                    f"(HTTP {fallback_exc.code}). Response: {fallback_text}"
                ) from fallback_exc
        else:
            raise RuntimeError(
                "LM Studio rejected the chat request "
                f"(HTTP {exc.code}). Response: {response_text}"
            ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to connect to LM Studio at {endpoint}. Ensure local server is running and URL is correct."
        ) from exc

    envelope = json.loads(body)
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Unexpected LM Studio response format: missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Unexpected LM Studio response format: missing message content.")
    return content


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
        print(f"  Warning: paragraph {paragraph_id} skipped — LLM response missing 'items' list.")
        return []
    if len(items) != len(tokens):
        print(
            f"  Warning: paragraph {paragraph_id} skipped — "
            f"LLM returned {len(items)} items for {len(tokens)} tokens."
        )
        return []

    rows: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each LLM item must be an object.")
        language = str(item.get("language", "OTHER")).upper()
        borrowing = str(item.get("borrowing", "UNKNOWN")).upper()
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
                "Explanation": explanation,
            }
        )

    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    language_counts = Counter(r["Language"] for r in rows)
    borrowing_counts = Counter(r["Borrowing"] for r in rows)

    return {
        "total_tokens": len(rows),
        "language_counts": dict(language_counts),
        "borrowing_counts": dict(borrowing_counts),
    }


def summarize_by_paragraph(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        paragraph = row["Paragraph"]
        grouped.setdefault(paragraph, []).append(row)

    result: list[dict[str, Any]] = []
    for paragraph, paragraph_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        summary = summarize(paragraph_rows)
        result.append(
            {
                "paragraph": int(paragraph),
                "total_tokens": summary["total_tokens"],
                "language_counts": summary["language_counts"],
                "borrowing_counts": summary["borrowing_counts"],
            }
        )
    return result


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
            fieldnames=["Paragraph", "Token", "Language", "Borrowing", "Explanation"],
        )
        writer.writeheader()
        writer.writerows(rows)


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
        rows.extend(
            analyze_paragraph_with_llm(
                paragraph,
                index,
                llm_config=llm_config,
            )
        )
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
