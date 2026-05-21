from collections import Counter
from typing import Any


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    language_counts = Counter(r["Language"] for r in rows)
    borrowing_counts = Counter(r["Borrowing"] for r in rows)
    pos_counts = Counter(r["POS"] for r in rows)

    return {
        "total_tokens": len(rows),
        "language_counts": dict(language_counts),
        "borrowing_counts": dict(borrowing_counts),
        "pos_counts": dict(pos_counts),
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
                "pos_counts": summary["pos_counts"],
            }
        )
    return result
