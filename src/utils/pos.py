from __future__ import annotations

UPOS_TAGS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}


def upos_instruction_block() -> str:
    return "- POS (UPOS): ADJ | ADP | ADV | AUX | CCONJ | DET | INTJ | NOUN | NUM | PART | PRON | PROPN | PUNCT | SCONJ | SYM | VERB | X"


def normalize_pos_tag(raw_tag: str) -> str:
    candidate = str(raw_tag).upper().strip()
    if candidate in UPOS_TAGS:
        return candidate
    return "X"
