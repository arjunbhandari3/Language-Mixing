import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from utils.pos import upos_instruction_block


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


def build_llm_prompt(tokens: list[str], paragraph_text: str) -> str:
    token_json = json.dumps(tokens, ensure_ascii=False)

    return f"""
You are a linguistics annotator for Nepali-English code-mixed text.

Annotate every token in order and return STRICT JSON only.

Labels:

Language:
- NE = Nepali
- EN = English
- OTHER = punctuation, symbols, emoji, URLs, numbers, unknown

Borrowing:
- NATIVE = native Nepali word
- BORROWED = English-origin loanword adapted into Nepali usage
- CODE-MIXED = direct English insertion/code-switching
- UNKNOWN = unclear

Rules:
- Decide language from the written form/script.
- English-origin words written in Devanagari are usually:
  language=NE, borrowing=BORROWED
- English words directly written in Roman script inside Nepali text are usually:
  language=EN, borrowing=CODE-MIXED
- Nepali suffixes do not change borrowing status.

Examples:
- कम्प्युटर -> NE + BORROWED
- कम्प्युटरमा -> NE + BORROWED
- प्रोसेसरको -> NE + BORROWED
- मोबाइल -> NE + BORROWED
- किताब -> NE + NATIVE
- meeting -> EN + CODE-MIXED

POS tags:
ADJ | ADP | ADV | AUX | CCONJ | DET | INTJ | NOUN | NUM |
PART | PRON | PROPN | PUNCT | SCONJ | SYM | VERB | X

{upos_instruction_block()}

Paragraph:
{paragraph_text}

Tokens:
{token_json}

Return:
{{
  "items": [
    {{
      "token": "...",
      "language": "NE|EN|OTHER",
      "borrowing": "NATIVE|BORROWED|CODE-MIXED|UNKNOWN",
      "pos": "...",
      "explanation": "short reason"
    }}
  ]
}}
""".strip()

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
