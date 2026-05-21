# Translation Paragraph Analysis

This project analyzes extracted translation text paragraphs using an LLM-based classifier:

- `Token`
- `Language` (`NE` / `EN`)
- `Borrowing` (`NATIVE` / `BORROWED` / `CODE-MIXED`)
- `Explanation`

## Architecture

The project follows a modular pipeline architecture:

```
Input Text File
      ↓
Parse Paragraphs (main.py / analyze_translations.py)
      ↓
LLM Classifier (via LM Studio API)
      ↓
Process Results
      ↓
Output Files (CSV, JSON)
```

## Project Structure

```text
Language-Mixing/
├─ main.py
├─ sample_paragraphs.txt
├─ requirements.txt
├─ src/
│  ├─ analyze_translations.py
│  ├─ config/
│  │  └─ llm.json
│  └─ output/
└─ README.md
```

### Components

- **Input Layer**: Text files with paragraphs separated by blank lines
- **Processing Core**: `src/analyze_translations.py` — Handles paragraph parsing and LLM API communication
- **LLM Integration**: Connects to LM Studio local server for token classification
- **Output Layer**: Generates structured results in CSV and JSON formats
- **Configuration**: `src/config/llm.json` — Centralized model and endpoint settings
- **Entry Points**: `main.py` for CLI execution

### Data Flow

1. Read input text file and split into paragraphs
2. Send each token/paragraph to LM Studio API
3. Receive classification results (token, language, borrowing type, explanation)
4. Aggregate results into analysis, summary, and paragraph summary files

## 1) Environment (venv)

From workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your policy blocks script activation, you can still run with:

```powershell
.\.venv\Scripts\python.exe src\analyze_translations.py --help
```

## 2) Configure LM Studio Settings

Edit `src/config/llm.json` to set your model name and LM Studio server URL:

```json
{
  "model": "<your-loaded-model-id>",
  "base_url": "http://127.0.0.1:1234/v1",
  "api_key": "lm-studio"
}
```

Notes:

- `model` must match the model ID shown in LM Studio when the server is running.
- `base_url` is the LM Studio local server URL (default `http://127.0.0.1:1234/v1`).
- Start server in LM Studio → Local Server → Start Server.

## 3) Prepare Input Text

- Create a `.txt` file with extracted translation paragraphs.
- Separate each paragraph using a blank line.

## 4) Analyze paragraphs using LLM

```powershell
.\.venv\Scripts\python.exe src\analyze_translations.py sample_paragraphs.txt --out src/output
```

Default dataset split setup:

- Training: 70%
- Validation: 15%
- Test: 15%

You can override these with CLI flags if needed:

```powershell
.\.venv\Scripts\python.exe src\analyze_translations.py sample_paragraphs.txt --out src/output --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Input format for paragraph analysis:

- Plain text file where each paragraph is separated by a blank line.

Outputs:

- `src/output/<input_stem>_analysis.csv`
- `src/output/<input_stem>_summary.json`
- `src/output/<input_stem>_paragraph_summary.json`
- `src/output/<input_stem>_train.csv`
- `src/output/<input_stem>_validation.csv`
- `src/output/<input_stem>_test.csv`

Notes:

- Model and endpoint are configured in `src/config/llm.json`.
- Requires LM Studio server running with a chat/instruct model loaded.
- Uses LLM for all token classification decisions.

## 5) Run via main entrypoint

```powershell
.\.venv\Scripts\python.exe main.py sample_paragraphs.txt --out src/output
```

This command runs text analysis and generates output files using the LM Studio config.
