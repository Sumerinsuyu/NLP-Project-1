# Sentiment Format Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify 100 Yelp reviews as positive/negative with a local LLM under 3 Ollama `format` settings, save the raw results, and turn them into two slide charts plus a live one-review demo.

**Architecture:** Same shape as NLP-Project1: `scripts/data_builder.py` freezes the dataset into `data/`, `main.py` runs the model and writes `results/run.json`, `scripts/plot_results.py` reads that file only (no model) and draws the charts. `src/` holds the Ollama client and parser copied from NLP-Project1, plus a new `validate.py` that turns one raw answer into a verdict.

**Tech Stack:** Python 3, `ollama` (Python package), `datasets` (Hugging Face), `matplotlib`. Model `qwen2.5:1.5b` on CPU.

**Spec:** `docs/superpowers/specs/2026-10-26-sentiment-format-demo-design.md`

## Global Constraints

- Everything in English: code, comments, README, chart titles/labels, terminal output (including `--unique`).
- Default model `qwen2.5:1.5b`, CPU. `ollama pull qwen2.5:1.5b` needs the user's OK first.
- Copy then adapt NLP-Project1 files (`~/Documents/Perso/school/NLP-Project1`); never import from it, never modify it.
- NLP-Project1 style: UPPERCASE constants, `ROOT = Path(__file__).resolve()...`, module docstring with a `Usage:` section, short comments, `sys.exit("...")` for user errors.
- **No test files** in the repo (user preference). Verification steps run inline Python (`python - <<'EOF'`) or the scripts themselves; nothing is saved.
- **No commits, no `git init`**: the user commits. Each task ends with "stop for the user".
- Same prompt for the 3 conditions; the prompt is never weakened to make `prompt_only` fail.
- Before writing code that calls `ollama` or `datasets`, check their current API with context7 (`format=` on `Client.chat`, `load_dataset(..., split=...)`).
- Run every Python command from the project root with the venv active (`source venv/bin/activate`).

## Review Focus

1. `--unique` with an id outside 0–99, a non-number, or before `data/reviews.jsonl` exists → one clear `sys.exit` line, no traceback (it runs on stage).
2. Ollama not running, or model not pulled, during `--unique` or a full run → one clear line saying what to start or pull, no traceback.
3. Odd answers: empty string, valid JSON that is not an object (`"positive"`, `[...]`), `{"sentiment": null}`, `{"sentiment": ["positive"]}`, `{"sentiment": " positive"}` → classified (`not_json` or `invalid_value`), never a crash.
4. Interrupted run → `results/run.json` holds fewer than 100 samples; `plot_results.py` must use the real `n` (percentages and titles), not assume 100.
5. `results/run.json` missing → `plot_results.py` exits with "Run `python main.py` first."

The tests for these lines are in Tasks 2, 3 and 4.

---

### Task 1: Project setup and dataset

**Files:**
- Create: `requirements.txt` (`ollama`, `datasets`, `matplotlib`, one per line)
- Create: `.gitignore` (`venv`, `data/`, `results/`, `__pycache__/`, `.superpowers/`)
- Create: `scripts/data_builder.py`

**Interfaces:**
- Produces: `data/reviews.jsonl`, one JSON object per line: `{"id": int 0..99, "stars": int in {1,2,4,5}, "label": "positive"|"negative", "text": str}`.

- [ ] **Step 1: Create the venv and install the dependencies**

Run: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
Expected: installs without error.

- [ ] **Step 2: Write `scripts/data_builder.py`**

Constants: `DATASET = "Yelp/yelp_review_full"`, `SPLIT = "test"`, `PER_STAR = 25`, `STARS = (1, 2, 4, 5)`, `MAX_CHARS = 800`, `SEED = 42`, `ROOT = Path(__file__).resolve().parent.parent`, `OUTPUT = ROOT / "data" / "reviews.jsonl"`.

Functions:
- `clean_text(text: str) -> str`: replaces the literal two-character `\n` with a real newline, strips. Look at a few raw rows first; if other literal escapes show up (e.g. `\"`), undo them here too.
- `sample(rows, rng: random.Random) -> list[dict]`: `stars = label + 1`; skip 3 stars and texts longer than `MAX_CHARS` (after cleaning); `rng.sample(group, PER_STAR)` per star in `STARS`; shuffle the 100 with the same `rng`; number them `id` 0..99; `label = "negative" if stars <= 2 else "positive"`.
- `main()`: `load_dataset(DATASET, split=SPLIT)`, `sample(..., random.Random(SEED))`, writes `OUTPUT` (`ensure_ascii=False`), prints `100 reviews written to data/reviews.jsonl` and the count per star.

- [ ] **Step 3: Run it and check the file**

Run: `python scripts/data_builder.py` (downloads ~23 MB the first time), then:

```sh
python - <<'EOF'
import json; from collections import Counter
rows = [json.loads(l) for l in open("data/reviews.jsonl")]
assert [r["id"] for r in rows] == list(range(100))
assert Counter(r["stars"] for r in rows) == {1: 25, 2: 25, 4: 25, 5: 25}
assert Counter(r["label"] for r in rows) == {"negative": 50, "positive": 50}
assert all((r["stars"] <= 2) == (r["label"] == "negative") for r in rows)
assert all(len(r["text"]) <= 800 and "\\n" not in r["text"] for r in rows)
assert rows[0]["stars"] != rows[1]["stars"] or rows[1]["stars"] != rows[2]["stars"]  # shuffled
print("ok")
EOF
```
Expected: `ok`. Running `data_builder.py` a second time gives the same file (`md5sum` before/after).

- [ ] **Step 4: Stop for the user** (they review and commit)

---

### Task 2: Model client, prompt, schema and answer validation

**Files:**
- Create: `src/json_parser.py` (copy of NLP-Project1's, unchanged)
- Create: `src/llama.py` (copy of NLP-Project1's, adapted)
- Create: `src/prompts/classify.txt`
- Create: `src/schemas/sentiment.json`
- Create: `src/validate.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `src.llama`: `MODEL = "qwen2.5:1.5b"`, `build_prompt(text: str) -> str`, `load_schema() -> dict`, `ask(prompt: str, format: str | dict | None = None) -> str` (raw answer).
  - `src.json_parser`: `parse(raw: str) -> dict` with `status` in `first_try` / `cleaned` / `failed`, `data`, `error` (as in NLP-Project1).
  - `src.validate`: `LABELS = ("positive", "negative")`, `ERROR_TYPES = ("not_json", "text_around", "wrong_key", "invalid_value")`, `judge(raw: str, label: str) -> dict` returning `{"raw", "status", "data", "error", "valid": bool, "error_type": str | None, "prediction": str | None, "correct": bool}`.

- [ ] **Step 1: Copy `json_parser.py`, adapt `llama.py`**

`llama.py` changes only: `MODEL = "qwen2.5:1.5b"`; `MAX_TOKENS = 256` (update the comment: the expected answer is ~10 tokens); `PROMPT_FILE = .../prompts/classify.txt`; `SCHEMA_FILE = .../schemas/sentiment.json`; `ask(prompt, format=None)` passes `format` straight to `client.chat` and the docstring explains the three values (`None`: free answer, `"json"`: JSON mode, a dict: forced schema). Check `format=` with context7 first.

- [ ] **Step 2: Write the prompt and the schema**

`classify.txt`, in the style of NLP-Project1's `extract.txt`: classify the customer review; return ONLY a JSON object with exactly one key `"sentiment"`, value `"positive"` or `"negative"` (lowercase); no markdown, code block or text before/after; then `Review:` and `{text}`.

`sentiment.json`: `{"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative"]}}, "required": ["sentiment"], "additionalProperties": false}`.

- [ ] **Step 3: Write `judge(raw, label)` in `src/validate.py`**

Order of checks (first failure wins): `parse(raw)` status `failed` → `not_json`; `cleaned` → `text_around`; no `"sentiment"` key → `wrong_key`; value not exactly in `LABELS` → `invalid_value` (compare with `isinstance(value, str) and value in LABELS` so lists/null never crash). Valid → `prediction = value`; invalid → `prediction = None`. `correct = valid and prediction == label`. Extra keys are allowed. Module docstring explains "valid (strict)".

- [ ] **Step 4: Check `judge` on hand-written answers (no model)**

```sh
python - <<'EOF'
from src.validate import judge
cases = {
    '{"sentiment": "positive"}':                    (True,  None,            "positive", True),
    '{"sentiment": "negative", "reason": "slow"}':  (True,  None,            "negative", False),
    '```json\n{"sentiment": "positive"}\n```':      (False, "text_around",   None,       False),
    'Here is the JSON: {"sentiment": "positive"}':  (False, "text_around",   None,       False),
    '{"label": "positive"}':                        (False, "wrong_key",     None,       False),
    '{"sentiment": "Positive"}':                    (False, "invalid_value", None,       False),
    '{"sentiment": "mixed"}':                       (False, "invalid_value", None,       False),
    '{"sentiment": " positive"}':                   (False, "invalid_value", None,       False),
    '{"sentiment": null}':                          (False, "invalid_value", None,       False),
    '{"sentiment": ["positive"]}':                  (False, "invalid_value", None,       False),
    '"positive"':                                   (False, "not_json",      None,       False),
    'positive':                                     (False, "not_json",      None,       False),
    '':                                             (False, "not_json",      None,       False),
}
for raw, (valid, err, pred, correct) in cases.items():
    r = judge(raw, "positive")
    assert (r["valid"], r["error_type"], r["prediction"], r["correct"]) == (valid, err, pred, correct), (raw, r)
    assert r["raw"] == raw
print("ok")
EOF
```
Expected: `ok`.

- [ ] **Step 5: Pull the model (ask the user first) and call it once per format**

Run: `ollama pull qwen2.5:1.5b`, then:

```sh
python - <<'EOF'
from src.llama import ask, build_prompt, load_schema
p = build_prompt("The food was cold and the waiter was rude.")
for fmt in (None, "json", load_schema()):
    print(repr(fmt)[:20], "->", repr(ask(p, fmt)))
EOF
```
Expected: three raw answers printed; the schema one is exactly a `{"sentiment": ...}` object.

- [ ] **Step 6: Stop for the user**

---

### Task 3: `main.py`: full run and `--unique`

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: `data/reviews.jsonl` (Task 1); `ask`, `build_prompt`, `load_schema`, `MODEL`, `PROMPT_FILE` (`src.llama`); `judge` (`src.validate`).
- Produces: `results/run.json` = `{"model": str, "prompt": str (template text of classify.txt), "samples": [{"id", "stars", "label", "text", "prompt_only": <judge dict>, "json_mode": <judge dict>, "schema": <judge dict>}]}`. Constant `MODES = ("prompt_only", "json_mode", "schema")`.

- [ ] **Step 1: Write `main.py`**

- `load_reviews() -> list[dict]`: reads `data/reviews.jsonl`; `sys.exit("data/reviews.jsonl not found. Run `python scripts/data_builder.py` first.")` if missing.
- `classify(review: dict, schema: dict) -> dict`: one prompt, one `ask` per mode (format `None` / `"json"` / `schema`), returns the review fields plus `judge(raw, review["label"])` under each mode name.
- `run_all(schema)`: loops the reviews, prints `[i/100] id=..: prompt_only=valid json_mode=text_around ...` (the verdict = `"valid"` or the error type), rewrites `results/run.json` after each review, ends with a short per-mode summary (valid / correct counts) and `Results written to results/run.json`.
- `run_unique(review_id: int | None, schema)`: `None` → random review. Prints the review (id, stars, true label, text), then for each mode a header (`1. Prompt only (no format)`, `2. format="json"`, `3. Forced schema`), the raw answer as returned, and a one-line verdict such as `✔ valid JSON, correct (positive)`, `✔ valid JSON, wrong (negative)` or `✘ invalid: text_around`. Writes nothing.
- argparse: `--unique` with `nargs="?"`, `metavar="ID"`; given without a value → random. Parse the value yourself so `--unique abc` and an id outside the file give `sys.exit("Review id must be between 0 and 99.")`.
- Wrap the Ollama calls: on `ConnectionError` → `sys.exit("Ollama is not running. Start it with `ollama serve`.")`; on `ollama.ResponseError` → `sys.exit(f"Ollama error: {e}. Did you run `ollama pull {MODEL}`?")`.

- [ ] **Step 2: Check the live demo and its errors**

Run each and check the output:
- `python main.py --unique 0` → review shown, 3 blocks with raw answer and verdict, nothing in `results/`.
- `python main.py --unique` → a random review.
- `python main.py --unique 100` and `python main.py --unique abc` → the one-line id message, exit code ≠ 0.
- With Ollama stopped (`systemctl stop ollama` or equivalent; ask the user if needed): `python main.py --unique 0` → the "Ollama is not running" line.
- With `data/` renamed temporarily: the "Run data_builder" line.

- [ ] **Step 3: Full run (≈15 min, run in background)**

Run: `python main.py`, then:

```sh
python - <<'EOF'
import json
run = json.load(open("results/run.json"))
assert run["model"] == "qwen2.5:1.5b" and "{text}" in run["prompt"]
assert len(run["samples"]) == 100
for s in run["samples"]:
    for m in ("prompt_only", "json_mode", "schema"):
        assert set(s[m]) >= {"raw", "valid", "error_type", "prediction", "correct"}
print("ok")
EOF
```
Expected: `ok`. Report the per-mode summary to the user as is (no prompt tweaking).

- [ ] **Step 4: Stop for the user**

---

### Task 4: `scripts/plot_results.py`: table, examples, two charts

**Files:**
- Create: `scripts/plot_results.py` (built from NLP-Project1's `plot_results.py`: same `matplotlib.use("Agg")`, `INK`, `SURFACE`, constrained layout, `dpi=200`)

**Interfaces:**
- Consumes: `results/run.json` (Task 3). Reads it only; never imports `src.llama`.
- Produces: `figures/validity.png`, `figures/accuracy.png`, terminal table + examples.

- [ ] **Step 1: Write the script**

Constants: `MODES` as in `main.py`; `NAMES = {"prompt_only": "Prompt only", "json_mode": 'format="json"', "schema": "Forced schema"}`; colors `GREEN = "#0ca30c"`, `AMBER = "#fab219"`, `RED = "#d03b3b"`.

- `count(samples) -> dict[str, Counter]`: per mode, keys `valid`, `correct`, `wrong` (valid but wrong), `invalid`, and one key per error type.
- `print_table(counts, n)`: one row per mode: valid, correct, valid but wrong, invalid, each error type, accuracy among valid (`correct / valid`, `-` if no valid).
- `print_examples(samples, per_type=2)`: for each error type that occurs, up to 2 examples: mode, review id, raw answer (one line, cut at ~120 chars).
- `plot_validity(counts, n)`: chart 1 = one green bar per mode, height = % valid, label `NN%` above the bar, y axis 0–100 %, title `Valid JSON by format setting (n = {n})`.
- `plot_accuracy(counts, n)`: chart 2 = one bar per mode stacked to 100 %: green `Correct`, amber `Valid JSON but wrong`, red `Invalid JSON`; count written inside each segment ≥ 5 %; legend below; title `Correct answers by format setting (n = {n})`.
- `main()`: `sys.exit("results/run.json not found. Run `python main.py` first.")` if missing; `n = len(samples)`; prints table + examples; saves both PNGs and prints their paths.

- [ ] **Step 2: Run it on the real results and look at the charts**

Run: `python scripts/plot_results.py`
Expected: the table matches the summary printed by `main.py`; both PNGs exist; open them (Read the images) and check bars, labels, percentages and that nothing overlaps.

- [ ] **Step 3: Check the edge cases**

- Rename `results/run.json` temporarily → the "Run `python main.py` first." line.
- Truncated run: write a copy with the first 10 samples to `results/run.json` (keep the real one aside), run → titles say `n = 10` and percentages are out of 10. Restore the real file.

- [ ] **Step 4: Stop for the user**

---

### Task 5: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README in NLP-Project1's style**

Sections: what the demo shows (the two messages); Setup (Ollama + `ollama pull qwen2.5:1.5b`, venv, `python scripts/data_builder.py`); Run (full run, `--unique [ID]`, `plot_results.py`); How it works (table of the 3 conditions with their `format` value; the strict "valid" rule and the 4 error types; "correct"); Result file (`results/run.json` shape, short example); Project structure tree. Mention that the 100 results are precomputed before the talk and only `--unique` runs live.

- [ ] **Step 2: Follow the README from a clean shell**

Run every command from the README in order in a new shell (skip the 15-min full run if `results/run.json` is already there) and check each works as written.

- [ ] **Step 3: Stop for the user**
