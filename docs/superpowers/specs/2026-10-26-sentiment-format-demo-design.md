# Demo: a JSON format guarantees the shape, not the answer

Date: 2026-10-26. Project: `~/Documents/Perso/school/NLP-Pres` (empty, not a git repo yet).
Based on `~/Documents/Perso/school/NLP-Project1` (Harry Potter JSON extraction with Ollama).

## Intent

A demo for an **oral presentation**. Two messages:

1. The more Ollama is constrained (prompt only → `format="json"` → forced
   schema), the more often the JSON comes out valid. A prompt alone guarantees
   less.
2. Valid JSON is not necessarily a **correct** answer: even with the forced
   schema, the model can get the sentiment wrong.

Task: classify customer reviews as `positive` / `negative` with a local LLM.

## Goals

- 100 reviews, 3 conditions, **exactly the same prompt**; only Ollama's
  `format` parameter changes:
  - `prompt_only`: no `format`
  - `json_mode`: `format="json"`
  - `schema`: `format` = a JSON schema with `sentiment` as enum `["positive", "negative"]`
- Two images for the slides, built from the saved results without running the
  model again. The 100 reviews are precomputed before the talk.
- One live command on **a single review** that shows the answer of each of the
  3 conditions.

## Non-goals / anti-goals

- No tests (same as NLP-Project1: short, explainable code, a README instead).
- Do not touch NLP-Project1.
- **No rigging**: the prompt is written honestly, like NLP-Project1's (it asks
  for strict JSON). It is never weakened to make `prompt_only` fail.
- No `--model` option and no fallback model: the user is confident the
  precomputed 100 reviews will tell the story.

## Constraints

- **Everything in English**: code, comments, README, chart titles and labels,
  terminal output (including the live `--unique` output).
- Default model: `qwen2.5:1.5b`, on CPU. It is not installed yet:
  `ollama pull qwen2.5:1.5b` (ask the user first).
- Reuse NLP-Project1's code and style as much as possible: same helpers, same
  layout, UPPERCASE constants, `ROOT = Path(__file__)...`, module docstring with
  a "Usage:" section, short comments.
- NLP-Project1 files are **copied then adapted** into NLP-Pres (no import across
  the two repos).
- The user makes the commits (and the `git init` if they want one).
- Check the `ollama` (`format`) and `datasets` APIs with context7 before writing code.

## Data (decided)

- Hugging Face dataset `Yelp/yelp_review_full`, split `test` (50,000 reviews,
  fields `label` 0–4 = 1–5 stars, `text`). Checked on 2026-10-26.
- 1–2 stars = `negative`, 4–5 = `positive`, 3 stars excluded.
- 100 balanced reviews: 50 negative, 50 positive.
- The sample is frozen once into `data/reviews.jsonl` by
  `scripts/data_builder.py`, so every condition and every rerun sees the same reviews.

*Claude's call:* 25 reviews per star count (1, 2, 4, 5), to include the
lukewarm reviews (2 and 4 stars) where the model can be wrong; fixed seed (42);
reviews longer than 800 characters are skipped to keep CPU time reasonable.
The 100 reviews are shuffled (same seed) before being numbered. Yelp's literal
`\n` sequences are replaced by real newlines. One line per review:
`{"id": 0..99, "stars", "label", "text"}`.

## Measures (decided)

For each answer:

- **Valid (strict)** = the raw answer passes `json.loads` on the first try
  (`parse()` status `first_try`), has the key `sentiment`, and its value is
  exactly `"positive"` or `"negative"`.
  *Claude's call:* extra keys (e.g. `"reason"`) are tolerated.
- Otherwise, **one** error type, the first one met in this order:
  - `not_json`: no JSON, even after cleaning (`parse()` → `failed`)
  - `text_around`: JSON only after cleaning (`cleaned`): markdown fences or text around it
  - `wrong_key`: valid JSON but no `sentiment` key
  - `invalid_value`: right key, value outside the enum (`"Positive"`, `"mixed"`, `"neutral"`…)
- **Correct** = valid and prediction = true label.
  *Claude's call:* for an invalid answer the prediction is `null` and `correct`
  is `false` (it counts as "invalid", never as "correct").

Per condition, over the 100 reviews: number valid, number correct, number
"valid but wrong", number invalid (with the breakdown by error type), and the
accuracy among valid answers (= correct / valid).

## Raw results (decided)

`main.py` writes **one file**, `results/run.json`: the model, the prompt, then
for each review its fields and, for each condition, the raw answer, the parse
status, the parsed data, the error type, the prediction and whether it is
correct. The raw answer is **kept** (NLP-Project1 removed it) because the
error examples need it.
*Claude's call:* the file is rewritten after each review, so a stopped run
(~15 minutes) is not lost. A full run always writes the 3 conditions; there is
no `--mode` option (not requested, `--unique` covers quick checks, and a
partial run could overwrite or mix `run.json`).

## The two charts (decided with the user)

Built by `scripts/plot_results.py` from `results/run.json`, as PNGs for the
slides. One bar per condition, x axis: "Prompt only", "format=\"json\"",
"Forced schema".

- **Chart 1** (`figures/validity.png`): one green bar per condition = % of
  valid JSON (**simple** version, no error breakdown).
- **Chart 2** (`figures/accuracy.png`): one bar **stacked to 100** per
  condition: green = correct, amber = valid JSON but wrong, red = invalid JSON.
  The count is written inside each segment that is big enough.

*Claude's call:* NLP-Project1's colors (`#0ca30c`, `#fab219`, `#d03b3b`).

The same script prints in the terminal (for the speaker, not the slides): the
comparison table of the 3 conditions (valid, correct, valid but wrong, invalid
by error type, accuracy among valid) and a few real error examples (raw answer
+ type).

## Commands

```sh
python scripts/data_builder.py     # downloads Yelp, writes data/reviews.jsonl
python main.py                     # 100 reviews × 3 conditions → results/run.json
python main.py --unique            # live demo: one random review
python main.py --unique 12         # live demo: review id 12
python scripts/plot_results.py     # table + examples + figures/*.png
```

`--unique` (asked for by the user): runs one review in the 3 conditions and
shows, for each, the raw JSON / answer of the model.
*Claude's call:* it first shows the review, its stars and its true label, and
under each answer a one-line verdict (valid or error type, correct or wrong).
It does not write to `results/`. Without an id, the review is picked at random.

## Layout

```
NLP-Pres/
  README.md
  requirements.txt        ollama, datasets, matplotlib
  .gitignore              venv, data/, results/, __pycache__/, .superpowers/
  main.py                 the experiment (all reviews, or --unique)
  src/
    llama.py              copy of NLP-Project1's: MODEL = "qwen2.5:1.5b",
                          ask(prompt, format) takes None, "json" or a dict
    json_parser.py        copied as is from NLP-Project1
    validate.py           parse() + key + enum → valid / error type / prediction
    prompts/classify.txt  the single prompt ({text} replaced by the review)
    schemas/sentiment.json
  scripts/
    data_builder.py       Hugging Face → data/reviews.jsonl
    plot_results.py       results/run.json → table, examples, 2 PNGs
  figures/                validity.png, accuracy.png
```

*Claude's call:* `num_predict` lowered to 256 (the expected answer is ~10
tokens; the limit cuts a model stuck in a loop). `temperature: 0`,
`keep_alive` and the client timeout stay as in NLP-Project1.

## Left to the builder

- The exact prompt wording, in the style of NLP-Project1's `extract.txt` (asks
  for a JSON object with the single key `sentiment`, lowercase values, no
  markdown or text around it), within the "no rigging" anti-goal.
- The terminal layout (`--unique` and the table), how many error examples to show.
- matplotlib details, following NLP-Project1's `plot_results.py`.
