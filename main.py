"""Classify the sentiment of each review with the local LLM, in 3 format settings.

The same prompt is sent 3 times; only Ollama's format parameter changes:
  prompt_only  no format
  json_mode    format="json"
  schema       format = the JSON schema (sentiment must be "positive" or "negative")

Usage:
    python main.py                # every review in data/ -> results/run.json
    python main.py --unique       # one random review, the 3 answers printed
    python main.py --unique 12    # the review with id 12
"""

import argparse
import json
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

import ollama

from src.llama import MODEL, PROMPT_FILE, ask, build_prompt, load_schema
from src.validate import judge

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "reviews.jsonl"
RESULTS_FILE = ROOT / "results" / "run.json"

MODES = ("prompt_only", "json_mode", "schema")
TITLES = {
    "prompt_only": "1. Prompt only (no format)",
    "json_mode": '2. format="json"',
    "schema": "3. Forced schema",
}


def load_reviews() -> list[dict]:
    if not DATA_FILE.exists():
        sys.exit("data/reviews.jsonl not found. Run `python scripts/data_builder.py` first.")
    return [json.loads(line) for line in DATA_FILE.read_text().splitlines()]


def formats(schema: dict) -> dict:
    return {"prompt_only": None, "json_mode": "json", "schema": schema}


def classify(review: dict, schema: dict) -> dict:
    """Ask the model about one review in each mode and judge each answer."""
    prompt = build_prompt(review["text"])
    result = dict(review)
    for mode, format in formats(schema).items():
        result[mode] = judge(ask(prompt, format), review["label"])
    return result


def verdict(answer: dict) -> str:
    if not answer["valid"]:
        return f"✘ invalid: {answer['error_type']}"
    return f"✔ valid JSON, {'correct' if answer['correct'] else 'wrong'} ({answer['prediction']})"


def to_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def run_all(schema: dict) -> None:
    reviews = load_reviews()
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    run = {"model": MODEL, "prompt": PROMPT_FILE.read_text(), "samples": []}
    for i, review in enumerate(reviews, 1):
        result = classify(review, schema)
        run["samples"].append(result)
        RESULTS_FILE.write_text(to_json(run))  # after each review: a stopped run keeps what is done
        print(f"[{i}/{len(reviews)}] id={review['id']}: " + "  ".join(
            f"{mode}={'valid' if result[mode]['valid'] else result[mode]['error_type']}" for mode in MODES))

    total = len(reviews)
    print(f"\nOver {total} reviews:")
    for mode in MODES:
        c = Counter()
        for sample in run["samples"]:
            c["valid"] += sample[mode]["valid"]
            c["correct"] += sample[mode]["correct"]
        print(f"  {mode:<12} valid: {c['valid']} ({c['valid'] / total:.0%})"
              f"  correct: {c['correct']} ({c['correct'] / total:.0%})")
    print(f"Results written to {RESULTS_FILE}")


def run_unique(review_id: int | None, schema: dict) -> None:
    """Show one review and the model's answer in each mode. Writes nothing."""
    reviews = load_reviews()
    review = random.choice(reviews) if review_id is None else reviews[review_id]

    stars = f"{review['stars']} star" + ("s" if review["stars"] > 1 else "")
    print(f"Review {review['id']} ({stars}, true label: {review['label']})\n")
    for paragraph in review["text"].splitlines():
        print(textwrap.fill(paragraph, width=88, initial_indent="  ", subsequent_indent="  "))

    prompt = build_prompt(review["text"])
    for mode, format in formats(schema).items():
        print(f"\n{TITLES[mode]}")
        answer = judge(ask(prompt, format), review["label"])
        print(textwrap.indent(answer["raw"] or "(empty answer)", "  "))
        print(f"  -> {verdict(answer)}")


def parse_id(value: str) -> int:
    last = len(load_reviews()) - 1
    try:
        review_id = int(value)
    except ValueError:
        review_id = -1
    if not 0 <= review_id <= last:
        sys.exit(f"Review id must be between 0 and {last}.")
    return review_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unique", nargs="?", const="random", metavar="ID",
                        help="run one review only (random if no ID) and print the 3 answers")
    args = parser.parse_args()

    schema = load_schema()
    try:
        if args.unique is None:
            run_all(schema)
        else:
            run_unique(None if args.unique == "random" else parse_id(args.unique), schema)
    except ConnectionError:
        sys.exit("Ollama is not running. Start it with `ollama serve`.")
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e.error}. Did you run `ollama pull {MODEL}`?")


if __name__ == "__main__":
    main()
