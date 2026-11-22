"""Build the dataset: 100 balanced Yelp reviews in data/reviews.jsonl.

Downloads the test split of Yelp/yelp_review_full from Hugging Face, keeps
25 reviews for each of 1, 2, 4 and 5 stars (3 stars are neither positive nor
negative), and freezes them in one file so every run sees the same reviews.

Usage:
    python scripts/data_builder.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET = "Yelp/yelp_review_full"
SPLIT = "test"
STARS = (1, 2, 4, 5)
PER_STAR = 25
# long reviews make each call slower on the CPU
MAX_CHARS = 800
SEED = 42
ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "reviews.jsonl"

def unescape(match: re.Match) -> str:
    code = match.group(1)
    if code.startswith("u"):  # ä -> ä
        return chr(int(code[1:], 16))
    return {"n": "\n", "r": ""}.get(code, code)  # \" -> ", \\ -> \


def clean_text(text: str) -> str:
    """Yelp keeps escape sequences as literal text: \\n, \\", \\u00e4, ..."""
    return ESCAPE.sub(unescape, text).strip()


def sample(rows, rng: random.Random) -> list[dict]:
    """PER_STAR reviews for each star count in STARS, shuffled and numbered."""
    groups = {stars: [] for stars in STARS}
    for row in rows:
        stars = row["label"] + 1  # label 0..4 = 1..5 stars
        text = clean_text(row["text"])
        if stars in groups and len(text) <= MAX_CHARS:
            groups[stars].append({"stars": stars, "text": text})

    reviews = [review for stars in STARS for review in rng.sample(groups[stars], PER_STAR)]
    rng.shuffle(reviews)
    return [
        {"id": i, "stars": r["stars"], "label": "negative" if r["stars"] <= 2 else "positive", "text": r["text"]}
        for i, r in enumerate(reviews)
    ]


def main() -> None:
    reviews = sample(load_dataset(DATASET, split=SPLIT), random.Random(SEED))

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews))

    print(f"{len(reviews)} reviews written to {OUTPUT}")
    counts = Counter(r["stars"] for r in reviews)
    print("  " + "  ".join(f"{stars} stars: {counts[stars]}" for stars in STARS))


if __name__ == "__main__":
    main()
