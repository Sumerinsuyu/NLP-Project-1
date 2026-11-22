"""Turn results/run.json into a comparison table and the two slide charts.

Reads only results/run.json written by main.py (the model is not called), then:
  - prints the comparison table of the 3 format settings and a few real errors
  - saves figures/validity.png: % of valid JSON per setting
  - saves figures/accuracy.png: correct / valid but wrong / invalid, per setting

Usage:
    python scripts/plot_results.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results" / "run.json"
FIGURES_DIR = ROOT / "figures"

MODES = ("prompt_only", "json_mode", "schema")
NAMES = {"prompt_only": "Prompt only", "json_mode": 'format="json"', "schema": "Forced schema"}
ERROR_TYPES = ("not_json", "text_around", "wrong_key", "invalid_value")

# green = correct, amber = valid but wrong, red = invalid (checked for colorblind readers)
GREEN, AMBER, RED = "#0a850a", "#eba00c", "#d03b3b"
INK, MUTED, SURFACE = "#1a1a19", "#6b6b66", "#ffffff"


def count(samples: list[dict]) -> dict[str, Counter]:
    """{mode: Counter} with valid, correct, wrong (valid but wrong), invalid and each error type."""
    counts = {mode: Counter() for mode in MODES}
    for sample in samples:
        for mode in MODES:
            answer, c = sample[mode], counts[mode]
            if answer["valid"]:
                c["valid"] += 1
                c["correct" if answer["correct"] else "wrong"] += 1
            else:
                c["invalid"] += 1
                c[answer["error_type"]] += 1
    return counts


def print_table(counts: dict[str, Counter], n: int) -> None:
    columns = ("valid", "correct", "wrong", "invalid", *ERROR_TYPES)
    print(f"Over {n} reviews (wrong = valid JSON but wrong sentiment):\n")
    print(f"{'':<14}" + "".join(f"{col:>{len(col) + 2}}" for col in columns) + "  acc. on valid")
    for mode in MODES:
        c = counts[mode]
        accuracy = f"{c['correct'] / c['valid']:.0%}" if c["valid"] else "-"
        print(f"{NAMES[mode]:<14}" + "".join(f"{c[col]:>{len(col) + 2}}" for col in columns) + f"{accuracy:>15}")


def print_examples(samples: list[dict], per_type: int = 2) -> None:
    print("\nError examples (raw answer of the model):")
    for error in ERROR_TYPES:
        found = [(mode, s) for s in samples for mode in MODES if s[mode]["error_type"] == error]
        if not found:
            continue
        print(f"\n  {error} ({len(found)} answers)")
        for mode, sample in found[:per_type]:
            raw = " ".join(sample[mode]["raw"].split()) or "(empty answer)"  # one line
            raw = raw if len(raw) <= 120 else raw[:117] + "..."
            print(f"    [{NAMES[mode]}, review {sample['id']}] {raw}")


def new_axes(title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE, layout="constrained")
    ax.set_title(title, color=INK, fontsize=13, pad=14)
    ax.set_ylim(0, 110)  # room for the label above a 100% bar
    ax.set_yticks(range(0, 101, 25), [f"{y}%" for y in range(0, 101, 25)])
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="x", colors=INK, labelsize=11)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    return fig, ax


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_validity(counts: dict[str, Counter], n: int) -> Path:
    """Chart 1: one green bar per setting, height = % of valid JSON."""
    fig, ax = new_axes(f"Valid JSON by format setting (n = {n})")
    names = [NAMES[mode] for mode in MODES]
    pcts = [100 * counts[mode]["valid"] / n for mode in MODES]
    ax.bar(names, pcts, width=0.55, color=GREEN)
    for x, pct in enumerate(pcts):
        ax.text(x, pct + 2, f"{pct:.0f}%", ha="center", va="bottom", color=INK, fontsize=12, fontweight="bold")
    return save(fig, "validity.png")


def plot_accuracy(counts: dict[str, Counter], n: int) -> Path:
    """Chart 2: one bar per setting stacked to 100%: correct, valid but wrong, invalid."""
    fig, ax = new_axes(f"Correct answers by format setting (n = {n})")
    names = [NAMES[mode] for mode in MODES]
    bottom = [0.0] * len(MODES)
    for key, label, color in (("correct", "Correct", GREEN),
                              ("wrong", "Valid JSON but wrong", AMBER),
                              ("invalid", "Invalid JSON", RED)):
        pcts = [100 * counts[mode][key] / n for mode in MODES]
        # white edge = 2px gap between the segments
        ax.bar(names, pcts, width=0.55, bottom=bottom, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        for x, (mode, pct) in enumerate(zip(MODES, pcts)):
            if pct >= 5:  # a count only where it fits
                ax.text(x, bottom[x] + pct / 2, str(counts[mode][key]), ha="center", va="center",
                        color=INK, fontsize=11, fontweight="bold")
        bottom = [b + p for b, p in zip(bottom, pcts)]
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), labelcolor=INK)
    return save(fig, "accuracy.png")


def main() -> None:
    if not RESULTS_FILE.exists():
        sys.exit("results/run.json not found. Run `python main.py` first.")
    run = json.loads(RESULTS_FILE.read_text())
    samples = run["samples"]
    if not samples:
        sys.exit("results/run.json has no reviews. Run `python main.py` first.")

    n = len(samples)
    counts = count(samples)
    print(f"Model: {run['model']}")
    print_table(counts, n)
    print_examples(samples)
    print()
    for path in (plot_validity(counts, n), plot_accuracy(counts, n)):
        print(f"Chart saved to {path}")


if __name__ == "__main__":
    main()
