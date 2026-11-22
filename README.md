# NLP-Pres

A demo for a talk: **a JSON format guarantees the shape of an answer, not that
the answer is right.**

A local LLM classifies 100 Yelp reviews as `positive` or `negative`. The same
prompt is sent in 3 settings; only Ollama's `format` parameter changes. Two
charts then show:

1. the more the output is constrained, the more often the JSON is valid;
2. valid JSON is not always a correct answer: even with a forced schema, the
   model can get the sentiment wrong.

The 100 results are computed before the talk. Only `--unique` runs live.

Based on [NLP-Project1](../NLP-Project1) (same Ollama client and JSON parser).

## Setup

1. Install [Ollama](https://ollama.com/download) and download the model
   (it runs on the CPU):

   ```sh
   ollama pull qwen2.5:1.5b
   ```

   Ollama must be running (`ollama serve`, or the system service).

2. Create the virtual environment and install the dependencies:

   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Build the dataset (downloads ~23 MB from Hugging Face the first time):

   ```sh
   python scripts/data_builder.py
   ```

## Run

**All 100 reviews**, before the talk (300 calls, ~3 minutes on an 8-core CPU):

```sh
python main.py
```

Every answer is saved to `results/run.json`, rewritten after each review, so a
stopped run keeps what is done. A new run replaces the file.

**The charts and the table**, from the saved results (the model is not called):

```sh
python scripts/plot_results.py
```

It prints the comparison table and a few real error examples, and saves:

| File | Chart |
|---|---|
| `figures/validity.png` | % of valid JSON per setting |
| `figures/accuracy.png` | per setting, out of 100: correct / valid JSON but wrong / invalid JSON |

**Live demo on one review**, the 3 answers printed one after the other:

```sh
python main.py --unique       # a random review
python main.py --unique 12    # the review with id 12 (0 to 99)
```

It prints the review, its stars and true label, then for each setting the raw
answer of the model and a verdict (`✔ valid JSON, correct`, `✘ invalid:
text_around`, ...). It writes nothing.

## How it works

Each review is sent 3 times with the prompt `src/prompts/classify.txt`:

| Setting | `format` passed to Ollama | What the model is held to |
|---|---|---|
| `prompt_only` | none | only the prompt asks for JSON |
| `json_mode` | `"json"` | valid JSON, but any keys and values |
| `schema` | `src/schemas/sentiment.json` | an object with `sentiment` = `"positive"` or `"negative"` |

An answer is **valid** (strict) only if all three hold:

1. the raw answer is JSON as is, without cleaning;
2. it has the key `sentiment`;
3. the value is exactly `"positive"` or `"negative"`.

Otherwise it gets one error type, the first check it fails:

| Error | Example |
|---|---|
| `not_json` | `The review is positive.` |
| `text_around` | ```` ```json {"sentiment": "positive"} ``` ```` or `Here is the JSON: {...}` |
| `wrong_key` | `{"label": "positive"}` |
| `invalid_value` | `{"sentiment": "Positive"}`, `{"sentiment": "mixed"}` |

An answer is **correct** if it is valid and matches the label of the review.

## Dataset

`scripts/data_builder.py` takes the test split of
[`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full):

- 1–2 stars = `negative`, 4–5 stars = `positive`, 3 stars excluded;
- 25 reviews for each of 1, 2, 4 and 5 stars (50 negative, 50 positive);
- reviews over 800 characters are skipped (CPU time);
- fixed seed, so the file is the same at every build.

`data/reviews.jsonl`, one review per line:

```json
{"id": 0, "stars": 1, "label": "negative", "text": "The taxi never showed up. ..."}
```

## Result file

`results/run.json`:

```json
{
  "model": "qwen2.5:1.5b",
  "prompt": "Classify the sentiment of the customer review below. ...",
  "samples": [
    {
      "id": 0, "stars": 1, "label": "negative", "text": "...",
      "prompt_only": {
        "raw": "{\"sentiment\": \"negative\"}",
        "status": "first_try", "data": {"sentiment": "negative"}, "error": null,
        "valid": true, "error_type": null, "prediction": "negative", "correct": true
      },
      "json_mode": { "...": "..." },
      "schema": { "...": "..." }
    }
  ]
}
```

`status`, `data` and `error` come from the JSON parser (`first_try`, `cleaned`
or `failed`, as in NLP-Project1).

## Project structure

```
main.py                    runs the 3 settings on every review, or on one (--unique)
src/
  llama.py                 calls the model through the ollama package
  json_parser.py           cleans and parses the answer (from NLP-Project1)
  validate.py              valid / error type / correct, for one answer
  prompts/classify.txt     the prompt ({text} is replaced by the review)
  schemas/sentiment.json   the JSON schema of the forced setting
scripts/
  data_builder.py          builds data/reviews.jsonl from Hugging Face (run it first)
  plot_results.py          table, error examples and charts from results/run.json
data/                      the 100 reviews (generated, not in git)
results/                   run.json (generated, not in git)
figures/                   the charts (generated)
```
