import json
from pathlib import Path

import ollama

MODEL = "qwen2.5:1.5b"

# The expected answer is ~10 tokens. Without a limit, Ollama lets the model
# generate forever, and a small model sometimes gets stuck repeating itself.
# The limit cuts such an answer off (it then fails to parse) instead of
# blocking the run.
MAX_TOKENS = 256

SRC = Path(__file__).resolve().parent
PROMPT_FILE = SRC / "prompts" / "classify.txt"
SCHEMA_FILE = SRC / "schemas" / "sentiment.json"

# the default client waits forever; give up on a call after 10 minutes
client = ollama.Client(timeout=600)

def build_prompt(text: str) -> str:
    return PROMPT_FILE.read_text().replace("{text}", text)

def load_schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text())

def ask(prompt: str, format: str | dict | None = None) -> str:
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=format,
        options={
            "temperature": 0,
            "num_predict": MAX_TOKENS,
        },
        keep_alive="30m",
    )
    return response.message.content
