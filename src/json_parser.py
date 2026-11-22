import json
import re

FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)

def clean(raw: str) -> str:
    text = FENCE.sub("", raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return text.strip()
    return text[start:end + 1]

def load_object(text: str) -> dict:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data

def parse(raw: str) -> dict:
    try:
        return {"status": "first_try", "data": load_object(raw), "error": None}
    except ValueError:
        pass
    try:
        return {"status": "cleaned", "data": load_object(clean(raw)), "error": None}
    except ValueError as e:
        return {"status": "failed", "data": None, "error": str(e)}
