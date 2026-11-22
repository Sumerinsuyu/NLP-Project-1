from src.json_parser import parse

LABELS = ("positive", "negative")
ERROR_TYPES = ("not_json", "text_around", "wrong_key", "invalid_value")

def error_type(parsed: dict) -> str | None:
    """The first check the parsed answer fails, or None if it is valid."""
    if parsed["status"] == "failed":
        return "not_json"
    if parsed["status"] == "cleaned":
        return "text_around"
    if "sentiment" not in parsed["data"]:
        return "wrong_key"
    value = parsed["data"]["sentiment"]
    if not (isinstance(value, str) and value in LABELS):
        return "invalid_value"
    return None

def judge(raw: str, label: str) -> dict:
    parsed = parse(raw)
    error = error_type(parsed)
    prediction = parsed["data"]["sentiment"] if error is None else None
    return {
        "raw": raw,
        **parsed,
        "valid": error is None,
        "error_type": error,
        "prediction": prediction,
        "correct": prediction == label,
    }
