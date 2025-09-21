import logging
import json
import re

def extract_json(raw: str) -> str:
    """Extract first JSON object/array from text output."""
    match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw

def safe_json_parse(raw: str):
    """Try to parse JSON safely, repair if needed."""
    try:
        return json.loads(raw)
    except Exception:
        fixed = extract_json(raw)
        try:
            return json.loads(fixed)
        except Exception as e:
            logging.error(f"[safe_json_parse] Failed to parse JSON: {e}")
            return {"error": "Invalid JSON output", "raw": raw}

def run_debug(query: str, model_provider="ollama"):
    from v1 import run_stock_research  # import your function

    print("\n=== USER QUERY ===")
    print(query)

    # Run agent
    result = run_stock_research(query, model_provider)

    print("\n=== RAW OUTPUT FROM MODEL ===")
    print(result)

    # Try parsing JSON
    parsed = safe_json_parse(result)
    print("\n=== PARSED JSON ===")
    print(json.dumps(parsed, indent=2))

    return parsed

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    query = "Comprehensive 3-month analysis of Apple Inc. (AAPL)"
    run_debug(query)
