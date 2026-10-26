"""Machine-readable CLI output; metric reports/plots belong to later stages."""

import json


def format_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
