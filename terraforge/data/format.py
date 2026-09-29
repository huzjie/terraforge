"""Data format helpers: export records to JSON / JSONL."""
import json
from ..utils.io import ensure_dir


def to_jsonl(records, path):
    ensure_dir(path.parent if hasattr(path, "parent") else ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def to_json(records, path):
    from ..utils.io import write_json
    return write_json(path, records)
