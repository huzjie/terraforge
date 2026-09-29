"""Filesystem I/O helpers."""
import json
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_text(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path, content, encoding="utf-8"):
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj, indent=2):
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path
