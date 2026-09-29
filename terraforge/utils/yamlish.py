"""Minimal YAML subset parser (no PyYAML dependency).

Supports nested mappings, scalar values (int/float/bool/str/quoted), lists of
scalars and lists of mappings, and inline `#` comments.
"""
import re


def _strip_comment(line):
    for i, ch in enumerate(line):
        if ch == "#":
            if i == 0 or line[i - 1] in " \t":
                return line[:i].rstrip()
    return line.rstrip()


def _scalar(s):
    s = s.strip()
    if not s:
        return ""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    if re.match(r"^[-+]?\d+$", s):
        try:
            return int(s)
        except ValueError:
            pass
    if re.match(r"^[-+]?\d*\.\d+([eE][-+]?\d+)?$", s) or re.match(r"^[-+]?\d+[eE][-+]?\d+$", s):
        try:
            return float(s)
        except ValueError:
            pass
    return s


def _inline_map(s):
    key, _, val = s.partition(":")
    return {key.strip(): _scalar(val)}


def _parse_value(entries, i, indent):
    if i >= len(entries):
        return {}, i
    content = entries[i][1]
    if content.startswith("- "):
        result = []
        while i < len(entries):
            ind, c = entries[i]
            if c.startswith("- ") and ind == indent:
                item = c[2:].strip()
                if i + 1 < len(entries) and entries[i + 1][0] > indent and ":" in item and item.split(":", 1)[1].strip() == "":
                    key = item.split(":", 1)[0].strip()
                    child, i = _parse_value(entries, i + 1, entries[i + 1][0])
                    result.append({key: child})
                elif i + 1 < len(entries) and entries[i + 1][0] > indent and ":" in item:
                    d = {item.split(":", 1)[0].strip(): _scalar(item.split(":", 1)[1])}
                    i += 1
                    while i < len(entries) and entries[i][0] > indent and not entries[i][1].startswith("- "):
                        c2 = entries[i][1]
                        if ":" in c2:
                            k, _, v = c2.partition(":")
                            if v.strip():
                                d[k.strip()] = _scalar(v)
                            elif i + 1 < len(entries) and entries[i + 1][0] > entries[i][0]:
                                child, i = _parse_value(entries, i + 1, entries[i + 1][0])
                                d[k.strip()] = child
                                continue
                        i += 1
                    result.append(d)
                    continue
                elif ":" in item:
                    result.append(_inline_map(item))
                else:
                    result.append(_scalar(item))
                i += 1
            elif ind > indent:
                i += 1
            else:
                break
        return result, i

    result = {}
    while i < len(entries):
        ind, c = entries[i]
        if ind != indent:
            break
        if ":" not in c:
            i += 1
            continue
        key, _, val = c.partition(":")
        key = key.strip()
        val = val.strip()
        if val:
            result[key] = _scalar(val)
            i += 1
        else:
            if i + 1 < len(entries) and entries[i + 1][0] > indent:
                child, i = _parse_value(entries, i + 1, entries[i + 1][0])
                result[key] = child
            else:
                result[key] = {}
                i += 1
    return result, i


def parse(text):
    entries = []
    for raw in text.splitlines():
        content = _strip_comment(raw)
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        entries.append((indent, content.strip()))
    if not entries:
        return {}
    value, _ = _parse_value(entries, 0, entries[0][0])
    return value if isinstance(value, dict) else {"_": value}
