"""Environment self-check: report backends, optional deps, and core sanity."""
import importlib.util
import platform
import sys


def _has(module):
    return importlib.util.find_spec(module) is not None


def run_doctor():
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "optional_deps": {
            "numpy": _has("numpy"),
            "yaml": _has("yaml"),
            "requests": _has("requests"),
            "torch": _has("torch"),
            "transformers": _has("transformers"),
            "openai": _has("openai"),
        },
        "backends": [],
    }
    try:
        from .backends import list_backends
        report["backends"] = sorted(list_backends())
    except Exception as e:  # pragma: no cover
        report["backends_error"] = str(e)
    try:
        from .utils.hashing import stable_float
        report["hash_sanity"] = round(stable_float("doctor:sanity", 0, 100), 4)
    except Exception as e:  # pragma: no cover
        report["hash_error"] = str(e)
    return report


def print_doctor(report=None):
    report = report or run_doctor()
    print("terraforge doctor")
    print("=" * 50)
    print(f"python    : {report['python']}")
    print(f"platform  : {report['platform']}")
    print("optional deps:")
    for k, v in report.get("optional_deps", {}).items():
        print(f"  {k:<14}: {'yes' if v else 'no'}")
    print("backends  :", ", ".join(report.get("backends", [])) or "(none)")
    if "hash_sanity" in report:
        print(f"hash sanity: {report['hash_sanity']}")
    return report


if __name__ == "__main__":
    print_doctor()
