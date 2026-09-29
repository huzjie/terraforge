"""Serve demo: launch the HTTP server briefly and hit /health."""
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.config import load_config
from terraforge.serving.http import serve

cfg = load_config(None)
cfg.serving.port = 8765

t = threading.Thread(target=serve, args=(cfg,), daemon=True)
t.start()
time.sleep(1.0)
try:
    with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=5) as r:
        print("health:", r.read().decode())
finally:
    print("done (server thread is daemon; will exit with process)")
