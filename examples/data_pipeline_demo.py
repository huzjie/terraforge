"""Data pipeline demo: produce synthetic scenes + annotations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.config import load_config
from terraforge.data.pipeline import DataPipeline

cfg = load_config(None)
cfg.data.num_scenes = 20
p = DataPipeline(cfg)
r = p.run()
print("pipeline result:", r)
