"""Train demo: SFT raises skill 0.5 -> 1.0 and accuracy 0.75 -> 1.0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.config import load_config
from terraforge.training.sft import run_sft
from terraforge.training.pretrain import run_pretrain

cfg = load_config(None)
print("--- pretrain (reconstruction) ---")
run_pretrain(cfg, steps=200)
print("--- sft (spatial QA) ---")
run_sft(cfg, steps=200)
