"""World model demo: roll out a latent state over a horizon."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.world.world_model import WorldModel
from terraforge.world.rollout import rollout
from terraforge.utils.hashing import stable_vector

wm = WorldModel(dim=8, seed=0)
init = stable_vector("init", 8)
actions = ["move_0", "move_1", "move_2"]
traj = rollout(wm, init, actions, horizon=5)
print("trajectory length:", len(traj))
for i, s in enumerate(traj):
    print(f"  t={i}: [{', '.join(f'{v:.2f}' for v in s[:4])} ...]")
