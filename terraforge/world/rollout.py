"""Roll out the world model to produce a trajectory."""
from ..utils.hashing import stable_vector


def rollout(world_model, initial_state, actions, horizon=3):
    states = [list(initial_state)]
    for i in range(horizon):
        act = actions[i % len(actions)] if actions else None
        states.append(world_model.step(states[-1], act))
    return states
