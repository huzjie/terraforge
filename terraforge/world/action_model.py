"""Action model: predict the action that best matches a goal state."""
from ..utils.hashing import stable_vector


def candidate_actions(n=8, dim=8):
    return [("move_" + str(i), stable_vector(f"action:{i}", dim, -1, 1)) for i in range(n)]


def select_action(goal, candidates):
    from ..spatial.geometry import distance
    best, best_d = candidates[0][0], float("inf")
    for name, vec in candidates:
        d = distance(goal, vec)
        if d < best_d:
            best, best_d = name, d
    return best
