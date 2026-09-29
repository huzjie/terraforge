"""Episodic memory: store and retrieve past scene observations."""
from ..utils.hashing import stable_float


class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self._store = []

    def add(self, scene_id, embedding):
        self._store.append({"scene_id": scene_id, "embedding": embedding})
        if len(self._store) > self.capacity:
            self._store = self._store[-self.capacity:]

    def retrieve(self, query_embedding, top_k=1):
        from ..spatial.geometry import distance
        ranked = sorted(self._store,
                        key=lambda e: distance(query_embedding, e["embedding"]))[:top_k]
        return [e["scene_id"] for e in ranked]
