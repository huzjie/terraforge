"""Abstract backend interface."""
from abc import ABC, abstractmethod


class Backend(ABC):
    name = "base"

    @abstractmethod
    def predict_spatial(self, scene_id, query):
        """Answer a spatial query about a scene."""

    @abstractmethod
    def embed_scene(self, scene_id):
        """Return an embedding for a scene."""

    def generate(self, prompt, max_tokens=64, sample=False):
        raise NotImplementedError

    def train_step(self, scene_id, query):
        raise NotImplementedError

    @property
    def skill(self):
        return getattr(self, "_skill", 0.0)
