"""Spatial dataset: scenes + images + annotations, with train/val split."""
from ..utils.hashing import stable_float


class SpatialDataset:
    def __init__(self, scenes, images=None, annotations=None):
        self.scenes = scenes
        self.images = images or [None] * len(scenes)
        self.annotations = annotations or [None] * len(scenes)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return {
            "scene": self.scenes[idx],
            "image": self.images[idx],
            "annotation": self.annotations[idx],
        }

    def split(self, ratio=0.9):
        n = len(self.scenes)
        k = int(n * ratio)
        from .dataset import SpatialDataset
        train = SpatialDataset(self.scenes[:k], self.images[:k], self.annotations[:k])
        val = SpatialDataset(self.scenes[k:], self.images[k:], self.annotations[k:])
        return train, val

    def to_records(self):
        recs = []
        for i in range(len(self)):
            a = self.annotations[i] or {}
            recs.append({"scene_id": self.scenes[i].id, **a})
        return recs
