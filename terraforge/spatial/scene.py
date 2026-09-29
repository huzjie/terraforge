"""Scene representation: objects with 3D position, size, and category."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Object3D:
    id: str
    category: str
    x: float
    y: float
    z: float
    w: float = 1.0
    h: float = 1.0
    d: float = 1.0

    def center(self):
        return (self.x, self.y, self.z)


@dataclass
class Scene:
    id: str
    objects: List[Object3D] = field(default_factory=list)

    def add(self, obj: Object3D):
        self.objects.append(obj)
        return obj

    def get(self, obj_id):
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None

    def __len__(self):
        return len(self.objects)
