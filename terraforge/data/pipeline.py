"""End-to-end spatial-multimodal data production pipeline."""
import json
import os

from ..utils.logging import get_logger
from .synth import SynthEngine
from .render import render_scene, render_video
from .annotation import annotate_scene
from .dataset import SpatialDataset

log = get_logger("terraforge.data")


class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.synth = SynthEngine(seed=cfg.data.synth_seed)

    def run(self):
        dcfg = self.cfg.data
        n = dcfg.num_scenes
        scenes = self.synth.generate_many(n)
        images = []
        annotations = []
        for s in scenes:
            images.append(render_scene(s))
            annotations.append(annotate_scene(s))
        ds = SpatialDataset(scenes, images, annotations)
        train, val = ds.split(dcfg.train_ratio)
        out_dir = os.path.join(self.cfg.output_dir, dcfg.root)
        os.makedirs(out_dir, exist_ok=True)
        self._dump(train, val, out_dir)
        log.info(f"produced {n} scenes -> train={len(train)} val={len(val)} -> {out_dir}")
        return {"train": len(train), "val": len(val), "out_dir": out_dir}

    def _dump(self, train, val, out_dir):
        from .format import to_json
        to_json(train.to_records(), os.path.join(out_dir, "train.json"))
        to_json(val.to_records(), os.path.join(out_dir, "val.json"))
        meta = {"num_scenes": len(train) + len(val),
                "train": len(train), "val": len(val)}
        to_json(meta, os.path.join(out_dir, "meta.json"))


def run_pipeline(cfg):
    p = DataPipeline(cfg)
    result = p.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
