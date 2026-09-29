# API 参考

## 顶层

- `terraforge.load_config(path)` → `Config`
- `terraforge.backends.build_backend(name, cfg)` → `Backend`

## 视觉

- `VisionEncoder.encode(img)` → `(embedding, grid)`
- `patchify(img, patch_size)` → `(patches, grid)`

## 空间

- `build_scene(scene_id)` → `Scene`
- `SceneGraph(scene)` → 关系边
- `SpatialReasoner(scene).answer(question)` → str
- `world_truth(scene_id, query)` → 真值

## 数据

- `SynthEngine.generate_scene(id)` / `generate_many(n)`
- `render_scene(scene)` / `annotate_scene(scene)` / `qa_pairs(scene)`
- `DataPipeline(cfg).run()`

## 训练

- `run_pretrain(cfg)` / `run_sft(cfg)`

## 评测

- `BenchmarkSuite(backend, cfg).run()` → `{summaries, avg}`
