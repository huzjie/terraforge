# 空间多模态数据生产管线

对应 ZDTaichu5.0-9B 一并开放的「空间多模态数据生产管线」。

## 流程

```
SynthEngine（自主式合成场景）
   -> render_scene / render_video（渲染成图像/视频帧）
   -> annotate_scene（自动标注关系/深度/位姿/布局）
   -> qa_pairs（生成空间 QA 对）
   -> SpatialDataset（切分 train/val）
   -> to_json / to_jsonl（导出）
```

## 产物

- `outputs/data/<root>/train.json` / `val.json` / `meta.json`
- 每个场景含对象、关系、深度、位姿、布局标注

## 使用

```bash
python -m terraforge data --config configs/default.yaml
```
