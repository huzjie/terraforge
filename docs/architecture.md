# 架构总览

`terraforge` 采用**分层模块化**架构，各层通过清晰接口解耦，内核零依赖。

## 分层

```
┌─────────────────────────────────────────────┐
│  serving  (HTTP / OpenAI 兼容)               │
├─────────────────────────────────────────────┤
│  backends (mock / cpu / openai / vllm / tf)  │
├──────────────┬──────────────┬───────────────┤
│  benchmark   │  training    │  data pipeline │
├──────────────┴──────────────┴───────────────┤
│  multimodal (fusion / MoE / router)          │
├──────────────┬──────────────┬───────────────┤
│  vision      │  spatial     │  world         │
├──────────────┴──────────────┴───────────────┤
│  core (tensor / nn / attention)              │
└─────────────────────────────────────────────┘
```

## 关键设计

1. **core 层**：`tensor.py` 实现最小张量（向量/矩阵、matmul、softmax、layernorm）；`attention.py` 实现全注意力 / 因果 / 滑窗 / 稀疏；`modules.py` 提供 TransformerBlock、MLP、MoELayer。
2. **vision 层**：任意分辨率图像先 patch 化，再按 2D 网格加位置编码，经 ViT 后全局池化。
3. **spatial 层**：确定性场景生成 + 场景图 + 几何基元，是「物理世界」的符号底座。
4. **multimodal 层**：视觉特征经投影器对齐到文本空间，门控融合或交叉注意力融合，MoE 路由负责快慢分支。
5. **world 层**：潜状态动力学 + 下一状态预测 + 动作模型 + 推演 + 情景记忆。
6. **data 层**：自主式合成场景 → 渲染 → 自动标注 → 数据集切分导出，是「数据生产管线」的落点。
7. **backends 层**：统一接口 `predict_spatial / embed_scene / generate / train_step`，mock 确定性可训练。
8. **training 层**：预训练（重建）+ SFT（空间 QA），skill 单调趋 1。
9. **benchmark 层**：九项空间理解基准，复用 backends 接口。
