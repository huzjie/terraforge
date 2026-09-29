# 后端

统一接口：`predict_spatial / embed_scene / generate / train_step / skill`。

| 后端 | 说明 | 依赖 |
|---|---|---|
| **mock** | 确定性可训练，skill 单调趋 1，训练/评测可复现 | 无 |
| **cpu** | 用纯 Python 视觉编码器 + 空间推理器真实计算 | 无 |
| **openai** | 调用 OpenAI 兼容 API | openai / urllib |
| **vllm** | vLLM 推理 | vllm |
| **transformers** | HuggingFace Transformers | transformers |

## 切换

```bash
python -m terraforge bench --backend mock
python -m terraforge bench --backend cpu
```

## Mock 的确定性原理

对每个查询只抽一次 `u = stable_float(key)`，错误集 = `u < noise * (1 - skill)`。
skill 越高，阈值越小，错误越少，准确率单调上升。训练信号恒为正（对 +0.05 / 错 +0.03），skill 单调趋 1。
