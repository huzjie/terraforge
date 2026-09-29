# 推理服务

零依赖 stdlib HTTP 服务。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 兼容对话 |
| POST | `/v1/embeddings` | 嵌入 |
| POST | `/v1/spatial/predict` | 空间查询 |

## 示例

```bash
python -m terraforge serve --config configs/default.yaml

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/spatial/predict \
  -H "Content-Type: application/json" \
  -d '{"scene_id":"scene-001","query":"What is left of the table?"}'
```
