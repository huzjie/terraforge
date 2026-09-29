# 多模态融合

## 投影器（projector）

两层 MLP + LayerNorm，把视觉特征对齐到语言嵌入空间。

## 门控融合（GatedFusion）

```
gate = sigmoid(W [text ; visual])
out  = gate * visual + (1 - gate) * text
```

## 交叉注意力融合（CrossAttentionFusion）

视觉 token 作为 query，文本 token 作为 key/value，输出文本条件化的视觉表示。

## MoE 路由（MultimodalMoE / Router）

- `MultimodalMoE`：Top-K 专家 + 负载均衡辅助损失；
- `Router`：快慢路由，判断查询是否需要视觉 grounding。
