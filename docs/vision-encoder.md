# 任意分辨率视觉编码器

## 能力

- **单图**：`encode(img)` 返回全局视觉嵌入 + patch 网格；
- **多图**：`encode_multi(imgs)` 逐图编码后均值池化；
- **长视频**：`encode_video(frames)` 跨帧池化。

## 处理流程

```
image (C,H,W)
   -> patchify: 切分为 C*P*P 维 patch，自适应分块（限制 tile 数）
   -> PatchEmbedding: 线性投影到 hidden
   -> 2D 位置编码（注入网格坐标信号）
   -> ViT（多层多头自注意力）
   -> 全局池化（mean / max / cls）
```

## 任意分辨率

`adaptive_tiling()` 在总 tile 数超预算时对半降采样 tile 网格，保证任意长宽比/分辨率都能映射到固定预算的 token 序列。
