# 训练

## 预训练（重建）

`pretrain.run_pretrain`：在合成场景上训练重建 skill（对象数量预测），误差下降。

## SFT（空间 QA）

`sft.run_sft`：在空间 QA 对上训练 QA skill，准确率上升。

## 优化器 / 调度器

- `optimizer.SGD` / `Adam`；
- `scheduler.cosine` / `linear` / `constant`。

## 检查点

`checkpoint.save/load_checkpoint` 以 JSON 保存/恢复 skill 状态。

## 预期曲线

| 阶段 | 指标 | 初始 → 最终 |
|---|---|---|
| 预训练 | recon_skill | 0.50 → 1.00 |
| SFT | skill | 0.50 → 1.00 |
| SFT | accuracy | ~0.75 → ~1.00 |
