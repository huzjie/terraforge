# 配置

## 配置结构

```yaml
model:      # 模型超参
vision:     # 视觉编码器超参
train:      # 训练超参
data:       # 数据生产超参
serving:    # 服务超参
backend:    # 默认后端
output_dir: # 输出目录
seed:       # 随机种子
```

## 样例

- `configs/default.yaml`：默认（mock）
- `configs/mock.yaml`：最小 mock
- `configs/cpu.yaml`：cpu 后端
- `configs/full.yaml`：更大模型配置

## YAML 回退

无 PyYAML 时自动回退到内置极简 YAML 子集解析器（`utils/yamlish.py`）。
