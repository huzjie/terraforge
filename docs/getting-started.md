# 快速上手

## 前置条件

- Python 3.8+（无需任何第三方依赖）

## 三步跑通

```bash
git clone https://github.com/huzjie/terraforge
cd terraforge

# 1) 自检
python -m terraforge doctor

# 2) 数据 + 训练
python -m terraforge data --config configs/default.yaml
python -m terraforge train --config configs/default.yaml
python -m terraforge sft   --config configs/default.yaml

# 3) 评测 + 服务
python -m terraforge bench --config configs/default.yaml
python -m terraforge serve --config configs/default.yaml
```

## 示例脚本

```bash
python examples/quickstart.py
python examples/train_demo.py
python examples/data_pipeline_demo.py
python examples/benchmark_demo.py
python examples/spatial_demo.py
python examples/world_model_demo.py
```

## 首次运行的预期输出

- `doctor`：列出 5 个后端、可选依赖检测、哈希一致性数值；
- `data`：产出 N 个合成场景及标注 JSON；
- `train`：重建 skill 0.5 → 1.0，误差下降；
- `sft`：QA skill 0.5 → 1.0，准确率上升；
- `bench`：九项基准汇总 + AVG。
