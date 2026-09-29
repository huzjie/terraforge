# 九项空间理解基准

| # | 任务 | 度量 |
|---|---|---|
| 1 | 空间 QA | accuracy |
| 2 | 关系分类 | accuracy |
| 3 | 场景图边预测 | accuracy |
| 4 | 视角一致性 | accuracy |
| 5 | 遮挡推理 | accuracy |
| 6 | 深度估计 | 误差 |
| 7 | 位姿估计 | 误差 |
| 8 | 布局预测 | IoU |
| 9 | 导航（近邻检索） | accuracy |

## 运行

```bash
python -m terraforge bench --config configs/default.yaml
```

## 训练前后对比

学习型任务（1-5、9）随 skill 提升而改善；确定性任务（6-8）恒为高准确率。
