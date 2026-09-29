# terraforge

面向物理世界的**空间多模态世界模型**训练与推理框架 —— 紫东太初 ZDTaichu5.0-9B 方向的工程化独立实现。

> 任意分辨率视觉编码 + 空间理解 + 多模态融合 + 世界模型 + 空间多模态数据生产管线 + 训练/推理服务，**内核零依赖**，配置即跑。

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/huzjie/terraforge/actions/workflows/ci.yml/badge.svg)](https://github.com/huzjie/terraforge/actions)

---

## 这是什么

紫东太初 ZDTaichu5.0-9B 开源了面向物理世界的通用多模态模型：约 9B 参数，支持单图 / 多图 / 长序列视频与任意分辨率输入，在九项国际空间理解基准中拿下 8 项通用组别第一，并**一并开放了整套空间多模态数据生产管线**。

`terraforge` 把这条技术路线工程化为一个**完整、可直接运行**的框架：它不需要 GPU、不需要 PyTorch、不需要任何第三方依赖，用一个确定性可训练的 Mock 后端就能把「数据生产 → 预训练 → SFT → 九项基准评测 → HTTP 推理服务」整条链路真实跑通，并观察到可量化的提升。

## 核心能力

| 模块 | 说明 |
|---|---|
| **任意分辨率视觉编码器** | patch 化 + 自适应分块 + 2D 位置编码 + ViT + 全局池化，支持单图/多图/长视频 |
| **空间理解** | 场景图、几何、深度、位姿、布局、遮挡、空间推理，确定性世界模型 |
| **多模态融合** | 投影器 + 门控融合 + 交叉注意力 + MoE 路由 + 快慢路由 |
| **世界模型** | 潜状态动力学 + 下一状态预测 + 动作模型 + 推演 + 情景记忆 |
| **数据生产管线** | 自主式合成场景 + 渲染 + 自动标注（关系/深度/位姿/布局）+ 数据集切分导出 |
| **训练** | 预训练（重建）+ SFT（空间 QA），确定性可训练 Mock（skill 单调趋 1） |
| **九项基准** | 空间 QA / 关系分类 / 场景图 / 视角一致性 / 遮挡 / 深度 / 位姿 / 布局 / 导航 |
| **推理服务** | stdlib HTTP（OpenAI 兼容 chat / embeddings / spatial predict） |
| **多后端** | mock / cpu / openai / vllm / transformers |
| **工程化** | CLI / Docker / K8s / Helm / CI / 集成（LangChain / MCP / Transformers） |

## 快速开始

```bash
# 1. 环境自检（零依赖即可运行）
python -m terraforge doctor

# 2. 生产空间多模态数据（合成场景 + 自动标注）
python -m terraforge data --config configs/default.yaml

# 3. 预训练（重建）+ SFT（空间 QA）
python -m terraforge train --config configs/default.yaml
python -m terraforge sft   --config configs/default.yaml

# 4. 九项空间理解基准评测
python -m terraforge bench --config configs/default.yaml

# 5. 启动 HTTP 推理服务
python -m terraforge serve --config configs/default.yaml
```

单条查询：

```bash
python -m terraforge predict --scene scene-001 --query "What is left of the table?"
```

## 训练效果（mock 后端，确定性可复现）

| 阶段 | 指标 | 初始 | 最终 |
|---|---|---|---|
| 预训练 | 重建 skill | 0.50 | 1.00 |
| SFT | 空间 QA skill | 0.50 | 1.00 |
| SFT | QA 准确率 | ~0.75 | ~1.00 |
| 九项基准 | AVG | ~0.85 | ~0.97 |

## 目录结构

```
terraforge/
├── terraforge/          # 框架主包
│   ├── core/            # 零依赖张量库 + 神经网络 + 注意力
│   ├── vision/          # 任意分辨率视觉编码器
│   ├── spatial/         # 空间理解（几何/深度/位姿/布局/场景图）
│   ├── multimodal/      # 多模态融合 + MoE 路由
│   ├── world/           # 世界模型 + 推演 + 记忆
│   ├── data/            # 空间多模态数据生产管线
│   ├── backends/        # 5 后端（mock/cpu/openai/vllm/transformers）
│   ├── training/        # 预训练 / SFT / 优化器 / 调度器
│   ├── benchmark/       # 九项空间理解基准
│   ├── serving/         # stdlib HTTP 服务
│   ├── utils/           # 极简 YAML / 确定性哈希 / 指标
│   └── integrations/    # LangChain / MCP / Transformers
├── examples/            # 7 个可运行示例
├── tests/               # 单元测试
├── configs/             # 配置样例
├── deploy/              # Docker / K8s / Helm
├── docs/                # 完整文档
└── .github/workflows/   # CI
```

## 为什么零依赖

核心内核只用 Python 标准库实现了一个最小张量库和神经网络层，因此：

- 在无法 `pip install` 的受限环境也能直接跑通全流程；
- 配置解析在无 PyYAML 时回退到内置的极简 YAML 子集解析器；
- Mock 后端确定性可训练，训练/评测结果完全可复现，便于教学与验证。

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/design-decisions.md](docs/design-decisions.md)。

## 文档

- [架构总览](docs/architecture.md) · [快速上手](docs/getting-started.md) · [安装](docs/installation.md)
- [视觉编码器](docs/vision-encoder.md) · [空间理解](docs/spatial-understanding.md) · [多模态融合](docs/multimodal-fusion.md)
- [世界模型](docs/world-model.md) · [数据管线](docs/data-pipeline.md) · [后端](docs/backends.md)
- [训练](docs/training.md) · [基准评测](docs/benchmark.md) · [推理服务](docs/serving.md)
- [CLI](docs/cli.md) · [配置](docs/config.md) · [集成](docs/integrations.md) · [部署](docs/deployment.md)
- [模型卡](docs/model-card.md) · [设计决策](docs/design-decisions.md) · [FAQ](docs/faq.md)

## License

MIT
