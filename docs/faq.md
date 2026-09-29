# FAQ

**Q: 需要 GPU 吗？**
A: 不需要。内核零依赖，mock/cpu 后端在 CPU 上即可跑通全流程。

**Q: 训练/评测结果可复现吗？**
A: 是。mock 后端完全确定性，同一种子结果一致。

**Q: 为什么准确率卡在 0.75 就上不去了？**
A: mock 的 skill 初始 0.5，准确率约 0.75；SFT 后 skill → 1.0，准确率 → 1.0。

**Q: 如何接真实模型？**
A: 用 openai / vllm / transformers 后端，或实现 `Backend` 接口。

**Q: 报 `unknown backend`？**
A: 确认 `backends/__init__.py` 已 `from . import <你的后端>` 以触发注册装饰器。
