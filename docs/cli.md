# CLI

```bash
python -m terraforge <command> [--config path] [--backend name] [--seed N]

commands:
  doctor   环境自检
  data     运行数据生产管线
  train    预训练（重建）
  sft      SFT（空间 QA）
  bench    九项基准评测
  serve    启动 HTTP 服务
  predict  单条空间查询
```

## 通用参数

| 参数 | 说明 |
|---|---|
| `--config` | 配置文件路径（YAML/JSON） |
| `--backend` | 后端覆盖（mock/cpu/openai/vllm/transformers） |
| `--seed` | 随机种子 |

`--config` 通过共享父解析器被所有子命令继承。
