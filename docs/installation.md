# 安装

## 零依赖（推荐）

`terraforge` 内核无第三方依赖，克隆即用：

```bash
git clone https://github.com/huzjie/terraforge
cd terraforge
python -m terraforge doctor
```

## 可编辑安装

```bash
pip install -e .
```

## 可选依赖

| 后端 / 能力 | 依赖 | 安装 |
|---|---|---|
| 配置 YAML 增强 | PyYAML | `pip install pyyaml` |
| openai 后端 | openai | `pip install openai` |
| vllm 后端 | vllm | `pip install vllm` |
| transformers 后端 | transformers | `pip install transformers` |

## 环境变量

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | openai 后端鉴权 |
| `OPENAI_BASE_URL` | 自定义 OpenAI 兼容网关 |
| `OPENAI_MODEL` | 模型名（默认 gpt-4o-mini） |
| `TF_BACKEND` | 默认后端名（可选，等价于 `--backend`） |
