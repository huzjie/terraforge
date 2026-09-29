# 集成

## LangChain

```python
from terraforge.integrations.langchain import TerraForgeLLM
llm = TerraForgeLLM(backend="mock")
llm("What is left of the table?")
```

## MCP

```python
from terraforge.integrations.mcp import build_mcp_tools
tools, backend = build_mcp_tools(backend="mock")
```

## Transformers

```python
from terraforge.integrations.transformers_opt import load_pretrained_vision
proc, model = load_pretrained_vision("google/vit-base-patch16-224")
```
