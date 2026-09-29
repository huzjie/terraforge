"""MCP tool definitions for terraforge (JSON schema, framework-agnostic)."""
TOOLS = [
    {
        "name": "spatial_predict",
        "description": "Answer a spatial reasoning query about a synthetic scene.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "scene identifier"},
                "query": {"type": "string", "description": "spatial question"},
            },
            "required": ["scene_id", "query"],
        },
    },
    {
        "name": "embed_scene",
        "description": "Embed a scene into a fixed-size vector.",
        "inputSchema": {
            "type": "object",
            "properties": {"scene_id": {"type": "string"}},
            "required": ["scene_id"],
        },
    },
]


def build_mcp_tools(backend="mock", cfg=None):
    from ..backends import build_backend
    be = build_backend(backend, cfg)
    return TOOLS, be
