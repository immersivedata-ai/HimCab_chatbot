"""
app/prompts — Prompt Management Layer
───────────────────────────────────────
Centralised prompt template for the Router node.

Usage:
    from app.prompts.builder import build_router_prompt
"""

from app.prompts.builder import build_router_prompt

__all__ = [
    "build_router_prompt",
]
