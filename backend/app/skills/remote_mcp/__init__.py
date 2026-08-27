"""远程 MCP Skill 适配层。"""

from .registry import (
    build_remote_mcp_skill_definition,
    register_remote_mcp_skills,
    register_remote_mcp_skills_from_settings,
)

__all__ = [
    "build_remote_mcp_skill_definition",
    "register_remote_mcp_skills",
    "register_remote_mcp_skills_from_settings",
]
