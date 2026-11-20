"""Optional MCP client support: external tools and resources for agent workflows.

Configuration, authorization, records, and task glue import without the MCP
SDK. :mod:`matsim_agents.backends.mcp.session` needs the optional extra::

    pip install -e ".[mcp]"
"""

from matsim_agents.backends.mcp.authorization import AuthorizationDecision, MCPAuthorizer
from matsim_agents.backends.mcp.config import (
    MCPConfig,
    MCPServerConfig,
    load_mcp_config,
    parse_mcp_config,
)
from matsim_agents.backends.mcp.records import (
    MCPAuditLog,
    MCPCallRecord,
    MCPResourceInfo,
    MCPToolInfo,
)
from matsim_agents.backends.mcp.tasks import MCP_TASK_KINDS, execute_mcp_task, planner_catalog

__all__ = [
    "MCP_TASK_KINDS",
    "AuthorizationDecision",
    "MCPAuditLog",
    "MCPAuthorizer",
    "MCPCallRecord",
    "MCPConfig",
    "MCPResourceInfo",
    "MCPServerConfig",
    "MCPToolInfo",
    "execute_mcp_task",
    "load_mcp_config",
    "parse_mcp_config",
    "planner_catalog",
]
