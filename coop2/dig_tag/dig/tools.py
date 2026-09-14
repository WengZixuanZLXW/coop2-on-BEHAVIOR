"""The DIG's own tool: send, the information tool that makes an event
available to other agents."""

from ..base import ToolCall, declare_tools

SEND_TOOL = "send"

declare_tools(ToolCall.Label.INFORMATION, [SEND_TOOL])
