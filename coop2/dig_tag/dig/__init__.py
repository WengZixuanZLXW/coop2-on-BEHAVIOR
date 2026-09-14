"""D on its own: the Dynamic Interaction Graph.

  event.py        DIGEvent, the information node, with its two provenances
  activation.py   DIGActivation, one activation of an agent: its inputs and its calls
  tools.py        send, the DIG's own tool
  graph.py        DIGGraph, D = (E, H, L_D): the stores, the inboxes, the four funnels
  environment.py  the Z seam: Environment, NullEnvironment, RecordedEnvironment
  interface.py    DIGParallelInterface, the interface over D and Z: activations, inject, send, environment calls, the record
  agent.py        AsyncAgent, the asynchronously activated agent over the interface, on asyncio
  views/          ways of looking at one recorded D

`from dig_tag import dig` imports nothing of the TAG."""

from .activation import DIG_ERROR, DIG_METADATA, DIGActivation
from .agent import AsyncAgent, Behavior
from .environment import Environment, NullEnvironment, RecordedEnvironment
from .event import ENVIRONMENT_ORIGIN, EXTERNAL_ORIGIN, DIGEvent
from .graph import DIGGraph
from .interface import DIGParallelInterface, EventRef
from .tools import SEND_TOOL

__all__ = [
    "DIGEvent", "ENVIRONMENT_ORIGIN", "EXTERNAL_ORIGIN",
    "DIGActivation", "DIG_METADATA", "DIG_ERROR",
    "SEND_TOOL",
    "DIGGraph",
    "Environment", "NullEnvironment", "RecordedEnvironment",
    "DIGParallelInterface", "EventRef",
    "AsyncAgent", "Behavior",
]
