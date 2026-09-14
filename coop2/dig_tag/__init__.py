"""DIG-TAG: the Dynamic Interaction Graph coupled with the Task Activity
Graph, a collaboration interface for LLM multi-agent systems.

  base/       what both graphs are built on: the tool call, id minting, the helpers
  dig/        D on its own  --  `from dig_tag import dig`
  tag/        T on its own  --  `from dig_tag import tag`
  interface/  DIGTAG, the coupled interface DT = (D, T, Z, U), built on both
  views/      ways of looking at one recorded DT
  render/     drawing a record, and following one as it changes (matplotlib; nothing below imports it)
"""

__all__ = ["base", "dig", "tag", "interface", "views", "render"]
