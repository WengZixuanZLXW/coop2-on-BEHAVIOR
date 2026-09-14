# dig_tag Architecture

`dig_tag` implements the DIG-TAG collaboration interface: it records a
multi-agent run as `DT = (D, T, Z, U)` -- a Dynamic Interaction Graph of
activations and events, a Task Activity Graph of task versions, actions, and
evidence, an environment seam, and the three tool families that are the only
way anything changes. It is LLM- and environment-agnostic.

The two graphs are two packages that stand on their own, and DIG-TAG is
built on both:

```python
from dig_tag import dig                 # D on its own
from dig_tag import tag                 # T on its own
from dig_tag.interface import DIGTAG    # DT = (D, T, Z, U), built on both
```

```text
dig_tag/
+-- base/        What both graphs are built on. ToolCall and the tool vocabulary
|                (declare_tools), Minter (sequential ids), the leaf helpers. Stdlib only.
+-- dig/         D on its own.
|   +-- event.py        DIGEvent, the information node, with its two provenances
|   +-- activation.py   DIGActivation, one activation: its inputs and its calls
|   +-- tools.py        send
|   +-- graph.py        DIGGraph -- D = (E, H, L_D): the stores, the inboxes, the four funnels
|   +-- environment.py  the Z seam: Environment, NullEnvironment, RecordedEnvironment
|   +-- interface.py    DIGParallelInterface, the interface over D and Z: issue, open_activation, inject, the record
|   +-- agent.py        AsyncAgent, the asynchronously activated agent, on asyncio
|   +-- views/          L_D as a trace; ancestry; the calls on Z; the agent graphs
+-- tag/         T on its own.
|   +-- spec.py         TaskSpec, the goal and the rule
|   +-- version.py      Version, one exact task version
|   +-- action.py       TaskAction, a task call bound to its issuer; the task tools
|   +-- evidence.py     Evidence, an attachment to an exact version
|   +-- graph.py        TAGGraph -- T = (Q, A, Phi, L_T): the stores, the six transitions, attach
|   +-- interface.py    TAGParallelInterface, the interface over T, and what a tool hands back (its return policy)
|   +-- views/          L_T as a task graph; versions, frontier, closure, evidence; an identity's history and relations; lineage
+-- interface/   DIG-TAG, built on both.
|   +-- digtag.py       DIGTAG(DIGParallelInterface): the TAG's tools in the write path; a task call
|                       transforms T and what it hands back is the event D records
+-- views/       The DIG's and the TAG's views over one recorded DIGTAG.
+-- render/      Drawing a record, and following one as it changes. The one place
    |            matplotlib is imported (dig-tag[render]); nothing below imports it.
    +-- style.py, canvas.py   one style, panels measured in inches
    +-- timeline.py           where each moment goes: a DIG's activations, or a TAG's actions
    +-- record.py             the T / D / Z bands of a DIG-TAG, a DIG, or a TAG alone
    +-- live.py               LiveFigure, a window that follows a record while agents change it
```

Ownership is intentionally one-way:

- `base` imports nothing above the standard library.
- `dig` and `tag` import `base` and never each other: `from dig_tag import
  dig` loads nothing of the TAG, and `from dig_tag import tag` nothing of
  the DIG. The tool vocabulary is declared by the package that owns the
  tools (`send` by the DIG; the six task tools, `attach`, and `observe` by the TAG).
- `interface` imports both. D and T stay separate records: `DIGTAG`
  extends `DIGParallelInterface` with the task graph and takes the TAG's
  tools into the same write path, where a task call transforms T and what
  it hands back, by the tool's return policy, is recorded in D as an
  event.
- Views read a graph and own no state; no interface imports them.
  `render` draws through the views and imports every package below it;
  none of them imports it. Only `dig/agent.py` imports asyncio; only
  `dig/interface.py` and `tag/interface.py` take a lock; only `render` imports matplotlib.

The runtime that uses the task graph, the MA-Crafter environment and its
agent teams, lives beside this package in `ma_crafter/`; the root README
says how to run it. This package never imports it, an LLM client, or the
environment.
