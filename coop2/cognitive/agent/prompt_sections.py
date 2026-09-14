"""The team prompt, section by section, in the order the user fixed (2026-09-13).

System prompt
    1. ROLE                 -- <你的职责>          who you are, what you answer with
    2. ENVIRONMENT RULES    -- <环境规则>          what the world lets an action do
    3. ROBOT CAPABILITIES   -- <不同机器人能力描述>  arm / carrier / drone, and this team's robots
    4. COOPERATION MODE     -- <当前合作模式规则>    individual / broadcast_chain / centralized /
                               decentralized_messageboard
    5. RESERVED             -- <预留的提示prompt>    the digtag manual, once digtag exists

User prompt
    6. OBSERVATIONS         -- 全队的<机器人observation和available action>, plus the
                               reserved <digtag任务observation> slot
    7. CURRENT MESSAGES     -- <当前收到的消息>, worded for the cooperation mode; under
                               decentralized_messageboard this is the shared MESSAGE BOARD
    8. CONVERSATION HISTORY -- <对话历史>
    9. ACTION HISTORY       -- <行动历史和失败原因>

Each section is one function or one constant here, and ``llm_team`` assembles
them in this order and no other. The texts themselves were lifted out of
``prompts.TEAM_ENV_DESCRIPTION`` and ``llm_team.TEAM_ROLE``; what is new is
the split (rules of the world vs. what a robot is vs. how teams talk), the
per-team robot roster, the cooperation-mode texts (none existed), and the
two digtag slots, which are empty until that system is written.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "ENVIRONMENT_RULES",
    "ROBOT_CAPABILITIES",
    "COOPERATION_RULES",
    "MESSAGEBOARD_NOTIFY_RULES",
    "role_section",
    "robot_capabilities_section",
    "cooperation_section",
    "reserved_section",
    "build_team_system_prompt",
    "observations_section",
    "current_messages_section",
    "conversation_history_section",
    "action_history_section",
    "assemble_user_prompt",
]


# =============================================================================
# 1. ROLE  <你的职责>
# =============================================================================

def role_section(team_name: str, member_ids: Sequence[str], max_actions: int = 6) -> str:
    robots = ", ".join(member_ids)
    n = len(member_ids)
    return f"""## 1. YOUR ROLE
You command TEAM '{team_name}' -- {n} robots ({robots}) -- in a multi-agent
cooperative household task, alongside other teams doing the same. You are the
TEAM CONTROLLER: you are given all {n} robots' observations in one call and
answer with one plan per robot, in that same call.

- Divide the work. Two robots sent to the same object waste one of them: the
  loser burns the whole trip and its grasp fails with OBJECT_CLAIMED.
- Your robots start their plans together and you are not asked again until
  every one of them has finished. A robot that finishes early holds position
  and does nothing useful, so plans of wildly different lengths waste the
  short ones.
- Other teams are driven by their own controllers, not by you. You coordinate
  with them only through the messages quoted in this prompt.
- At the end of the prompt are each robot's last few plans, with their
  actions and how each ended, and the messages this team has sent and
  received. A plan that failed for a reason that still holds will fail the
  same way again, and a question you already asked has its answer in the
  record.

PLAN RESPONSE:
- Return a structured plan for EVERY robot listed above, one each, tagged with
  that robot's agent_id.
- Give each robot one task and at most {max_actions} actions; 3-6 short
  executable actions are usually enough.
- Use exact item_id values from that robot's own section of the prompt.
- Say in the team-level `reasoning` how you divided the work between them."""


# =============================================================================
# 2. ENVIRONMENT RULES  <环境规则>
# =============================================================================

ENVIRONMENT_RULES = """## 2. ENVIRONMENT RULES
Your robots act through high-level primitives, not joint commands. Each one
takes many simulation steps: driving across a room costs 20 ticks per metre, so
distance is the main cost you control. None of them can see -- each is given a
symbolic description of the room it is standing in, and only that room, and
those descriptions are listed below one robot at a time.

What each action costs, in ticks (one tick is one simulation step, 1/30 s):
- navigate_to: 20 x the metres driven, plus about 10 to settle on arrival.
  Five metres is roughly 110 ticks. This is where almost all of a robot's
  time goes, so when you make one robot wait for another, size the wait by
  the other's travel, not by its grasp.
- grasp, place_on_top, load_onto, unload_from, open, close, toggle: nearly
  instant -- 1 to about 10 ticks.
- wait(n): n ticks plus one to issue it, n at most 600.
- An action refused before it starts (TOO_FAR, held by another robot, nothing
  in the gripper) costs 1 tick. One that fails part-way costs about 50.

Rules that decide whether an action succeeds:
- To grasp, place, open or toggle an object a robot must be closer to it than
  that object's own distance threshold; beyond it the action fails with TOO_FAR.
  Each object in a robot's room listing carries what that robot can do to it
  after "->", and a distance in metres ONLY when it is out of range -- no
  distance means that robot is already close enough to act on it now.
- An out-of-range object is marked "unreachable" and offers navigate_to and
  NOTHING else. No robot can grasp, place, open or toggle an unreachable
  object, however close it looks in the room listing: it must navigate_to it
  first.
- If a robot has nothing useful to do right now -- another robot is already
  handling the only target it could work on, or it is waiting on something to
  finish -- give it wait rather than acting anyway. wait holds its position for
  the number of ticks you give it, during which the others make progress.
- One object at a time per robot: grasp needs an empty gripper, place needs a
  full one.
- Objects are exclusive, including between your own robots. If anyone is
  holding something, a grasp for it fails with "held by <agent>". Sending two
  robots after one target costs the loser the whole trip for nothing.
- Some tasks are a route: THE TEAM'S TASK lists supports the cargo must rest on
  in order, marked done / NEXT. Only the NEXT support counts when the cargo is
  placed on it; a support reached out of order does not count and costs
  nothing -- progress resumes when NEXT holds. Progress is judged on where the
  cargo rests, not on what a robot intended.
- Every id in THE TEAM'S TASK is a valid navigate_to target for any robot from
  ANY room, even when it is not in that robot's listing: navigate_to(<that id>)
  drives it there, through the house, and its next listing shows that room.
  That is how a robot changes rooms. The cargo is where the task says it
  starts (or on the last support marked done) -- send a robot there; it will
  not be found where the robots stand.
- The line "Rooms in this house" names every room. navigate_to(<room name>)
  drives a robot to a free spot inside that room, so it can reach a room whose
  objects it cannot name yet, or go and look for something.
- Refer to objects only by the ids in that robot's own room listing, which are
  the objects in the room it is standing in. They look like apple.n.01_1. Never
  invent one, and never give one robot an id that appeared only under another
  robot -- it is in a room that robot cannot see.
- An entry marked "blocked" is in that room but currently unavailable -- the
  bracketed note says why.

When an action fails, that robot's plan is abandoned. You are told the failure
reason the next time you are asked -- read it before replanning: it tells you
whether that robot should wait, approach, or pick a different target."""


# =============================================================================
# 3. ROBOT CAPABILITIES  <不同机器人能力描述>
# =============================================================================

ROBOT_CAPABILITIES = """## 3. ROBOT CAPABILITIES
Three kinds of robot. A robot's kind is on its own header ("(drone)") and on
its line in a teammate's listing ("[carrier, ...]", "[base locks while
holding]", "[drone]").

ARM -- a robot with a gripper: grasp, place_on_top, release, open/close,
toggle, load_onto, unload_from, navigate_to, wait.
- Some arms' bases LOCK while the gripper is loaded, and navigate_to then fails
  with BASE_LOCKED; such a robot cannot deliver what it picks up. Have it load
  the object onto a carrier robot with load_onto, let the carrier drive, and
  have an arm take it back at the far end with unload_from. Both actions name
  the *carrier* as their target and need the two robots within reach of each
  other. This is a division of labour, not a detour: it is why a task can need
  more than one of your robots.
- ONLY a robot marked [base locks while holding] needs a carrier. Any other
  robot that can grasp -- a drone, an unmarked arm -- carries what it holds by
  itself: grasp, navigate_to the support, place_on_top, done. A drone that can
  lift the cargo needs nobody: routing its cargo through a carrier and back is
  a detour that costs two extra actions and a round of waiting for nothing.
  Use the carrier for the base-locked robot's legs, and only for those.

CARRIER -- a robot with no arm: cargo rides on its back.
- A carrier and what it is carrying are ONE line in that robot's room listing:

    - agent_1  (teammate)  [carrier, carrying box.n.01_1]  -> unload_from

  The cargo has no line of its own and cannot be grasped where it sits.
  unload_from(agent_1) takes it off that carrier's back and into the acting
  robot's hand; load_onto(agent_1) puts what that robot holds onto it. Both
  name the carrier, both need the two robots within reach of each other, and
  unload_from is the only way to get cargo back.
- **A carrier has no arm**, so give it navigate_to and wait and nothing else:
  no grasp, no place, no release, no load_onto or unload_from -- those are done
  *to* it by a robot that does have a hand. Its own section lists its load on
  an "On your back:" line, and it cannot put that down itself. Its job is to
  drive to where an arm is waiting, which is the whole reason a task needs two
  of them.

DRONE -- flies, holds its load under a suction mount, and its base never locks:
it carries what it grasps by itself.
- What a drone may lift is limited. The drone cannot lift the notebook. A
  robot tagged [drone] can grasp and carry the die (die.n.01_1) but NOT the
  notebook (notebook.n.01_1): grasp or unload_from on it fails with
  CANNOT_LIFT, and getting closer does not change that. Only a robot with an
  arm can lift the notebook, so give that leg to an arm; this division of
  labour is what the task is built around."""


def _describe_robot(profile: Dict[str, object]) -> str:
    """One phrase from an observation's flags: what this robot is."""
    if profile.get("is_carrier"):
        return "CARRIER (no arm; navigate_to and wait only)"
    role = profile.get("lift_role") or "arm"
    if role == "drone":
        return "DRONE (flies, carries what it holds; see the lift table)"
    if profile.get("base_locked_while_holding"):
        return "ARM, base LOCKS while holding (needs a carrier to move cargo)"
    return "ARM (carries what it holds by itself)"


def robot_capabilities_section(
    robot_profiles: Optional[Dict[str, Dict[str, object]]] = None,
    lift_rules: Optional[Dict[str, Iterable[str]]] = None,
) -> str:
    """The generic text plus, when known, this team's roster and the lift table.

    @robot_profiles: ``{robot name: {is_carrier, base_locked_while_holding,
    lift_role}}`` read off each member's own observation. @lift_rules: the
    activity's ``lift`` table (cargo synset -> roles that may lift it).
    """
    parts = [ROBOT_CAPABILITIES]
    if robot_profiles:
        parts.append("\nTHIS TEAM'S ROBOTS:")
        for name, profile in robot_profiles.items():
            parts.append(f"  - {name}: {_describe_robot(profile)}")
    if lift_rules:
        parts.append("\nWHO MAY LIFT WHAT (this activity):")
        for synset, roles in lift_rules.items():
            parts.append(f"  - {synset}: {', '.join(roles)}")
    return "\n".join(parts)


# =============================================================================
# 4. COOPERATION MODE  <当前合作模式规则>
# =============================================================================

COOPERATION_RULES: Dict[str, str] = {
    "individual": """## 4. COOPERATION MODE: INDIVIDUAL
Every team plans for itself. No team sends or receives messages; you learn
what the other teams are doing only from what your robots see (a teammate of
another team in the listing, an object held by one of its robots). Do not
wait for others to speak. When another team already holds the cargo, give
your robots something else or wait.""",
    "broadcast_chain": """## 4. COOPERATION MODE: BROADCAST CHAIN
Teams speak in a fixed order, one after another. When it is your turn you
have already been told what every team ahead of you committed to this round
(section 7); plan around it, do not duplicate it, and your own allocation is
then broadcast to the teams after you. A message from a team ahead interrupts
your robots: decide per robot whether its plan still stands (resume) or must
change (replan).""",
    "centralized_leader": """## 4. COOPERATION MODE: CENTRALIZED -- YOU ARE THE LEADER
Each round you speak first: your message to the follower teams IS the
assignment -- which team takes which route leg or object, and which robot
kind does it. The followers answer it (section 7): each accepts, or says what
it will do instead and why. Then you plan for your own robots knowing what
every team committed to. Make the assignment concrete, in the ids and node
names of the task; a vague one gets a vague answer and duplicated work.""",
    "centralized_follower": """## 4. COOPERATION MODE: CENTRALIZED -- YOU ARE A FOLLOWER
Each round the leader team sends you an assignment (section 7): the leg or
object your team should take and which robot kind should do it. You answer it
in two or three sentences, grounded in your robots' actual state -- accept it
and say which robot does which part, or say what you will do instead and why
(a robot that cannot lift the cargo, a leg already done, a target another
team holds). Then plan for your robots consistently with what you answered.""",
    "decentralized_messageboard": """## 4. COOPERATION MODE: DECENTRALIZED MESSAGE BOARD
Every team plans for itself, as in individual mode: no team waits for
another, no team sends anyone a message, and nothing you write interrupts
anybody. What you share is ONE MESSAGE BOARD that every team reads and
writes. Each time you plan you are shown the whole board (section 7), oldest
post first, with the posts that appeared since you last planned marked
(new); and every plan you return carries one `board_post` of your own, which
the other teams read the next time THEY plan.

- Read the board before dividing the work. A cargo or task leg another team
  has posted it is taking is taken: send your robots elsewhere, or have them
  wait, exactly as you would if you saw that team's robot holding it.
- Write a post that lets the others do the same: which of your robots goes
  for which cargo or leg THIS round, in the task's ids, and what you are
  leaving for others. One or two sentences. Do not narrate, ask questions, or
  repeat what you posted before -- the board keeps the record.
- A post is a commitment, not a message: nobody is told about it, nobody
  answers it, and a team already executing will not see it until it next
  plans. Plan as if the others act on what they posted, not on what you
  post.""",
    "tag": """## 4. COOPERATION MODE: SHARED TASK GRAPH
Every team plans for itself, and the teams coordinate through ONE TASK GRAPH
that every team reads and writes; its manual is section 5. Each time you plan,
and each time another team notifies you, you are shown the graph in section 6
-- the open tasks, their current versions, who did what to each, and the
evidence attached -- above your robots' observations. Every answer you return
carries two more fields: `tag_actions`, changes to the graph, applied in order
BEFORE your plans or decisions take effect; and `notify`, the names of teams
to wake so they read the graph now.

- Read the graph before dividing the work. A task whose current version says
  another team is doing it is taken: open what nobody is doing, update the
  state of what your robots are doing, attach what you learned, close what is
  done. Name versions by their ids (q3) and tasks by their identities (k1).
- Notifying a team INTERRUPTS all its robots and costs it a decision. Your
  notify budget until the next environment step is shown with the graph; a
  notification beyond it is dropped. Leave `notify` empty unless a team's
  robots are doing something the graph now shows to be wrong or finished.
- Nobody reads what you did not write down. The graph is the only memory the
  teams share across rounds; a plan that is not on it is invisible to them.""",
}

#: Reserved for the notify tool: appended to section 4 only when a brain has
#: ``NOTIFY_TOOL_ENABLED`` set. It exists now so the rule is written next to
#: the board's, not invented later.
MESSAGEBOARD_NOTIFY_RULES = """
NOTIFY (a tool): a post waits for its readers; `notify` does not. Set the
`notify` field to interrupt named teams with one or two sentences, and only
when what they are doing right now is wrong because of what you know -- a
cargo they are heading for that you already hold, a leg you have just
completed. Interrupting a team stops all its robots and costs it a decision;
leave `notify` null otherwise."""


def cooperation_section(mode: str, extra: str = "") -> str:
    """Section 4 for @mode, plus @extra (the reserved notify rule) when given."""
    try:
        text = COOPERATION_RULES[mode]
    except KeyError:
        raise ValueError(f"unknown cooperation mode {mode!r}; have {sorted(COOPERATION_RULES)}") from None
    extra = (extra or "").rstrip()
    return f"{text}\n{extra}" if extra else text


# =============================================================================
# 5. RESERVED  <预留的提示prompt>  (the digtag manual)
# =============================================================================

def reserved_section(text: str = "") -> str:
    """The slot for digtag's manual. Empty text -> no section, no tokens."""
    text = (text or "").strip()
    return f"## 5. DIGTAG\n{text}" if text else ""


def build_team_system_prompt(
    team_name: str,
    member_ids: Sequence[str],
    max_actions: int = 6,
    cooperation_mode: str = "individual",
    robot_profiles: Optional[Dict[str, Dict[str, object]]] = None,
    lift_rules: Optional[Dict[str, Iterable[str]]] = None,
    reserved: str = "",
    cooperation_extra: str = "",
) -> str:
    sections = [
        role_section(team_name, member_ids, max_actions),
        ENVIRONMENT_RULES,
        robot_capabilities_section(robot_profiles, lift_rules),
        cooperation_section(cooperation_mode, cooperation_extra),
        reserved_section(reserved),
    ]
    return "\n\n".join(s for s in sections if s)


# =============================================================================
# 6-9. USER PROMPT
# =============================================================================

def observations_section(
    env_step: int,
    team_name: str,
    member_blocks: Sequence[str],
    task_block: str = "",
    goal_instruction: str = "",
    reserved_task_observation: str = "",
    preface: str = "",
) -> str:
    """6. Every robot's observation and available actions, the team's task
    once, and the reserved digtag task observation."""
    parts = [f"=== STEP {env_step} ==="]
    if preface:
        parts.append(preface)
    if goal_instruction:
        parts.append(f"\nGLOBAL OBJECTIVE: {goal_instruction}")
    parts.append(f"\n## 6. OBSERVATIONS -- TEAM {team_name} ({len(member_blocks)} robots)")
    if task_block:
        parts.append(task_block)
    reserved = (reserved_task_observation or "").strip()
    if reserved:
        parts.append(f"\nDIGTAG TASK OBSERVATION:\n{reserved}")
    for block in member_blocks:
        parts.append("\n" + block)
    return "\n".join(parts)


_CURRENT_HEADINGS = {
    "individual": "## 7. MESSAGES RECEIVED NOW",
    "broadcast_chain": "## 7. MESSAGES RECEIVED NOW -- what the teams ahead of you committed to",
    "centralized_leader": "## 7. MESSAGES RECEIVED NOW -- your followers' responses to your assignment",
    "centralized_follower": "## 7. MESSAGES RECEIVED NOW -- your leader's assignment",
    # Reserved: the board mode has no direct messages until the notify tool
    # delivers one; this is the heading such a message would arrive under.
    "decentralized_messageboard": "## 7. MESSAGES RECEIVED NOW -- a team interrupted you with notify",
    "tag": "## 7. MESSAGES RECEIVED NOW -- a team notified you: read the task graph in section 6",
}


def current_messages_section(mode: str, messages: Sequence[Dict], render) -> str:
    """7. What arrived since the team last planned, worded for the mode.
    @render turns one message dict into one line."""
    if not messages:
        return ""
    lines = [_CURRENT_HEADINGS.get(mode, _CURRENT_HEADINGS["individual"])]
    lines.extend(f"  {render(m)}" for m in messages)
    return "\n".join(lines)


def conversation_history_section(history_block: str) -> str:
    """8. Everything this team has said and been told, oldest first."""
    return history_block or ""


def action_history_section(per_robot: Sequence[str]) -> str:
    """9. Each robot's last plans, with actions and why they ended."""
    blocks = [b for b in per_robot if b]
    if not blocks:
        return ""
    return "## 9. ACTION HISTORY AND FAILURES\n" + "\n\n".join(blocks)


def assemble_user_prompt(
    observations: str,
    current_messages: str,
    conversation_history: str,
    action_history: str,
    closing: str,
) -> str:
    return "\n".join(part for part in
                     (observations, "\n" + current_messages if current_messages else "",
                      "\n" + conversation_history if conversation_history else "",
                      "\n" + action_history if action_history else "",
                      "\n" + closing) if part)
