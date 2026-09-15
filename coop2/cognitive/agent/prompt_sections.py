"""The team prompt, section by section, in the order the user fixed (2026-09-13).

System prompt
    1. ROLE                 -- <你的职责>          who you are, what you answer with
    2. ENVIRONMENT RULES    -- <环境规则>          what the world lets an action do
    3. ROBOT CAPABILITIES   -- <不同机器人能力描述>  arm / carrier / drone, and this team's robots
    4. COOPERATION MODE     -- <当前合作模式规则>    individual / broadcast_chain / centralized /
                               decentralized_messageboard
    5. BEST PRACTICE        -- <预留的提示prompt>    what works (handoff order last), then the digtag manual under tag

User prompt
    6. OBSERVATIONS         -- 全队的<机器人observation和available action>, plus the
                               reserved <digtag任务observation> slot
    7. CURRENT MESSAGES     -- <当前收到的消息>, worded for the cooperation mode; under
                               decentralized_messageboard this is the shared MESSAGE BOARD
    8. CONVERSATION HISTORY -- <对话历史>
    9. ACTION HISTORY       -- <行动历史和失败原因>

Each section is one function or one constant here, and ``llm_team`` assembles
them in this order and no other. Cut to about half on 2026-09-13 (user): one
statement per rule, no rule stated twice across sections, no exhortation, no
summary, no blank lines inside a section. The load-bearing tokens the tests
pin (``test_team_prompt_stubbed.SHARED_RULES`` and the mode tests) are the
list of what a rewrite must keep.
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
    "BEST_PRACTICES",
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
You command TEAM '{team_name}' ({n} robots: {robots}) as its TEAM CONTROLLER in a household task shared with other teams, each with its own controller.
You get all {n} robots' observations in one call and return one plan per robot.
- Plans start together and you are not asked again until every robot has finished; an early finisher holds position.
- Sections 8-9 hold this team's messages and each robot's last plans with their outcomes.
PLAN RESPONSE: one structured plan for EVERY robot above, tagged with its agent_id; one task and at most {max_actions} actions each; ids only from that robot's own section; `reasoning` says how the work was divided.
`task` is yours to name: the BDDL predicate (ontop, inside, open, closed, toggled_on, holding) when the plan makes one true, else an honest name -- `standby` for a robot you are deliberately keeping out of the way, naming what it waits for. Never name a goal the plan is not pursuing."""


# =============================================================================
# 2. ENVIRONMENT RULES  <环境规则>
# =============================================================================

ENVIRONMENT_RULES = """## 2. ENVIRONMENT RULES
Robots act through primitives that take many simulation steps (one tick = 1/30 s).
No robot can see: each gets a symbolic listing of the room it stands in, and only that room.
Costs (ticks): navigate_to 10 ticks per metre + ~10 to settle (5 m ~ 60), the bulk of a robot's time; grasp, place_on_top, load_onto, unload_from, open, close, toggle 1-10; wait(n) holds position for n+1, n <= 600; refused before starting (TOO_FAR, held by another robot, empty gripper) 1; failed part-way ~50.
- grasp, place, open, toggle need the robot within the object's distance threshold, else TOO_FAR.
  A room listing shows each object's verbs after "->", and a distance ONLY when out of range: then it is marked "unreachable" and offers navigate_to alone.
- One object per robot: grasp needs an empty gripper, place a full one.
  Objects are exclusive, also between your own robots: grasping what anyone holds fails with "held by <agent>" (OBJECT_CLAIMED) after the whole trip.
- Some tasks are a route: THE TEAM'S TASK lists the supports the cargo must rest on in order, marked done / NEXT.
  Only placing the cargo on the NEXT support counts; out of order counts nothing; progress is judged on where the cargo rests.
- Every id in THE TEAM'S TASK is a valid navigate_to target from ANY room, listed or not: navigate_to(<id>) drives the robot there through the house and its next listing shows that room.
  The cargo is where the task says (its start, or the last support marked done), not where the robots stand.
- "Rooms in this house" names every room; navigate_to(<room name>) drives to a free spot in it.
- Use only ids from a robot's own listing (they look like apple.n.01_1); never invent one or give a robot an id seen only under another robot.
  "blocked" means present but unavailable; the note says why.
- A failed action abandons that robot's plan; its failure reason is in section 9 next time."""


# =============================================================================
# 3. ROBOT CAPABILITIES  <不同机器人能力描述>
# =============================================================================

ROBOT_CAPABILITIES = """## 3. ROBOT CAPABILITIES
A robot's kind is on its own header ("(drone)") and on its line in others' listings ("[carrier, ...]", "[base locks while holding]", "[drone]").
ARM: grasp, place_on_top, release, open/close, toggle, load_onto, unload_from, navigate_to, wait.
An arm marked [base locks while holding] cannot navigate_to while loaded (BASE_LOCKED): it load_onto(<carrier>) what it holds, the carrier drives, and an arm unload_from(<carrier>) at the far end; both name the carrier and need the two robots within reach.
ONLY such an arm needs a carrier; every other robot that can grasp carries what it holds by itself (grasp, navigate_to the support, place_on_top).
CARRIER: no arm; navigate_to and wait only.
Cargo rides on its back: one line with the carrier in others' listings ("[carrier, carrying box.n.01_1]  -> unload_from"), an "On your back:" line in its own; it cannot be grasped there and only unload_from takes it off.
load_onto / unload_from are done TO a carrier by an arm.
DRONE: flies, base never locks, carries what it grasps by itself.
The drone cannot lift the notebook: grasp or unload_from on it fails with CANNOT_LIFT at any distance; only an arm can (the lift table below)."""


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
        parts.append("THIS TEAM'S ROBOTS:")
        for name, profile in robot_profiles.items():
            parts.append(f"  - {name}: {_describe_robot(profile)}")
    if lift_rules:
        parts.append("WHO MAY LIFT WHAT (this activity):")
        for synset, roles in lift_rules.items():
            parts.append(f"  - {synset}: {', '.join(roles)}")
    return "\n".join(parts)


# =============================================================================
# 4. COOPERATION MODE  <当前合作模式规则>
# =============================================================================

COOPERATION_RULES: Dict[str, str] = {
    "individual": """## 4. COOPERATION MODE: INDIVIDUAL
Every team plans for itself; no messages are sent or received.
You learn what other teams do only from what your robots see (their robots in a listing, an object one holds).
Do not wait for others; when another team holds the cargo, give your robots something else or wait.""",
    "broadcast_chain": """## 4. COOPERATION MODE: BROADCAST CHAIN
Teams speak in a fixed order, each planning on what the teams ahead of it said; their commitments are in section 7 and yours goes out in `broadcast`.
A message from a team ahead interrupts your robots: per robot, resume or replan.""",
    "centralized_leader": """## 4. COOPERATION MODE: CENTRALIZED -- YOU ARE THE LEADER
You speak first each round: your message to the follower teams IS the assignment -- which team takes which route leg or object, with which robot kind -- in the task's ids.
Followers answer in section 7 (accept, or what they do instead and why); then you plan your own robots knowing every commitment.""",
    "centralized_follower": """## 4. COOPERATION MODE: CENTRALIZED -- YOU ARE A FOLLOWER
Each round the leader's assignment arrives in section 7: your team's leg or object and the robot kind for it.
Answer in two or three sentences from your robots' actual state -- accept and name which robot does which part, or say what you do instead and why (cannot lift the cargo, leg already done, target held by another team) -- then plan consistently with your answer.""",
    "decentralized_messageboard": """## 4. COOPERATION MODE: DECENTRALIZED MESSAGE BOARD
Every team plans for itself as in individual mode: nobody waits, nobody is messaged, nothing you write interrupts anybody.
Shared: ONE MESSAGE BOARD.
Each plan call shows the whole board (section 7), oldest first, posts since you last planned marked (new); each plan you return carries one `board_post`, read by the other teams when they next plan.
- A cargo or leg another team posted it is taking is taken: send your robots elsewhere or have them wait.
- Post which of your robots takes which cargo or leg THIS round, in the task's ids, and what you leave to others: one or two sentences, no narration, no questions, no repeats.
- A post is a commitment, not a message: nobody is told, nobody answers, an executing team sees it only when it next plans.
  Plan as if the others act on what they posted.""",
    "tag": """## 4. COOPERATION MODE: SHARED TASK GRAPH
Every team plans for itself; the teams share ONE TASK GRAPH (manual in section 5), shown in section 6 above the robots each time you plan or are notified.
Every answer carries `tag_actions` (changes to the graph, applied in order BEFORE your plans or decisions take effect) and `notify` (teams to wake so they read the graph now).
- A task whose current version says another team is doing it is taken; what is not on the graph is invisible to the other teams.
- Notifying a team INTERRUPTS all its robots and costs it a decision; your notify budget until the next environment step is shown with the graph, and a notification beyond it is dropped.
  Leave `notify` empty unless a team's robots are doing what the graph now shows to be wrong or finished.
- The budget counts notifications, not teams: `notify` is a list, and one notification wakes every team you name for the same one unit.
  Name all of them at once rather than one per step.""",
}

#: The notify tool's rule, appended to section 4 only when a brain has
#: ``NOTIFY_TOOL_ENABLED`` set.
MESSAGEBOARD_NOTIFY_RULES = """
NOTIFY (a tool): a post waits for its readers; `notify` does not.
Set it to interrupt named teams with one or two sentences, only when what they are doing now is wrong given what you know (a cargo they head for that you hold, a leg you just completed).
It stops all their robots and costs them a decision; leave `notify` null otherwise."""


def cooperation_section(mode: str, extra: str = "") -> str:
    """Section 4 for @mode, plus @extra (the reserved notify rule) when given."""
    try:
        text = COOPERATION_RULES[mode]
    except KeyError:
        raise ValueError(f"unknown cooperation mode {mode!r}; have {sorted(COOPERATION_RULES)}") from None
    extra = (extra or "").rstrip()
    return f"{text}\n{extra}" if extra else text


# =============================================================================
# 5. BEST PRACTICE  <预留的提示prompt>  (+ the digtag manual under tag)
# =============================================================================

#: What works, as opposed to what the world enforces (sections 2-3) or the
#: mode requires (section 4). One statement each; each is stated only here.
#: The rest are what the runs of 2026-09-11..13 kept paying for when ignored
#: (CLAUDE.md: the sequencing problem, the idle holds, the repeated failed
#: plans).
#:
#: The last one, the handoff order, is the measured one. On v4_s1_v4_lh
#: (ridgeback + jackal, cabinet), 2026-09-13: the arm driving to the carrier
#: satisfies both gates in 14 of 25 sampled poses, the carrier driving to the
#: arm in 25 of 25. `unload_from` locks the arm's base the instant the cargo
#: is back in its hand, so an arm that drove to the carrier can be stranded
#: 2.48 m from the support -- a real failure at step 3101 of that run.
BEST_PRACTICES = (
    "When you acts on an object in plan, navigate_to it first so it is in reach; "
    "One robot per object; the others take another target or standby.",
    "Give the robots plans of similar length: the team is not asked again until the longest one finishes.",
    "Before replanning a robot, read its failure in section 9: a plan that failed for a reason that still holds fails again.",
    "To transfer an object with carrier: first navigate the arm to the target, then navigate the carrier to the arm."
)


def reserved_section(text: str = "") -> str:
    """Section 5: the standing best practices, plus digtag's manual when given.

    Standing rather than mode-specific: a base-locked arm and a carrier appear
    in every S1 layout, so individual, broadcast_chain and centralized all
    need the handoff order, and only `TagTeamBrain` ever fills @text.
    """
    lines = ["## 5. BEST PRACTICE"] + [f"- {practice}" for practice in BEST_PRACTICES]
    text = (text or "").strip()
    if text:
        # Numbered, like every other block in the prompt: the manual arrived as
        # a bare "DIGTAG:" label, the only unnumbered heading in a prompt whose
        # sections the rules refer to by number ("manual in section 5").
        lines.append(f"\n## 5.1 DIGTAG MANUAL\n{text}")
    return "\n".join(lines)


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
    house: str = "",
) -> str:
    """6. The team's task once, the house's rooms once (@house, lifted out of
    the per-robot views), the digtag task observation, then every robot's
    observation and available actions."""
    parts = [f"=== STEP {env_step} ==="]
    if preface:
        parts.append(preface)
    if goal_instruction:
        parts.append(f"GLOBAL OBJECTIVE: {goal_instruction}")
    parts.append(f"## 6. OBSERVATIONS -- TEAM {team_name} ({len(member_blocks)} robots)")
    if task_block:
        parts.append(task_block.strip("\n"))
    if house:
        parts.append(house.strip("\n"))
    reserved = (reserved_task_observation or "").strip()
    if reserved:
        parts.append(f"## 6.1 DIGTAG TASK OBSERVATION\n{reserved}")
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
                     (observations, current_messages, conversation_history, action_history, closing)
                     if part)
