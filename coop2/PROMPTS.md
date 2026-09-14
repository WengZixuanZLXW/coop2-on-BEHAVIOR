# Where every prompt comes from

**Cut to about half on 2026-09-13 (user):** one statement per rule, no rule
stated twice across sections, no exhortation, no summary, no blank lines
inside a section; semantic completeness first, so the cut stopped at ~53 %
(system prompt, chars/4: tag 2950 -> 1563, individual ~2220 -> 1171). The
user side lost its per-robot repetition: the "Rooms in this house" line and
the "Step N/M | " prefix are stated once in section 6, and the route block's
trailing rule became a one-line legend (the rule is section 2's). What a
rewrite must keep is what the tests pin: `test_team_prompt_stubbed.SHARED_RULES`
and the mode tests' tokens.

The prompt an LLM team receives is assembled in one fixed order (user,
2026-09-13). Each row below is one section, the file and symbol that produce
it, and what feeds it. The two **digtag** slots -- section 5 and the task
observation inside section 6 -- are empty in every mode but `tag`, where the
task-graph brain ([`comm_topology/llm_tag.py`](comm_topology/llm_tag.py))
fills them.

## System prompt

| # | Section | Produced by | Fed by |
|---|---------|-------------|--------|
| 1 | 你的职责 -- YOUR ROLE (team controller duties, plan response format) | [`cognitive/agent/prompt_sections.py`](cognitive/agent/prompt_sections.py) `role_section` | team name, member ids, `max_actions` |
| 2 | 环境规则 -- ENVIRONMENT RULES (ticks, local view, TOO_FAR, unreachable, wait, exclusivity, routes, task ids navigable, rooms, id discipline, failure -> replan) | `prompt_sections.py` `ENVIRONMENT_RULES` | constant |
| 3 | 不同机器人能力描述 -- ROBOT CAPABILITIES (arm / base-lock / carrier / drone, who needs a carrier, lift rule) + this team's roster + the activity's lift table | `prompt_sections.py` `ROBOT_CAPABILITIES`, `robot_capabilities_section` | each robot's own `SymbolicObservation` flags (`is_carrier`, `base_locked_while_holding`, `lift_role`) and `lift_rules`, read in [`comm_topology/llm_team.py`](comm_topology/llm_team.py) `TeamBrain._robot_profiles` / `_lift_rules`; the flags come from the layout via [`behavior_env/team_config.py`](behavior_env/team_config.py) and [`behavior_env/world_state.py`](behavior_env/world_state.py); the lift table from the activity's `route.json` via [`behavior_env/route_spec.py`](behavior_env/route_spec.py) |
| 4 | 当前合作模式规则 -- COOPERATION MODE (individual / broadcast_chain / centralized leader / centralized follower / decentralized_messageboard / tag) | `prompt_sections.py` `COOPERATION_RULES`, `cooperation_section`; the notify rule `MESSAGEBOARD_NOTIFY_RULES` is appended only while `MessageboardTeamBrain.NOTIFY_TOOL_ENABLED` | `TeamBrain.COOPERATION_MODE`, set per brain class in `llm_team.py` (`TeamBrain`, `ChainTeamBrain`, `LeaderTeamBrain`, `FollowerTeamBrain`, `MessageboardTeamBrain`) and `llm_tag.py` (`TagTeamBrain`) |
| 5 | 预留的提示prompt -- BEST PRACTICE (what works, as against what the world enforces in 2-3 and the mode requires in 4), then the DIGTAG manual under `tag` | `prompt_sections.py` `BEST_PRACTICES` (first entry `HANDOFF_ORDER`), `reserved_section` | constant, plus `TeamBrain.reserved_system_prompt` (empty except `TagTeamBrain` = `llm_tag.TAG_MANUAL`). Six one-line practices, each stated only here: the handoff order (navigate the arm to the target first, then the carrier to the arm -- `unload_from` locks a base-locked arm the instant the cargo is in its hand; measured on `v4_s1_v4_lh`, arm-to-carrier satisfies both gates 14/25, carrier-to-arm 25/25); navigate_to an object before acting on it unless its listing shows no distance; one robot per object; plans of similar length; size a wait by the other's travel; read section 9 before replanning |


Under `broadcast_chain` the plan call returns `LLMChainPlanResponse`: the plans **and** a
`broadcast` sentence for the teams behind, written by the same reasoning, appended to the
closing instruction by `ChainTeamBrain._plan_closing` and sent by `after_plan`. It replaced
`_plan_summary`, a join of each robot's `plan.specification`, which with one cargo made every
team's broadcast identical and silent on who held it (measured, 2026-09-13 LH run). The
summary remains only as the fallback for a response with no sentence. The chain's
*interrupt* round says its piece the same way -- `LLMChainInterruptResponse`
adds `broadcast` to the per-robot decisions, `_interrupt_closing` asks for it,
`after_interrupt` sends it, and `_current_allocation()` is that path's fallback.
Section 4 also names this team's own neighbours -- `ChainTeamBrain._cooperation_extra`
appends UPSTREAM (every team ahead in the chain order, whose commitments section 7
carries) and DOWNSTREAM (`send_to`, who plan on the broadcast), with the head and
tail told they have none (user, 2026-09-14). The mode text above them could only say
"the teams ahead of you", leaving a team to infer from robot ids whether a sender
was ahead of it or behind. Upstream is NOT `wait_for`, which is only the immediate
predecessor this team blocks on.

Assembled by `prompt_sections.build_team_system_prompt`, called from
`TeamBrain._system_prompt`. [`cognitive/agent/prompts.py`](cognitive/agent/prompts.py)
`build_team_system_prompt` is a thin wrapper kept for callers and tests.

## User prompt

| # | Section | Produced by | Fed by |
|---|---------|-------------|--------|
| 6 | 全队的机器人 observation 和 available action, THE TEAM'S TASK once, and the digtag 任务 observation (under `tag`: the shared task graph, above the robots -- O_T before O_E, as in DIG-TAG) | `prompt_sections.py` `observations_section`; per-robot text from [`behavior_env/symbolic_view.py`](behavior_env/symbolic_view.py) `render_symbolic_view` (header with `(drone)`, `Rooms in this house`, room listing with `-> verbs [notes]`, relations, `Last action failed`) and `target_hints`; the task = one sentence of natural language from the activity's `description.txt` ([`behavior_env/task_description.py`](behavior_env/task_description.py), beside each `problem0.bddl`: what is carried from where to where) followed by `render_route_block` (routes, ids only, no rooms) or `render_goal_terms` (plain BDDL goals, no rooms), lifted out of each view by `TeamBrain._split_task` / `_team_task_block` | `coop_env._task_block`; the digtag slot is `TeamBrain.reserved_task_observation`, empty except under `tag`, where `TagTeamBrain._refresh_task_observation` fills it before every planning and interrupt prompt with `llm_tag.format_tag_observation`: each open task's current version (id, identity, goal, rule, state), its history (which team did what), its relations, the evidence on it, and the notify budget left |
| 7 | 当前收到的消息 -- MESSAGES RECEIVED NOW, heading worded per cooperation mode | `prompt_sections.py` `current_messages_section`, via `TeamBrain._current_messages_block` | what arrived since the last plan (`TeamBrain._heard`), or the interrupting messages in an interrupt round |
| 7 (decentralized_messageboard) | 公共留言板 -- MESSAGE BOARD, every team's posts oldest first, this team's labelled `You`, posts since it last planned marked `(new)`; a notify message, when the tool is on, follows under the mode's MESSAGES RECEIVED NOW heading | [`comm_topology/message_board.py`](comm_topology/message_board.py) `render_board`, called from `llm_team.py` `MessageboardTeamBrain._current_messages_block` | the run's one `MessageBoard` (`MessageboardTeamBrain.board`, shared by `create_llm_team_topology`); `_board_seen` marks what is new |
| 7 (tag) | MESSAGES RECEIVED NOW -- a team notified you: read the task graph in section 6 | `prompt_sections.py` `_CURRENT_HEADINGS["tag"]` via the default `_current_messages_block` | the notification (`metadata.type == "notify"`) that interrupted the team, or arrived while it was planning |
| 8 | 对话历史 -- CONVERSATION HISTORY, both directions, oldest first, minus section 7 | `llm_team.py` `TeamBrain._messages_block` | `AgentMemory` message events (`record_message_out`, broker deliveries) |
| 9 | 行动历史和失败原因 -- ACTION HISTORY AND FAILURES, per robot | `llm_team.py` `TeamBrain._action_history_block`; entries rendered by `prompts.py` `format_plan_history` | each robot's `plan` and `plan_history` |
| -- | closing instruction (N plans / resume-or-replan) | `llm_team.py` `_build_team_prompt`, `_build_interrupt_prompt` | -- |

Assembled by `prompt_sections.assemble_user_prompt`, called from
`TeamBrain._build_team_prompt` (planning) and `_build_interrupt_prompt`
(interrupt rounds). `GLOBAL OBJECTIVE` (`--goal`) sits at the top of section 6.

## Other prompts

| Prompt | Where | Notes |
|--------|-------|-------|
| Leader's assignment to the follower teams (centralized; its first message each round) | `llm_team.py` `LeaderTeamBrain._compose_assignment` | system = sections 1-5; user = sections 6-9 for the leader's own robots (task once, their listings, messages, history, action history) + the instruction to assign a leg and a robot kind to every follower team |
| Follower's response to the assignment (centralized) | `llm_team.py` `FollowerTeamBrain._compose_response` | system = sections 1-5; user = the assignment + `_status_report()`; accept, or say what the team does instead and why |
| Board post (decentralized_messageboard; written in the planning call itself) | `llm_client.py` `LLMMessageboardPlanResponse.board_post`; asked for by `MessageboardTeamBrain._plan_closing`; appended to the board by `_on_plan_response` | one or two sentences in the task's ids: which robot takes which cargo or leg this round, what is left to others |
| Notify request (the board's notify tool) | `llm_client.py` `NotifyRequest`, `LLMMessageboardNotifyPlanResponse.notify`; recorded by `message_board.py` `MessageBoard.notify` and delivered through `MessageBoard.deliverer`, which the factory sets to the sending brain's `MessageboardTeamBrain._deliver_notify` (one unit of the per-step notify budget, then `_say` with per-call recipients and `interrupts_execution`) | off until `MessageboardTeamBrain.NOTIFY_TOOL_ENABLED`; on, this mode is DIG-TAG's shared-message-board ablation |
| Task-graph part (tag) | `llm_client.py` `TagToolCall`, `TagPart`; `LLMTagPlanResponse` = the team plan + `tag_actions` + `notify`, `LLMTagInterruptResponse` = the interrupt decision + the same two; applied by `llm_tag.py` `TagTeamBrain._round`: actions on the shared graph in order (`tag_call_args`, a rejected one recorded with the graph's reason), then `notify` within the budget (`_notify`), then the plans or decisions go out; asked for by `_plan_closing` / `_interrupt_closing` | the vendored graph `coop2/dig_tag` (`TAGParallelInterface`); every round is recorded in `TagTeamBrain.tag_rounds` and saved as `tag_rounds.json` |
| Response schema (what a plan may contain: actions, targets, `wait` ticks, task specification) | [`cognitive/agent/llm_client.py`](cognitive/agent/llm_client.py) pydantic models `*Action`, `TaskSpecification`, and the team plan model | structured output, not prose |
| Single-robot prompts (legacy; not on the runtime path since one LLM per team) | `prompts.py` `ENV_DESCRIPTION`, `build_system_prompt`, `build_observation_prompt`, `build_message_prompt`; [`cognitive/agent/cognitive_agent.py`](cognitive/agent/cognitive_agent.py) `build_llm_messages`; per-robot topology roles `INDIVIDUAL_ROLE` in [`comm_topology/llm_individual.py`](comm_topology/llm_individual.py), `LEADER_ROLE` in [`comm_topology/llm_centralized.py`](comm_topology/llm_centralized.py), [`comm_topology/llm_broadcast_chain.py`](comm_topology/llm_broadcast_chain.py) | `test_team_prompt_stubbed` keeps `ENV_DESCRIPTION`'s rules in step with sections 2-3 |
| COOP2 repair channel | [`cognitive/agent/base_llm_agent.py`](cognitive/agent/base_llm_agent.py) `_repair_intention_system_prompt`; `prompts.py` `format_coop2_repair_context`, `format_coop2_repair_instruction` | only with `--coop2-repair` |

## Tests that pin the prompt

- [`feasibility_verify/test_team_prompt_stubbed.py`](../feasibility_verify/test_team_prompt_stubbed.py) -- shared rules present in both the team prompt and the legacy single-robot one; section order.
- [`feasibility_verify/test_llm_team_stubbed.py`](../feasibility_verify/test_llm_team_stubbed.py) -- one observation per robot, the task once above the robots, messages both ways, interrupt prompt shape.
- [`feasibility_verify/test_messageboard_stubbed.py`](../feasibility_verify/test_messageboard_stubbed.py) -- the board mode: section 4, the board as section 7 with `(new)` and `You`, `board_post` appended in the planning call and read by the next team, the notify tool behind its flag and, when on, delivered as an interrupt within the budget.
- [`feasibility_verify/test_tag_team_stubbed.py`](../feasibility_verify/test_tag_team_stubbed.py) -- the tag mode: sections 4 and 5, the graph in section 6 above the robots, actions landing before the notification goes out, the notified team's interrupt round, the budget, a notification during planning answered before the plans go out, and the run's `tag.json` / `tag_rounds.json` / `tag.pdf`.
- [`feasibility_verify/test_carrier_view_stubbed.py`](../feasibility_verify/test_carrier_view_stubbed.py), [`test_world_state_stubbed.py`](../feasibility_verify/test_world_state_stubbed.py), [`test_route_tracker_stubbed.py`](../feasibility_verify/test_route_tracker_stubbed.py) -- what the per-robot observation text contains.
