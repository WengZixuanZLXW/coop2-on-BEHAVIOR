# Where every prompt comes from

The prompt an LLM team receives is assembled in one fixed order (user,
2026-09-13). Each row below is one section, the file and symbol that produce
it, and what feeds it. Sections marked *reserved* exist as empty slots for
**digtag**, the next cooperation system; filling them is digtag's job.

## System prompt

| # | Section | Produced by | Fed by |
|---|---------|-------------|--------|
| 1 | 你的职责 -- YOUR ROLE (team controller duties, plan response format) | [`cognitive/agent/prompt_sections.py`](cognitive/agent/prompt_sections.py) `role_section` | team name, member ids, `max_actions` |
| 2 | 环境规则 -- ENVIRONMENT RULES (ticks, local view, TOO_FAR, unreachable, wait, exclusivity, routes, task ids navigable, rooms, id discipline, failure -> replan) | `prompt_sections.py` `ENVIRONMENT_RULES` | constant |
| 3 | 不同机器人能力描述 -- ROBOT CAPABILITIES (arm / base-lock / carrier / drone, who needs a carrier, lift rule) + this team's roster + the activity's lift table | `prompt_sections.py` `ROBOT_CAPABILITIES`, `robot_capabilities_section` | each robot's own `SymbolicObservation` flags (`is_carrier`, `base_locked_while_holding`, `lift_role`) and `lift_rules`, read in [`comm_topology/llm_team.py`](comm_topology/llm_team.py) `TeamBrain._robot_profiles` / `_lift_rules`; the flags come from the layout via [`behavior_env/team_config.py`](behavior_env/team_config.py) and [`behavior_env/world_state.py`](behavior_env/world_state.py); the lift table from the activity's `route.json` via [`behavior_env/route_spec.py`](behavior_env/route_spec.py) |
| 4 | 当前合作模式规则 -- COOPERATION MODE (individual / broadcast_chain / centralized leader / centralized follower) | `prompt_sections.py` `COOPERATION_RULES`, `cooperation_section` | `TeamBrain.COOPERATION_MODE`, set per brain class in `llm_team.py` (`TeamBrain`, `ChainTeamBrain`, `LeaderTeamBrain`, `FollowerTeamBrain`) |
| 5 | 预留的提示prompt -- DIGTAG manual (*reserved*) | `prompt_sections.py` `reserved_section` | `TeamBrain.reserved_system_prompt` (empty) |

Assembled by `prompt_sections.build_team_system_prompt`, called from
`TeamBrain._system_prompt`. [`cognitive/agent/prompts.py`](cognitive/agent/prompts.py)
`build_team_system_prompt` is a thin wrapper kept for callers and tests.

## User prompt

| # | Section | Produced by | Fed by |
|---|---------|-------------|--------|
| 6 | 全队的机器人 observation 和 available action, THE TEAM'S TASK once, and the *reserved* digtag 任务 observation | `prompt_sections.py` `observations_section`; per-robot text from [`behavior_env/symbolic_view.py`](behavior_env/symbolic_view.py) `render_symbolic_view` (header with `(drone)`, `Rooms in this house`, room listing with `-> verbs [notes]`, relations, `Last action failed`) and `target_hints`; the task from `render_route_block` (routes) or `render_goal_terms` (plain BDDL goals), lifted out of each view by `TeamBrain._split_task` / `_team_task_block` | `coop_env.get_info` -> `LLMTeamAgent.observe`; the reserved slot is `TeamBrain.reserved_task_observation` (empty) |
| 7 | 当前收到的消息 -- MESSAGES RECEIVED NOW, heading worded per cooperation mode | `prompt_sections.py` `current_messages_section` | what arrived since the last plan (`TeamBrain._heard`), or the interrupting messages in an interrupt round |
| 8 | 对话历史 -- CONVERSATION HISTORY, both directions, oldest first, minus section 7 | `llm_team.py` `TeamBrain._messages_block` | `AgentMemory` message events (`record_message_out`, broker deliveries) |
| 9 | 行动历史和失败原因 -- ACTION HISTORY AND FAILURES, per robot | `llm_team.py` `TeamBrain._action_history_block`; entries rendered by `prompts.py` `format_plan_history` | each robot's `plan` and `plan_history` |
| -- | closing instruction (N plans / resume-or-replan) | `llm_team.py` `_build_team_prompt`, `_build_interrupt_prompt` | -- |

Assembled by `prompt_sections.assemble_user_prompt`, called from
`TeamBrain._build_team_prompt` (planning) and `_build_interrupt_prompt`
(interrupt rounds). `GLOBAL OBJECTIVE` (`--goal`) sits at the top of section 6.

## Other prompts

| Prompt | Where | Notes |
|--------|-------|-------|
| Follower's report to the leader (centralized) | `llm_team.py` `FollowerTeamBrain._compose_report` | system = sections 1-5; user = the leader's request + `_status_report()` |
| Response schema (what a plan may contain: actions, targets, `wait` ticks, task specification) | [`cognitive/agent/llm_client.py`](cognitive/agent/llm_client.py) pydantic models `*Action`, `TaskSpecification`, and the team plan model | structured output, not prose |
| Single-robot prompts (legacy; not on the runtime path since one LLM per team) | `prompts.py` `ENV_DESCRIPTION`, `build_system_prompt`, `build_observation_prompt`, `build_message_prompt`; [`cognitive/agent/cognitive_agent.py`](cognitive/agent/cognitive_agent.py) `build_llm_messages`; per-robot topology roles `INDIVIDUAL_ROLE` in [`comm_topology/llm_individual.py`](comm_topology/llm_individual.py), `LEADER_ROLE` in [`comm_topology/llm_centralized.py`](comm_topology/llm_centralized.py), [`comm_topology/llm_broadcast_chain.py`](comm_topology/llm_broadcast_chain.py) | `test_team_prompt_stubbed` keeps `ENV_DESCRIPTION`'s rules in step with sections 2-3 |
| COOP2 repair channel | [`cognitive/agent/base_llm_agent.py`](cognitive/agent/base_llm_agent.py) `_repair_intention_system_prompt`; `prompts.py` `format_coop2_repair_context`, `format_coop2_repair_instruction` | only with `--coop2-repair` |

## Tests that pin the prompt

- [`feasibility_verify/test_team_prompt_stubbed.py`](../feasibility_verify/test_team_prompt_stubbed.py) -- shared rules present in both the team prompt and the legacy single-robot one; section order.
- [`feasibility_verify/test_llm_team_stubbed.py`](../feasibility_verify/test_llm_team_stubbed.py) -- one observation per robot, the task once above the robots, messages both ways, interrupt prompt shape.
- [`feasibility_verify/test_carrier_view_stubbed.py`](../feasibility_verify/test_carrier_view_stubbed.py), [`test_world_state_stubbed.py`](../feasibility_verify/test_world_state_stubbed.py), [`test_route_tracker_stubbed.py`](../feasibility_verify/test_route_tracker_stubbed.py) -- what the per-robot observation text contains.
