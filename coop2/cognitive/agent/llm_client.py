"""
LLM Client and Response Models for MA-Crafter.

This module provides:
- Pydantic models for structured LLM responses (plans, messages, interrupts)
- LLMClient wrapper for Azure OpenAI and OpenAI-compatible API calls
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Literal
import re
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI, RateLimitError, APIError
from pydantic import BaseModel, Field


# Load local credentials without overriding variables exported by the caller.
# An environment-specific file takes precedence over the repository-level file.
_ENVIRONMENT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ENVIRONMENT_ROOT / ".env", override=False)
load_dotenv(_ENVIRONMENT_ROOT.parent / ".env", override=False)


# ============================================================================
# Enums for constrained values
# ============================================================================

# ============================================================================
# Structured output vocabulary -- BEHAVIOR-1K
# ============================================================================
# This is one third of M5's "three-piece" change (PORTING_PLAN 5): the Pydantic
# schema, the L2 controllers and action_outcome.effects have to agree. Structured
# output *constrains* the model, so whatever is not expressible here is not
# emittable at all -- and anything expressible but unknown to L2 becomes a
# wasted decision. The verbs below are exactly
# coop2.cognitive.action.behavior_action.BEHAVIOR_ACTION_TO_PRIMITIVE plus the
# two communication actions.
#
# Targets are free-form strings, deliberately not enums: the legal set is
# scene-dependent and is handed to the model each turn as `target_hints`. An
# enum would freeze one scene's objects into the schema. They are BDDL instance
# ids (`apple.n.01_1`), which is also how the goal predicates are written.

_TARGET = "Entity id exactly as shown in Current Reachable Targets, e.g. 'apple.n.01_1'. Never invent one."


class NavigateToAction(BaseModel):
    """Drive the base to a standable pose near the target."""

    action_type: Literal["navigate_to"] = "navigate_to"
    target: str = Field(description=_TARGET)


class GraspAction(BaseModel):
    """Pick the target up. Requires being within reach and both hands empty."""

    action_type: Literal["grasp"] = "grasp"
    target: str = Field(description=_TARGET)


class PlaceOnTopAction(BaseModel):
    """Put what you are holding on top of the target."""

    action_type: Literal["place_on_top"] = "place_on_top"
    target: str = Field(description=_TARGET)


class PlaceInsideAction(BaseModel):
    """Put what you are holding inside the target."""

    action_type: Literal["place_inside"] = "place_inside"
    target: str = Field(description=_TARGET)


class ReleaseAction(BaseModel):
    """Drop what you are holding where you stand."""

    action_type: Literal["release"] = "release"


class OpenAction(BaseModel):
    action_type: Literal["open"] = "open"
    target: str = Field(description=_TARGET)


class CloseAction(BaseModel):
    action_type: Literal["close"] = "close"
    target: str = Field(description=_TARGET)


class ToggleOnAction(BaseModel):
    action_type: Literal["toggle_on"] = "toggle_on"
    target: str = Field(description=_TARGET)


class ToggleOffAction(BaseModel):
    action_type: Literal["toggle_off"] = "toggle_off"
    target: str = Field(description=_TARGET)


class LoadOntoAction(BaseModel):
    """Put what you are holding onto a carrier robot's back."""

    action_type: Literal["load_onto"] = "load_onto"
    target: str = Field(..., description="Carrier robot to load onto, e.g. agent_1")


class UnloadFromAction(BaseModel):
    """Take what a carrier robot is carrying into your gripper."""

    action_type: Literal["unload_from"] = "unload_from"
    target: str = Field(..., description="Carrier robot to unload from, e.g. agent_1")


class WaitAction(BaseModel):
    """Hold position, letting time pass so a teammate can finish.

    Costs real ticks, not zero: the world only advances while some agent has a
    primitive running, so a free wait would stop it rather than yield it.
    """

    action_type: Literal["wait"] = "wait"
    ticks: int = Field(
        default=200,
        ge=1,
        le=600,
        description="How long to hold, in simulation ticks. A teammate's "
                    "navigation takes roughly 300-500, so 200 is a short "
                    "pause and 600 is most of a long trip.",
    )


# Union type for all actions
LLMAction = Union[
    NavigateToAction,
    GraspAction,
    PlaceOnTopAction,
    PlaceInsideAction,
    ReleaseAction,
    OpenAction,
    CloseAction,
    ToggleOnAction,
    ToggleOffAction,
    LoadOntoAction,
    UnloadFromAction,
    WaitAction,
]


class Task(str, Enum):
    """The BDDL predicates the plan pipeline still understands structurally.

    Replaces crafter's achievement list (collect_wood, make_stone_pickaxe...).
    These are BDDL predicate tokens, so a task named with one is the same shape
    as the goal condition L1d evaluates and M9's check_goal will compare.

    **This is no longer the set a plan must choose from** (user, 2026-09-14).
    ``TaskSpecification.task`` is a free string: a team that is deliberately
    keeping a robot out of the way, or doing something these six do not name,
    says so in its own words. The enum stays because one thing does read the
    token -- ``_ensure_task_terminal_action`` appends the action that can
    achieve it, ``place_on_top`` for ``ontop`` and so on -- and that mapping
    only exists for these six. Anything else gets no appended action, which is
    the right default: the guard is for a plan that states a goal and lists
    only navigation, and a plan that names no predicate has no such omission.

    Nothing else reads it. The route and goal checkers work from the activity's
    own predicates, never from here.
    """

    ONTOP = "ontop"
    INSIDE = "inside"
    OPEN = "open"
    CLOSED = "closed"
    TOGGLED_ON = "toggled_on"
    HOLDING = "holding"


#: ``ontop(notebook.n.01_1, bed.n.01_1)`` -> ``("ontop", "notebook.n.01_1",
#: "bed.n.01_1")``. A plan's task is one free string (user, 2026-09-14); this
#: reads a BDDL predicate back out of it when the team wrote one, and returns
#: None for anything else -- ``standby(drone_3)``, ``transport die to bedroom``.
_PREDICATE = re.compile(r"^\s*([a-z_][a-z_0-9]*)\s*\(\s*([^,()]+?)\s*(?:,\s*([^,()]+?)\s*)?\)\s*$")


def parse_task(specification: str):
    """``(predicate, target, reference)`` for a predicate-shaped task, else None.

    The plan's task used to be a four-field object -- ``task`` from a closed
    enum, ``object_type``, an always-1 ``object_id``, ``reference`` -- and the
    model filled it in whether or not the plan had a goal to state. It is one
    string now, which is both what the prompt shows and what the log records,
    so there is nothing to keep in step. The two ids are recovered here only
    when the string is a predicate, because exactly one thing still needs them:
    `_ensure_task_terminal_action`, which appends ``place_on_top(bed.n.01_1)``
    to a plan that says ``ontop(notebook.n.01_1, bed.n.01_1)`` and then lists
    only navigation. A free-form name has no such omission to repair.
    """
    if not isinstance(specification, str):
        return None
    match = _PREDICATE.match(specification)
    if match is None:
        return None
    predicate, target, reference = match.group(1), match.group(2), match.group(3)
    return predicate.lower(), target, reference


class InterruptDecision(str, Enum):
    """Decision options when agent is interrupted."""
    RESUME = "resume"  # Continue with the current plan
    REPLAN = "replan"  # Generate a completely new plan


# ============================================================================
# Pydantic models for structured LLM output - Action Classes

# Union type for all actions


# ============================================================================
# Response Models
# ============================================================================

class LLMPlanResponse(BaseModel):
    """Structured response from LLM for plan generation."""
    task: str = Field(
        description="What this plan is for, in your own words. When it makes a "
                    "BDDL predicate true write it as `predicate(target)` or "
                    "`predicate(target, reference)` with ids from this robot's "
                    "own listing -- ontop, inside, open, closed, toggled_on, "
                    "holding -- so it matches the goal. Otherwise name it "
                    "honestly, e.g. `standby(jackal_1)` for a robot you are "
                    "deliberately keeping out of the way. Never name a goal the "
                    "plan is not pursuing"
    )
    actions: List[LLMAction] = Field(description="List of actions to execute")
    reasoning: str = Field(description="Brief explanation of why this plan was chosen")


class LLMMessageResponse(BaseModel):
    """Structured response from LLM for message generation."""
    recipients: List[str] = Field(description="List of agent IDs to send message to")
    content: str = Field(description="Message content to send")
    reasoning: str = Field(description="Brief explanation of why sending this message")


class LLMInterruptResponse(BaseModel):
    """Structured response from LLM for interrupt handling."""
    decision: InterruptDecision = Field(
        description="Whether to resume the current plan or generate a new plan"
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made based on the messages received"
    )
    # Optional new plan - only required if decision is REPLAN
    new_plan: Optional[LLMPlanResponse] = Field(
        default=None,
        description="The new plan to execute (required if decision is 'replan')"
    )


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Wrapper for Azure OpenAI and OpenAI-compatible API calls.
    
    Supports these backends:
    - Azure OpenAI: Azure-hosted OpenAI models with structured output support
    - Foundry: Azure AI Foundry OpenAI-compatible endpoint
    - DeepSeek: DeepSeek OpenAI-compatible endpoint
    
    Takes messages and a response format, returns structured response.
    """
    
    #: Seconds a single call may take before the SDK gives up. A hanging call
    #: does not stall one agent, it stalls the simulation: R freezes the env, so
    #: every robot stands still until the call returns. One Team Plan Generation
    #: in individual_agents12_..._060731 spent its whole 128k completion budget
    #: on reasoning tokens, emitted nothing parseable, and took 552 s -- 42 % of
    #: that run's wall clock, during which env_step went 3119 -> 3120. The
    #: healthy calls in the same run averaged 7.3 s and the slowest was 10.6 s,
    #: so this is ~10x the worst good call. Timing out raises, which lands on the
    #: same path as any other LLM failure: the hold-position fallback, and the
    #: agent asks again next round with the world having moved.
    #:
    #: The SDK's own ``max_retries`` is turned off wherever this is applied, or
    #: the bound is not a bound: the timeout is per attempt, so the default 2
    #: retries make 100 s mean 300 s of frozen simulation. Losing those retries
    #: costs little here -- ``generate`` has its own rate-limit loop, and a
    #: transient failure now spends one planning round on the hold-position
    #: fallback, which is itself a retry, and one that lets the world move in
    #: between instead of freezing it.
    REQUEST_TIMEOUT_SECONDS = 100.0

    # Backends that support OpenAI's structured output (beta.chat.completions.parse)
    STRUCTURED_OUTPUT_BACKENDS = {"azure"}
    JSON_MODE_BACKENDS = {"deepseek"}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-chat",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        backend: str = "azure",  # "azure", "foundry", or "deepseek"
        base_url: Optional[str] = None,
        verbose: bool = False,  # Print API calls and responses
        timeout: Optional[float] = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (reads from env var if not provided)
            model: Model name to use
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure API version
            backend: Backend type ("azure", "foundry", or "deepseek")
            base_url: Custom base URL for OpenAI-compatible backends
            verbose: If True, print API calls and responses for debugging
            timeout: Seconds per request; defaults to REQUEST_TIMEOUT_SECONDS
        """
        self.model = model
        self.backend = backend.lower()
        self.verbose = verbose
        self.timeout = self.REQUEST_TIMEOUT_SECONDS if timeout is None else timeout
        
        if self.backend == "azure":
            # Azure OpenAI
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                timeout=self.timeout,
                max_retries=0,
            )
        elif self.backend == "deepseek":
            # DeepSeek via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                timeout=self.timeout,
                max_retries=0,
            )
        elif self.backend == "foundry":
            # Azure AI Foundry via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("AZURE_FOUNDRY_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_FOUNDRY_API_KEY"),
                timeout=self.timeout,
                max_retries=0,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'azure', 'foundry', or 'deepseek'.")
    
    @classmethod
    def from_env(
        cls,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ) -> "LLMClient":
        """
        Create LLM client from environment variables.
        
        Args:
            backend: Override backend type. If None, auto-detect from env vars.
            model: Override model/deployment name. If None, use backend-specific env var.
            verbose: If True, print API calls and responses for debugging.
        
        Environment variables:
            - LLM_BACKEND: "azure", "foundry", or "deepseek"
            - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION
            - AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY, AZURE_FOUNDRY_MODEL
            - DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
        """
        # Determine backend
        if backend is None:
            backend = os.getenv("LLM_BACKEND", "").lower()
        
        # Auto-detect if not specified (prefer Azure)
        if not backend:
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                backend = "azure"
            elif os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"):
                backend = "foundry"
            elif os.getenv("DEEPSEEK_API_KEY"):
                backend = "deepseek"
            else:
                backend = "azure"  # Default to Azure
        
        if backend == "deepseek":
            return cls(
                backend="deepseek",
                base_url=os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model=model or os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
                verbose=verbose,
            )
        if backend == "foundry":
            return cls(
                backend="foundry",
                base_url=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
                api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
                model=model or os.getenv("AZURE_FOUNDRY_MODEL", "Llama-4-Scout-17B-16E-Instruct"),
                verbose=verbose,
            )
        else:  # azure (default)
            return cls(
                backend="azure",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                model=model or os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                verbose=verbose,
            )
    
    def _parse_json_response(self, content: str, response_format: type):
        """
        Parse JSON response manually for backends without structured output support.
        
        Args:
            content: Raw response content (may contain JSON in markdown code blocks)
            response_format: Pydantic model to parse into
            
        Returns:
            Parsed Pydantic model instance
        """
        import json
        import re
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        # Parse JSON and validate with Pydantic
        data = json.loads(json_str)
        return response_format.model_validate(data)
    
    def _get_json_schema_prompt(self, response_format: type) -> str:
        """
        Generate a prompt suffix with JSON schema for models without structured output.
        
        Args:
            response_format: Pydantic model to generate schema for
            
        Returns:
            Prompt string describing the expected JSON format
        """
        schema = response_format.model_json_schema()
        import json
        return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\nRespond ONLY with the JSON object, no other text."

    @staticmethod
    def _is_guard_filter_error(error: Exception) -> bool:
        """Return True for Azure/OpenAI content-filter guardrail failures."""
        parts = [
            str(error),
            str(getattr(error, "code", "")),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
        ]
        text = " ".join(parts).lower()
        markers = (
            "content_filter",
            "content filter",
            "responsibleaipolicy",
            "policy violation",
            "filtered due to",
            "safety system",
        )
        return any(marker in text for marker in markers)
    
    def generate(
        self,
        messages: List[Dict],
        response_format: Optional[type] = None,
        temperature: float = 0.7,
        max_retries: int = 6,
        retry_delay: float = 10.0
    ) -> tuple:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Pydantic model for structured output (LLMPlanResponse, 
                           LLMMessageResponse, LLMInterruptResponse, or None for free-form)
            temperature: Sampling temperature (ignored for models that don't support it)
            max_retries: Maximum number of retries on rate limit errors
            retry_delay: Seconds to wait before retrying (default: 30)
            
        Returns:
            Tuple of (response, usage_dict)
            - response: Parsed Pydantic model if response_format provided, else string
            - usage_dict: token counts plus latency/retry telemetry
        """
        # Some models (like gpt-5.2-chat) don't support temperature parameter
        # Check if model supports temperature, otherwise omit it
        supports_temperature = "gpt-5" not in self.model.lower()
        
        # Verbose logging: print API call info
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[LLM API CALL] Backend: {self.backend}, Model: {self.model}")
            print(f"  Response format: {response_format.__name__ if response_format else 'None (free-form)'}")
            print(f"  Temperature: {temperature if supports_temperature else 'N/A (model default)'}")
            print(f"  Messages ({len(messages)}):")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]
                print(f"    [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
            print(f"{'='*60}")
        
        raw_content = None  # For verbose logging
        call_start = time.monotonic()
        retry_count = 0
        rate_limit_retry_count = 0
        api_error_retry_count = 0
        
        # Build optional kwargs (only include temperature if supported)
        temp_kwargs = {"temperature": temperature} if supports_temperature else {}
        
        # Retry loop for rate limit errors
        for attempt in range(max_retries + 1):
            try:
                if response_format is not None and self.backend in self.STRUCTURED_OUTPUT_BACKENDS:
                    # Use OpenAI's structured output (beta.chat.completions.parse)
                    raw_response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=response_format,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.parsed
                    raw_content = str(response)
                elif response_format is not None:
                    # For OpenAI-compatible backends without structured output,
                    # add the JSON schema to the last user message and parse manually.
                    messages_with_schema = messages.copy()
                    if messages_with_schema and messages_with_schema[-1]["role"] == "user":
                        messages_with_schema[-1] = messages_with_schema[-1].copy()
                        messages_with_schema[-1]["content"] += self._get_json_schema_prompt(response_format)

                    response_kwargs = {}
                    if self.backend in self.JSON_MODE_BACKENDS:
                        response_kwargs["response_format"] = {"type": "json_object"}

                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_with_schema,
                        **response_kwargs,
                        **temp_kwargs
                    )
                    raw_content = raw_response.choices[0].message.content
                    response = self._parse_json_response(raw_content, response_format)
                else:
                    # Free-form output
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.content
                    raw_content = response
                
                # Success - break out of retry loop
                break
                
            except RateLimitError as e:
                if attempt < max_retries:
                    retry_count += 1
                    rate_limit_retry_count += 1
                    print(f"\n[RATE LIMIT] Hit rate limit. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                    raise
            except APIError as e:
                # Azure may return 429 as APIError instead of RateLimitError
                if hasattr(e, 'status_code') and e.status_code == 429:
                    if attempt < max_retries:
                        retry_count += 1
                        rate_limit_retry_count += 1
                        api_error_retry_count += 1
                        print(f"\n[RATE LIMIT] Azure 429 error. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                        raise
                else:
                    if self._is_guard_filter_error(e):
                        print(f"\n[GUARD FILTER] LLM request blocked by content filter/policy: {e}")
                    raise
        
        finish_reason = None
        guard_filter_count = 0
        if getattr(raw_response, "choices", None):
            finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
            if str(finish_reason).lower() == "content_filter":
                guard_filter_count = 1
                print("\n[GUARD FILTER] LLM response was filtered by content policy.")

        response_usage = getattr(raw_response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(response_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": time.monotonic() - call_start,
            "retry_count": retry_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "api_error_retry_count": api_error_retry_count,
            "guard_filter_count": guard_filter_count,
            "finish_reason": None if finish_reason is None else str(finish_reason),
        }
        
        # Verbose logging: print response
        if self.verbose:
            print(f"\n[LLM RESPONSE] Backend: {self.backend}")
            print(f"  Usage: {usage}")
            print(f"  Raw content ({len(raw_content) if raw_content else 0} chars):")
            if raw_content:
                # Print first 500 chars
                print(f"    {raw_content[:500]}{'...' if len(raw_content) > 500 else ''}")
            print(f"{'='*60}\n")
        
        return response, usage
    
    def generate_plan(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a plan response."""
        return self.generate(messages, response_format=LLMPlanResponse, temperature=temperature)
    
    def generate_message(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a message response."""
        return self.generate(messages, response_format=LLMMessageResponse, temperature=temperature)
    
    def generate_interrupt_decision(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate an interrupt decision response."""
        return self.generate(messages, response_format=LLMInterruptResponse, temperature=temperature)

    def generate_team_plan(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """One call that plans for every robot on a team."""
        return self.generate(messages, response_format=LLMTeamPlanResponse, temperature=temperature)

    def generate_team_interrupt_decision(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """One call that decides resume-or-replan for every robot on a team."""
        return self.generate(messages, response_format=LLMTeamInterruptResponse, temperature=temperature)


class TeamAgentPlan(BaseModel):
    """One team member's plan, inside a single team-wide response.

    Same three fields as :class:`LLMPlanResponse` plus the agent it is for, so
    the team response is N of these rather than a nested map -- a list keeps the
    JSON schema simple enough for structured output, and the agent_id travels
    *with* the plan rather than as a key that can drift from it.
    """

    agent_id: str = Field(description="Which robot this plan is for; must be one of the team's ids")
    task: str = Field(
        description="What this plan is for, in your own words. When it makes a "
                    "BDDL predicate true write it as `predicate(target)` or "
                    "`predicate(target, reference)` with ids from this robot's "
                    "own listing -- ontop, inside, open, closed, toggled_on, "
                    "holding -- so it matches the goal. Otherwise name it "
                    "honestly, e.g. `standby(jackal_1)` for a robot you are "
                    "deliberately keeping out of the way. Never name a goal the "
                    "plan is not pursuing"
    )
    actions: List[LLMAction] = Field(description="List of actions for this robot to execute")
    reasoning: str = Field(description="Brief explanation of why this robot is doing this")


class LLMTeamPlanResponse(BaseModel):
    """One call, one plan per robot on the team.

    The team brain sees every member's observation at once, so the interesting
    field is ``reasoning``: it is where the allocation is justified, and the only
    place the division of labour is visible before the robots act on it.
    """

    plans: List[TeamAgentPlan] = Field(description="Exactly one plan per robot on the team")
    reasoning: str = Field(description="Why the work is divided between the robots this way")


class LLMMessageboardPlanResponse(LLMTeamPlanResponse):
    """The team plan plus one post to the shared board -- `decentralized_messageboard`.

    The post travels in the same call as the plans on purpose: it is the team's
    public commitment for *this* round, so it has to be written by the same
    reasoning that produced the plans, not by a second call that could drift
    from them. Other teams read it when they next plan; nobody is interrupted.
    """

    board_post: str = Field(
        description=(
            "One or two sentences for the shared board, in the task's ids: which robot "
            "of yours goes for which cargo or task leg this round, and what you are "
            "leaving to other teams. Other teams read it before they plan."
        )
    )


class LLMChainPlanResponse(LLMTeamPlanResponse):
    """The team plan plus what to tell the teams behind -- `broadcast_chain`.

    In the same call as the plans, for the reason `board_post` is: the teams
    behind act on this, so it must come from the reasoning that produced the
    plans. What it replaces was `_plan_summary`, a mechanical join of each
    robot's `plan.specification`. Measured on the 2026-09-13 LH run: with one
    cargo in the activity every team's broadcast read
    `drone_1: ontop(notebook.n.01_1, cabinet.n.01_1); jackal_1:
    holding(notebook.n.01_1); ...` -- identical across teams, silent on who
    actually held the notebook and which leg was claimed, so the teams behind
    could not tell whose turn it was and grabbed the cargo from each other.
    """

    broadcast: str = Field(
        description=(
            "One or two sentences for the teams behind you in the chain, in the task's "
            "ids: which of your robots holds or is going for the cargo, which leg you "
            "are taking this round, and what you are leaving to them. They act on this, "
            "so say what you are committing to, not what you might do."
        )
    )


class NotifyRequest(BaseModel):
    """Reserved: a team's request to interrupt named teams -- the notify tool.

    Not offered to the model until ``MessageboardTeamBrain.NOTIFY_TOOL_ENABLED``
    is set. Its shape is fixed now so the board, the brain and the tests agree
    on it before anything is delivered.
    """

    teams: List[str] = Field(description="Which other teams to interrupt, by team name")
    content: str = Field(description="What they must know now, in one or two sentences")
    reasoning: str = Field(description="Why this cannot wait for them to read the board")


class LLMMessageboardNotifyPlanResponse(LLMMessageboardPlanResponse):
    """Reserved: the board response with the notify tool available."""

    notify: Optional[NotifyRequest] = Field(
        default=None,
        description="Leave null unless another team must be interrupted before it next plans",
    )


class TeamAgentInterruptDecision(BaseModel):
    """Resume or replan, decided per robot after the team was interrupted."""

    agent_id: str = Field(description="Which robot this decision is for")
    decision: InterruptDecision = Field(
        description="Whether this robot resumes its current plan or gets a new one"
    )
    reasoning: str = Field(description="Why, given the message and this robot's own progress")
    new_plan: Optional[LLMPlanResponse] = Field(
        default=None,
        description="Required when decision is 'replan'; ignored when it is 'resume'",
    )


class LLMTeamInterruptResponse(BaseModel):
    """A message interrupts the whole team; the brain answers for each member.

    Per-agent rather than team-wide on purpose: a message that changes what one
    robot should do usually leaves the others' plans perfectly good, and making
    the whole team replan would throw away work the message never contradicted.
    """

    decisions: List[TeamAgentInterruptDecision] = Field(
        description="Exactly one decision per robot on the team"
    )
    reasoning: str = Field(description="What the message means for the team as a whole")


class LLMChainInterruptResponse(LLMTeamInterruptResponse):
    """The interrupt decisions plus what to relay onward -- `broadcast_chain`.

    The chain relays once per interrupt round whatever it decided, resume
    included, or the team behind waits out `_await_relay` for a message that
    was never coming. What it relayed was `_current_allocation()`, a join of
    every robot's current plan specification, which has the fault
    `LLMChainPlanResponse` documents: with one cargo it reads the same for
    every team and names no holder. Same call as the decisions, for the same
    reason -- the teams behind act on it, so it must come from the reasoning
    that made the decision.
    """

    broadcast: str = Field(
        description=(
            "One or two sentences for the teams behind you, in the task's ids: what "
            "this message changed for your robots, which robot of yours holds or is "
            "going for the cargo, and what you leave to them. Say it even when "
            "everyone resumed -- they are waiting to hear from you."
        )
    )


# ============================================================================
# The task-graph mode (`tag`): DIG-TAG's task-graph actions and notify, riding
# in the team plan and in the team interrupt decision. The vocabulary is
# dig_tag's (coop2/comm_topology/llm_tag.py applies it); these are only the
# shapes the model fills in. Mirrors dig-tag-icra's TAGPart / TAGToolCall.
# ============================================================================

TagTool = Literal["open", "edit", "update", "split", "join", "close", "attach"]


class TagPart(BaseModel):
    """One part of a split."""

    goal: str = Field(description="The part's goal")
    rule: Optional[str] = Field(default=None, description="How the part is judged")
    state: Optional[str] = Field(default=None, description="The part's reported state")
    identity: Optional[str] = Field(
        default=None, description="An existing task identity the part continues, or null for a fresh one"
    )


class TagToolCall(BaseModel):
    """One task-graph action; the fields a tool does not take stay null."""

    tool: TagTool = Field(description="open, edit, update, split, join, close, or attach")
    task: Optional[str] = Field(default=None, description="Version id (q3) for edit, update, split, and attach")
    tasks: Optional[List[str]] = Field(default=None, description="Version ids to join")
    identity: Optional[str] = Field(
        default=None, description="Task identity (k1): the target of close, or the identity a join continues"
    )
    goal: Optional[str] = Field(default=None, description="Goal for open, edit, and join")
    rule: Optional[str] = Field(default=None, description="Rule for open, edit, and join")
    state: Optional[str] = Field(default=None, description="Reported state for open, update, and join")
    parts: Optional[List[TagPart]] = Field(default=None, description="Parts for split (two or more)")
    payload: Optional[str] = Field(default=None, description="Evidence or note text for attach")


class LLMTagPlanResponse(LLMTeamPlanResponse):
    """The team plan plus the task-graph part of the round -- `tag`.

    Applied in this order: `tag_actions` on the shared graph, then `notify`
    (each named team is interrupted and reads the graph), then the plans go
    to the robots. Same call on purpose: the graph is changed by the reasoning
    that made the plans, so the two cannot drift.
    """

    tag_actions: List[TagToolCall] = Field(
        description="Task-graph actions, applied to the shared graph in order; may be empty"
    )
    notify: List[str] = Field(
        description="Team names to wake so they read the task graph now; may be empty. "
                    "Name every team that must read it: one notification wakes them all "
                    "and costs one unit of the budget, however many are named"
    )


class LLMTagInterruptResponse(LLMTeamInterruptResponse):
    """The interrupt decision plus the task-graph part -- the same round, as
    DIG-TAG has it: a notified team observes the graph and answers the same way."""

    tag_actions: List[TagToolCall] = Field(
        description="Task-graph actions, applied to the shared graph in order; may be empty"
    )
    notify: List[str] = Field(
        description="Team names to wake so they read the task graph now; may be empty. "
                    "Name every team that must read it: one notification wakes them all "
                    "and costs one unit of the budget, however many are named"
    )


# ============================================================================
# Utility functions
# ============================================================================

def load_env_file(env_path: str = ".env"):
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file (will also check project root)
    """
    # Try current directory first
    if not os.path.exists(env_path):
        # Try to find .env in project root (relative to this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)


# Auto-load .env file on import
load_env_file()
