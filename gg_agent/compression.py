"""Keep the transcript inside the model's context window.

Three passes, cheapest first, each run only if the one before left the request
still over the threshold:

  A. **Reclaim** (no LLM): stub out old tool results, collapse duplicates, shrink
     oversized tool-call arguments. Free, idempotent, and in a tool-heavy session
     it wins back most of the window on its own — one 400KB file read is worth
     more than a hundred turns of conversation.
  B. **Summarize** (one auxiliary model call): replace the middle of the
     transcript with a structured summary, protecting head and tail. This is what
     a session needs once the *conversation itself* fills the window.
  C. **Pressure** (no LLM): give up everything except the tool round in flight.

Both A and B live under the boundary rule that matters: an assistant message with
``tool_calls`` and its ``role:"tool"`` replies are ONE unit, because every
provider rejects a request where they are separated (see ``loop.run_tool_round``).
Phase A rewrites tool *content* and never removes a message, so it cannot break
that pairing. Phase B drops messages, so it aligns its cut (``align_tail_start``)
and verifies the result before returning it.

Sizing is deliberately rough. A real tokenizer per provider would be exact and
wrong the moment the model changes, so instead a chars/4 estimate is calibrated
against the ``prompt_tokens`` the provider actually reported (see
``note_real_usage``): after one call the estimate is within a few percent, and
compression only ever needs to know "are we near the edge", not the exact count.

Mirrors hermes-agent: agent/context_compressor.py (phase A is roughly its
``prune_tool_results_only`` path, minus the durable cooldowns and telemetry)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any

from .prompts import SUMMARY_PREFIX, build_summary_prompt

logger = logging.getLogger(__name__)

# Set by the loop once a store has accepted a message. Defined here rather than
# imported from ``loop`` because the dependency runs the other way: the loop
# imports this module. ``loop.PERSISTED_KEY`` is this name.
PERSISTED_KEY = "_persisted"

# ── Sizing ───────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4           # English prose + code, close enough before calibration
PER_MESSAGE_OVERHEAD = 4      # role/delimiter framing every provider adds
DEFAULT_CONTEXT_LENGTH = 128_000
SCALE_BOUNDS = (0.5, 2.0)     # clamp on the calibration factor; usage reporting varies

# Window sizes by model, first substring match wins — so order matters: the more
# specific name has to come before the family it belongs to.
MODEL_CONTEXT_LENGTHS: tuple[tuple[str, int], ...] = (
    ("gpt-4.1", 1_047_576),
    ("gpt-5", 400_000),
    ("gpt-4o", 128_000),
    ("o4", 200_000),
    ("claude-3-5", 200_000),
    ("claude", 200_000),
    ("gemini", 1_000_000),
    ("llama-3.3", 128_000),
    ("llama", 128_000),
    ("deepseek", 128_000),
    ("mistral", 32_768),
    ("qwen", 32_768),
)

# ── Trigger ──────────────────────────────────────────────────────────────────

THRESHOLD_RATIO = 0.75        # of the usable input budget
MIN_THRESHOLD_RATIO = 0.85    # a threshold may never sit above this much of the budget
TAIL_RATIO = 0.30             # of the threshold: recent messages pruning may not touch
MIN_TAIL_MESSAGES = 6         # ...but always keep at least this many intact

# ── Pruning ──────────────────────────────────────────────────────────────────

MIN_PRUNE_CHARS = 600         # below this a tool result is not worth stubbing
MAX_TOOL_ARG_CHARS = 400      # per string value inside a tool call's arguments

# ── Summarizing ──────────────────────────────────────────────────────────────

SUMMARY_KEY = "_summary"          # marks the summary message in-process
SUMMARY_RATIO = 0.20              # target summary size, as a fraction of the middle
SUMMARY_MIN_TOKENS = 200
SUMMARY_MAX_TOKENS = 1_500
MIN_MIDDLE_TOKENS = 1_000         # a middle smaller than this is not worth a call
SUMMARY_INPUT_MAX_CHARS = 60_000  # what the summarizer is shown, after sampling
SUMMARY_TURN_MAX_CHARS = 2_000    # per rendered turn
SUMMARY_ARG_MAX_CHARS = 300       # per rendered tool call's arguments
MIN_SUMMARY_CHARS = 120           # shorter than this is a refusal or a stub, not a summary
SUMMARY_TIMEOUT = 90.0            # seconds; a hung summarizer must not hang the turn
SUMMARY_COOLDOWN = 300.0          # seconds before retrying a failed summarizer

# Anti-thrash: compression that does not actually reclaim anything still costs a
# cache miss on the next request, so two useless passes stop it for the session.
MIN_RECLAIM_RATIO = 0.10
STRIKE_LIMIT = 2


def estimate_tokens(value: Any) -> int:
    """Rough token count for a string (or anything JSON-serialisable)."""
    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    return (PER_MESSAGE_OVERHEAD + estimate_tokens(msg.get("content"))
            + estimate_tokens(msg.get("tool_calls")) + estimate_tokens(msg.get("name")))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def resolve_context_length(agent: Any) -> int:
    """This model's window: explicit setting, env override, profile, model table."""
    explicit = getattr(agent, "context_length", None)
    if explicit:
        return int(explicit)
    env = os.getenv("GG_CONTEXT_LENGTH", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    declared = getattr(agent.profile, "context_length", 0)
    if declared:
        return int(declared)
    model = (agent.model or "").lower()
    for fragment, length in MODEL_CONTEXT_LENGTHS:
        if fragment in model:
            return length
    return DEFAULT_CONTEXT_LENGTH


def compression_threshold(agent: Any) -> int:
    """Prompt-token count at which compression should run.

    The provider reserves ``max_tokens`` of *output* from the same window, so the
    usable input budget is smaller than the raw context length — a threshold based
    on the full window lets a session hit a provider 400 before compression ever
    fires. The result is also capped below the budget: a threshold at 100% is one
    that can never be reached, since the request is rejected first.
    """
    context_length = resolve_context_length(agent)
    budget = context_length - (agent.max_tokens or 0)
    if budget <= 0:                       # max_tokens >= window: trust the window
        budget = context_length
    return max(1, min(int(budget * THRESHOLD_RATIO), int(budget * MIN_THRESHOLD_RATIO)))


def prompt_tokens_now(agent: Any, messages: list[dict[str, Any]]) -> int:
    """Estimated prompt size of ``messages``, corrected by what the provider last billed."""
    return int(estimate_messages_tokens(messages) * getattr(agent, "_token_scale", 1.0))


def note_real_usage(agent: Any, rough_tokens: int, prompt_tokens: int) -> None:
    """Calibrate the estimator from one real ``prompt_tokens`` reading.

    ``rough_tokens`` is the uncalibrated estimate for the request that produced
    ``prompt_tokens``. The factor is clamped: some endpoints report cache hits
    oddly, and a wild factor would either wedge the session in a compression loop
    or let it sail past the window.
    """
    if rough_tokens <= 0 or prompt_tokens <= 0:
        return
    low, high = SCALE_BOUNDS
    agent._token_scale = max(low, min(high, prompt_tokens / rough_tokens))


# ── Phase A: reclaim without an LLM ──────────────────────────────────────────


def _human_size(chars: int) -> str:
    return f"{chars / 1024:.1f}KB" if chars >= 1024 else f"{chars}B"


def protected_tail_start(messages: list[dict[str, Any]], tail_tokens: int,
                         min_messages: int = MIN_TAIL_MESSAGES) -> int:
    """Index of the first message in the protected tail.

    A token budget, not a message count: a fixed "last N" either protects nothing
    (ten one-line turns) or everything (three 30KB tool results). The count floor
    only stops a single huge message from consuming the whole budget and leaving
    the recent exchange unprotected.
    """
    accumulated, cut = 0, len(messages)
    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[i])
        if accumulated > tail_tokens and (len(messages) - i) > min_messages:
            break
        cut = i
    return cut


def _prunable(msg: dict[str, Any], durable_only: bool) -> bool:
    """Tool results are prunable; everything else is the conversation itself.

    ``durable_only`` skips messages the store hasn't accepted yet: rewriting one
    of those would persist the stub instead of the original on the next flush,
    which is the one way this pass could actually lose data.
    """
    if msg.get("role") != "tool" or msg.get("_pruned"):
        return False
    return bool(msg.get("_persisted")) or not durable_only


def _dedupe_tool_results(messages: list[dict[str, Any]], start: int, end: int,
                         durable_only: bool) -> int:
    """Identical results collapse to their newest copy.

    A polling loop (``git status`` four times, the same file read twice) pays for
    the same bytes on every subsequent request. The scan starts in the protected
    tail so a result the model is still looking at counts as the copy to keep,
    but only messages before ``end`` are ever rewritten. The stub deliberately
    does not point at "the later copy": that copy may itself be stubbed by the
    pass below, and a reference to content that is no longer there is worse than
    no reference at all.
    """
    seen: set[tuple[str, str]] = set()
    reclaimed = 0
    for i in range(len(messages) - 1, start - 1, -1):     # newest first: the last copy wins
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < MIN_PRUNE_CHARS:
            continue
        key = (str(msg.get("name") or ""), content)
        if key not in seen:
            seen.add(key)
        elif i < end and _prunable(msg, durable_only):
            reclaimed += _replace_content(
                msg, f"[repeated {msg.get('name') or 'tool'} result pruned to save context "
                     f"— re-run the tool if you need it again]")
    return reclaimed


def _stub_old_tool_results(messages: list[dict[str, Any]], start: int, end: int,
                           durable_only: bool) -> int:
    """Replace big, old tool results with a line saying what used to be there.

    Naming the tool matters: the model has to be able to tell that re-running it
    is an option, or it will answer from a half-remembered summary instead.
    """
    reclaimed = 0
    for i in range(start, end):
        msg = messages[i]
        if not _prunable(msg, durable_only):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < MIN_PRUNE_CHARS:
            continue
        reclaimed += _replace_content(
            msg, f"[{msg.get('name') or 'tool'} result pruned to save context "
                 f"({_human_size(len(content))}) — re-run the tool if you need it again]")
    return reclaimed


def _replace_content(msg: dict[str, Any], stub: str) -> int:
    before = estimate_tokens(msg.get("content"))
    msg["content"] = stub
    msg["_pruned"] = True                        # never stub a stub
    return max(0, before - estimate_tokens(stub))


def _shrink_tool_call_args(messages: list[dict[str, Any]], start: int, end: int) -> int:
    """Trim long string values inside old tool-call arguments.

    Re-serialised as valid JSON rather than truncated as text: the Anthropic
    transport parses these back into a dict (``transports/anthropic.py``), and a
    half a JSON object parses to nothing.
    """
    reclaimed = 0
    for i in range(start, end):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            raw = function.get("arguments")
            if not isinstance(raw, str) or len(raw) <= MAX_TOOL_ARG_CHARS:
                continue
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue                          # unparseable args are the model's business
            shrunk = json.dumps(_shrink(parsed), ensure_ascii=False)
            if len(shrunk) < len(raw):
                function["arguments"] = shrunk
                reclaimed += max(0, estimate_tokens(raw) - estimate_tokens(shrunk))
    return reclaimed


def _shrink(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TOOL_ARG_CHARS:
        return value[:MAX_TOOL_ARG_CHARS] + f"…[{len(value) - MAX_TOOL_ARG_CHARS} more chars]"
    if isinstance(value, dict):
        return {k: _shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shrink(v) for v in value]
    return value


def pressure_tail_start(messages: list[dict[str, Any]]) -> int:
    """The smallest tail worth protecting: the tool round in flight.

    Used when the normal pass could not free enough — typically one enormous tool
    result in a short session, where the message-count floor protects the very
    thing that filled the window. Everything up to the newest assistant turn that
    asked for tools stays prunable; that turn and its replies do not, because the
    model is about to reason about them.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            return i
    return max(0, len(messages) - 1)


def reclaim(messages: list[dict[str, Any]], *, tail_tokens: int,
            durable_only: bool = False, pressure: bool = False) -> dict[str, int]:
    """Run every Phase A pass over the messages outside head and protected tail.

    Mutates in place: the dicts are shared with ``Agent.history`` and carry the
    loop's persistence markers, so replacing them would strand both.
    Returns ``{"tokens": reclaimed, "tail_start": ...}``.
    """
    head = _head_end(messages)
    tail_start = pressure_tail_start(messages) if pressure else protected_tail_start(messages, tail_tokens)
    tail_start = max(head, tail_start)
    if tail_start <= head:
        return {"tokens": 0, "tail_start": tail_start, "head_end": head}
    tokens = (_dedupe_tool_results(messages, head, tail_start, durable_only)
              + _stub_old_tool_results(messages, head, tail_start, durable_only)
              + _shrink_tool_call_args(messages, head, tail_start))
    return {"tokens": tokens, "tail_start": tail_start, "head_end": head}


def _head_end(messages: list[dict[str, Any]]) -> int:
    """End of the protected head: the system prompt and the first user turn.

    The original request is the one thing later reconstruction distorts most, and
    it is cheap to keep, so it is never touched by either phase.
    """
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            return i + 1
    return min(1, len(messages))


# ── Phase B: summarize the middle ────────────────────────────────────────────


def is_summary_message(msg: dict[str, Any]) -> bool:
    """A message this module wrote. Checked two ways because only one survives a
    round trip through the store: the marker key is dropped when a message becomes
    a row, the prefix is part of the content."""
    if msg.get(SUMMARY_KEY):
        return True
    content = msg.get("content")
    return isinstance(content, str) and content.startswith(SUMMARY_PREFIX)


def build_summary_message(text: str) -> dict[str, Any]:
    """The summary goes in as a user turn.

    Not assistant: a model that finds its own voice narrating work it cannot
    remember doing tends to keep narrating instead of acting. As a user turn it
    reads as what it is — a briefing handed to the agent.
    """
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\n{text.strip()}", SUMMARY_KEY: True}


def align_tail_start(messages: list[dict[str, Any]], idx: int) -> int:
    """Move a cut off the middle of a tool round.

    A tail that begins with a ``role:"tool"`` message is a reply whose assistant
    turn is about to be summarized away — an orphan, and an invalid request on
    every provider. Walking back lands on the assistant message that opened the
    round, which takes its whole round into the protected tail.
    """
    while 0 < idx < len(messages) and messages[idx].get("role") == "tool":
        idx -= 1
    return idx


def _durable_end(messages: list[dict[str, Any]], start: int, end: int, durable_only: bool) -> int:
    """Stop the middle at the first message the store has not accepted.

    Dropping a message that was never written is the one way this phase can
    actually destroy something: the store is the archive that makes summarizing
    safe in the first place.
    """
    if not durable_only:
        return end
    for i in range(start, end):
        if not messages[i].get(PERSISTED_KEY):
            return i
    return end


def _cap(text: Any, limit: int) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"…[{len(text) - limit} more chars]"


def render_for_summary(messages: list[dict[str, Any]]) -> str:
    """The middle, flattened into something a model can read in one pass."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            lines.append(f"[tool result: {msg.get('name') or 'tool'}]\n"
                         f"{_cap(content or '', SUMMARY_TURN_MAX_CHARS)}")
            continue
        if content:
            lines.append(f"[{role}]\n{_cap(content, SUMMARY_TURN_MAX_CHARS)}")
        for call in msg.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else {}
            if isinstance(function, dict):
                lines.append(f"[{role} calls {function.get('name')}] "
                             f"{_cap(function.get('arguments') or '', SUMMARY_ARG_MAX_CHARS)}")
    text = "\n\n".join(lines)
    if len(text) <= SUMMARY_INPUT_MAX_CHARS:
        return text
    # Keep both ends: the oldest turns carry the task, the newest carry the state.
    half = SUMMARY_INPUT_MAX_CHARS // 2
    return (text[:half] + f"\n\n…[{len(text) - SUMMARY_INPUT_MAX_CHARS} chars omitted from the "
                          f"middle of this window]…\n\n" + text[-half:])


def _summary_budget(middle: list[dict[str, Any]]) -> int:
    target = int(estimate_messages_tokens(middle) * SUMMARY_RATIO)
    return max(SUMMARY_MIN_TOKENS, min(SUMMARY_MAX_TOKENS, target))


def _split_previous_summary(middle: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull an earlier summary out of the middle so the new one *updates* it.

    Summarizing a summary along with everything else is how a long session turns
    to mush: each pass re-compresses text that was already lossy.
    """
    previous, turns = None, []
    for msg in middle:
        if is_summary_message(msg) and isinstance(msg.get("content"), str):
            previous = msg["content"].removeprefix(SUMMARY_PREFIX).strip()
        else:
            turns.append(msg)
    return previous, turns


def _summary_blocked(agent: Any) -> str | None:
    """Why phase B may not run right now, or None."""
    remaining = getattr(agent, "_summary_cooldown_until", 0.0) - time.monotonic()
    if remaining > 0:
        return f"cooldown:{remaining:.0f}s"
    if not (agent.profile.default_aux_model or agent.model):
        return "no model"
    return None


async def generate_summary(agent: Any, middle: list[dict[str, Any]],
                           previous: str | None) -> str | None:
    """One auxiliary-model call. None on any failure, with a cooldown armed.

    Deliberately not the main model by default: this runs mid-turn, the output is
    never shown, and a cheap model summarizes structured text about as well as an
    expensive one (``ProviderProfile.default_aux_model``).
    """
    budget = _summary_budget(middle)
    prompt = build_summary_prompt(render_for_summary(middle), budget_tokens=budget,
                                  previous_summary=previous)
    model = agent.profile.resolve_aux_model() or agent.profile.default_aux_model or agent.model
    started = time.monotonic()
    try:
        kwargs = agent.transport.build_kwargs(
            model=model, messages=[{"role": "user", "content": prompt}], tools=None,
            profile=agent.profile, max_tokens=budget * 2, cache_prompt=False,
            session_id=agent.session_id, base_url=agent.base_url)
        raw = agent.transport.call(agent.client, **kwargs)
        if inspect.isawaitable(raw):
            raw = await asyncio.wait_for(raw, SUMMARY_TIMEOUT)
        text = (agent.transport.normalize_response(raw).content or "").strip()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        agent._summary_cooldown_until = time.monotonic() + SUMMARY_COOLDOWN
        logger.warning("Summary failed on %s (%s); falling back for %ds", model, exc, SUMMARY_COOLDOWN)
        agent._emit("summary_failed", model=model, error=str(exc))
        return None
    if len(text) < MIN_SUMMARY_CHARS:
        # A refusal or an empty completion. Same treatment as an error: the next
        # call would almost certainly produce the same thing.
        agent._summary_cooldown_until = time.monotonic() + SUMMARY_COOLDOWN
        logger.warning("Summary from %s was too short to use (%d chars)", model, len(text))
        agent._emit("summary_failed", model=model, error=f"summary too short ({len(text)} chars)")
        return None
    logger.info("Summarized %d messages with %s in %.1fs", len(middle), model,
                time.monotonic() - started)
    return text


def fallback_summary(middle: list[dict[str, Any]], previous: str | None) -> str:
    """A summary assembled without a model, for when the summarizer is unavailable.

    Worse than the real thing and says so, but it keeps the session alive and it
    keeps the anchors that matter most: what the user asked, which files were
    touched, and where things stopped.
    """
    user_turns = [str(m.get("content"))[:400] for m in middle
                  if m.get("role") == "user" and m.get("content")]
    tools: dict[str, int] = {}
    files: list[str] = []
    for msg in middle:
        if msg.get("role") == "tool" and msg.get("name"):
            tools[msg["name"]] = tools.get(msg["name"], 0) + 1
        for call in msg.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else {}
            if isinstance(function, dict):
                _collect_paths(function.get("arguments"), files)
    last_assistant = next((str(m.get("content"))[:600] for m in reversed(middle)
                           if m.get("role") == "assistant" and m.get("content")), "")

    parts = ["(Mechanical summary — the summarizer model was unavailable, so this is an "
             "extract rather than a written summary. Treat the details as incomplete.)"]
    if previous:
        parts.append(f"\n## Earlier summary\n{_cap(previous, 3_000)}")
    if user_turns:
        parts.append("\n## What the user said\n" + "\n".join(f"- {t}" for t in user_turns[:8]))
    if tools:
        parts.append("\n## Tools used\n" + ", ".join(f"{name} ×{n}" for name, n in tools.items()))
    if files:
        parts.append("\n## Files mentioned\n" + ", ".join(files[:12]))
    if last_assistant:
        parts.append(f"\n## Last thing the assistant said\n{last_assistant}")
    return "\n".join(parts)


_PATH_KEYS = {"path", "file", "filename", "file_path", "target", "directory", "dir"}


def _collect_paths(arguments: Any, out: list[str]) -> None:
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError):
        return
    if not isinstance(parsed, dict):
        return
    for key, value in parsed.items():
        if key.lower() in _PATH_KEYS and isinstance(value, str) and value and value not in out:
            out.append(value[:120])


async def summarize_middle(agent: Any, messages: list[dict[str, Any]], *, tail_tokens: int,
                           durable_only: bool) -> int:
    """Replace the middle of ``messages`` with one summary. Returns tokens saved.

    Mutates the list in place (the loop and ``Agent.history`` share it). Returns 0
    without touching anything when the middle is too small to be worth a call,
    when the summarizer is in cooldown, or when the splice would produce a
    transcript the provider would reject.
    """
    blocked = _summary_blocked(agent)
    if blocked:
        logger.debug("Phase B skipped: %s", blocked)
        return 0
    head = _head_end(messages)
    end = _durable_end(messages, head, align_tail_start(messages, protected_tail_start(
        messages, tail_tokens)), durable_only)
    middle = messages[head:end]
    before = estimate_messages_tokens(middle)
    if before < MIN_MIDDLE_TOKENS:
        return 0
    previous, turns = _split_previous_summary(middle)
    if not turns:
        return 0                       # nothing but an earlier summary: already compacted

    agent._emit("summarizing", messages=len(middle), tokens=before)
    text = await generate_summary(agent, turns, previous)
    if text is None:
        text = fallback_summary(turns, previous)
    summary = build_summary_message(text)

    spliced = messages[:head] + [summary] + messages[end:]
    if not _valid_transcript(spliced):
        logger.warning("Phase B produced an invalid transcript; leaving the messages alone")
        agent._emit("summary_failed", model="", error="splice would orphan a tool result")
        return 0
    messages[:] = spliced
    return max(0, before - estimate_message_tokens(summary))


def _valid_transcript(messages: list[dict[str, Any]]) -> bool:
    """Every tool reply answers a call in the message right before its run.

    The check the whole phase hangs on, so it runs on the result rather than being
    argued about in comments: a summary is worthless if the request it produces
    is rejected.
    """
    wanted: set[str] = set()
    answered: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id not in wanted or call_id in answered:
                return False                  # orphan, or a second reply to one call
            answered.add(call_id)
            continue
        if wanted != answered:
            return False                      # the round before this message is unfinished
        wanted, answered = set(), set()
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            wanted = {tc.get("id") for tc in msg["tool_calls"]}
    return wanted == answered                 # nothing dangling at the end


# ── Loop entry point ─────────────────────────────────────────────────────────


async def compress_if_needed(agent: Any, s: Any) -> None:
    """Phase 0 of each loop iteration: make the next request fit.

    Async because Phase B will call a model here; Phase A never awaits.
    Runs per *iteration*, not per turn — a single tool round can blow the window
    on its own, and the next request in that same turn is the one that fails.
    """
    s.rough_tokens = estimate_messages_tokens(s.messages)
    if not getattr(agent, "compress", True) or getattr(agent, "_compression_strikes", 0) >= STRIKE_LIMIT:
        return
    threshold = compression_threshold(agent)
    before = _calibrated(agent, s.messages)
    if before < threshold:
        return

    tail_tokens = int(threshold * TAIL_RATIO)
    durable_only = s.persist is not None
    # A: prune tool output. B: summarize the conversation. C: give up the tail.
    # Each one only runs because the one before it was not enough.
    reclaimed = reclaim(s.messages, tail_tokens=tail_tokens, durable_only=durable_only)["tokens"]
    if _calibrated(agent, s.messages) >= threshold:
        reclaimed += await summarize_middle(agent, s.messages, tail_tokens=tail_tokens,
                                            durable_only=durable_only)
    if _calibrated(agent, s.messages) >= threshold:
        reclaimed += reclaim(s.messages, tail_tokens=0, durable_only=durable_only,
                             pressure=True)["tokens"]

    s.rough_tokens = estimate_messages_tokens(s.messages)
    after = _calibrated(agent, s.messages)
    if not reclaimed:
        # Nothing anywhere: a single message larger than the window, or every
        # phase blocked at once. Report it and let the provider have its say.
        if not s.compression_exhausted:
            s.compression_exhausted = True
            logger.warning("Context at %d/%d tokens with nothing left to compress", before, threshold)
            agent._emit("context_full", tokens=before, threshold=threshold)
        return

    logger.info("Context compressed: %d → %d tokens (threshold %d)", before, after, threshold)
    agent._emit("context_compressed", before=before, after=after, threshold=threshold,
                reclaimed=reclaimed)
    _note_effectiveness(agent, before, after, threshold)


def _note_effectiveness(agent: Any, before: int, after: int, threshold: int) -> None:
    """Stop compressing a session that compression is not helping.

    Every pass rewrites the prompt prefix, which costs a cache miss and a cache
    write on the next request. Paying that for a 2% reclaim, once per iteration,
    is worse than simply letting the provider refuse: at least that surfaces.
    """
    if after < threshold or (before - after) >= before * MIN_RECLAIM_RATIO:
        agent._compression_strikes = 0
        return
    agent._compression_strikes = getattr(agent, "_compression_strikes", 0) + 1
    if agent._compression_strikes >= STRIKE_LIMIT:
        logger.warning("Compression is not reclaiming enough (%d → %d, threshold %d); "
                       "disabling it for this session", before, after, threshold)
        agent._emit("compression_disabled", tokens=after, threshold=threshold)


def _calibrated(agent: Any, messages: list[dict[str, Any]]) -> int:
    return int(estimate_messages_tokens(messages) * getattr(agent, "_token_scale", 1.0))
