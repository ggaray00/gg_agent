"""Delegation — spawn subagents with isolated context.

The whole idea in one paragraph: a child is a brand-new Agent with its own
conversation, its own tool grant, and a system prompt built from goal + context.
The parent never sees the child's intermediate tool calls or reasoning — only
the final summary. That's what keeps a 40-step research subtask from eating the
parent's context window.

Three things make it safe rather than a fork bomb:
  * depth cap — a leaf child does not get ``delegate_task`` back,
  * concurrency cap — a bounded semaphore, not unbounded fan-out,
  * timeout — a wedged child returns a failure entry instead of hanging the parent.

Children run as sibling tasks on the parent's event loop. That is the whole
reason the core is async: N subagents, each waiting on its own API call and its
own tools, cost N tasks rather than N threads, and cancelling the parent
cancels the subtree in one move.

Mirrors hermes-agent: tools/delegate_tool*.py (7 modules there; the production
version adds live steering, heartbeats, worktree isolation, output schemas,
result-size budgets and per-child credential leasing).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..prompts import build_child_system_prompt
from .registry import registry

logger = logging.getLogger(__name__)

MAX_CONCURRENT_CHILDREN = 4
CHILD_TIMEOUT_SECONDS = 600
CHILD_MAX_ITERATIONS = 40
MAX_SUMMARY_CHARS = 8_000


def _build_child(parent_agent, task: dict[str, Any], index: int):
    """Build (don't run) one child agent on the calling thread."""
    from ..agent import DELEGATE_BLOCKED_TOOLS, SUBAGENT_BLOCKED_TOOLS, Agent

    child_depth = parent_agent.depth + 1
    # A child may delegate only while there is depth left beneath it. Capability is
    # depth-derived — the model never gets to ask for it.
    can_delegate = child_depth < parent_agent.max_depth
    blocked = set(parent_agent.blocked_tools) | SUBAGENT_BLOCKED_TOOLS
    if not can_delegate:
        blocked |= DELEGATE_BLOCKED_TOOLS

    child = Agent(
        provider=parent_agent.profile.name,
        model=task.get("model") or parent_agent.model,
        api_key=parent_agent.api_key,
        base_url=parent_agent.base_url,
        system_prompt=build_child_system_prompt(
            goal=task["goal"],
            context=task.get("context"),
            workspace=parent_agent.cwd,
            can_delegate=can_delegate,
            depth=child_depth,
            max_depth=parent_agent.max_depth,
        ),
        enabled_toolsets=parent_agent.enabled_toolsets,
        blocked_tools=blocked,
        max_iterations=CHILD_MAX_ITERATIONS,
        max_tokens=parent_agent.max_tokens,
        temperature=parent_agent.temperature,
        cwd=parent_agent.cwd,
        event_callback=parent_agent.event_callback,
        depth=child_depth,
        max_depth=parent_agent.max_depth,
        parent=parent_agent,
        # Same store (and pool) as the parent; the parent link supplies
        # parent_session_id, so a child transcript is traceable but hidden by default.
        store=parent_agent.store if parent_agent.store is not None else False,
        source="subagent",
    )
    child._task_index = index
    return child


async def _run_child(child, task: dict[str, Any], index: int) -> dict[str, Any]:
    """Run one child to completion and reduce it to a result entry."""
    started = time.time()
    try:
        # The goal is already in the system prompt; the user turn just starts it.
        result = await child.arun("Begin working on your assigned task now.")
        summary = (result.get("response") or "").strip()
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS] + "\n… [summary truncated]"
        return {
            "task_index": index,
            "goal": task["goal"],
            "status": "failed" if result.get("failed") else "completed",
            "summary": summary or None,
            "error": result.get("response") if result.get("failed") else None,
            "api_calls": result.get("api_calls", 0),
            "tool_calls": result.get("tool_calls", 0),
            "duration_seconds": round(time.time() - started, 2),
        }
    except asyncio.TimeoutError:
        child.interrupt()
        return {
            "task_index": index, "goal": task["goal"], "status": "timeout",
            "summary": None,
            "error": f"subagent exceeded {CHILD_TIMEOUT_SECONDS}s and was stopped",
            "duration_seconds": round(time.time() - started, 2),
        }
    except asyncio.CancelledError:
        # The parent was interrupted. Report it, don't swallow the cancellation.
        child.interrupt()
        raise
    except Exception as exc:
        logger.exception("subagent %d crashed", index)
        return {
            "task_index": index, "goal": task["goal"], "status": "error",
            "summary": None, "error": str(exc),
            "duration_seconds": round(time.time() - started, 2),
        }
    finally:
        try:
            await child.aclose()
        except Exception:
            logger.debug("child close failed", exc_info=True)


def _normalize_tasks(tasks: Any, goal: Any, context: Any) -> list[dict[str, Any]] | str:
    """Accept either the batch shape (``tasks=[...]``) or a single ``goal``."""
    if not tasks and goal:
        tasks = [{"goal": goal, "context": context}]
    if not isinstance(tasks, list) or not tasks:
        return "Provide either `tasks` (a list of {goal, context}) or a single `goal`."
    clean: list[dict[str, Any]] = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or not str(task.get("goal") or "").strip():
            return f"tasks[{i}] needs a non-empty `goal`."
        clean.append(task)
    if len(clean) > MAX_CONCURRENT_CHILDREN:
        return (f"Too many tasks ({len(clean)}); the limit is {MAX_CONCURRENT_CHILDREN} "
                "per call. Split the work across sequential calls.")
    return clean


async def delegate_task(goal: str | None = None, context: str | None = None,
                        tasks: list | None = None, parent_agent: Any = None) -> dict[str, Any]:
    """Spawn one or more subagents and return their summaries."""
    if parent_agent is None:
        return {"error": "delegate_task requires a parent agent context."}
    if not parent_agent.can_delegate():
        return {"error": f"Delegation depth cap reached (depth={parent_agent.depth}, "
                         f"max_depth={parent_agent.max_depth}). Do this work yourself."}

    normalized = _normalize_tasks(tasks, goal, context)
    if isinstance(normalized, str):
        return {"error": normalized}

    children = [_build_child(parent_agent, task, i) for i, task in enumerate(normalized)]
    parent_agent._children.extend(children)
    parent_agent._emit("delegate_start", count=len(children),
                       goals=[t["goal"][:80] for t in normalized])
    started = time.time()

    # Per-child timeout rather than one deadline for the batch: a single wedged
    # child no longer costs the results of its siblings.
    gate = asyncio.Semaphore(MAX_CONCURRENT_CHILDREN)

    async def guarded(child, task, index):
        async with gate:
            return await asyncio.wait_for(
                _run_child(child, task, index), timeout=CHILD_TIMEOUT_SECONDS)

    try:
        entries: list[dict[str, Any]] = list(await asyncio.gather(*(
            guarded(child, task, i)
            for i, (child, task) in enumerate(zip(children, normalized, strict=True))
        )))
    finally:
        for child in children:
            try:
                parent_agent._children.remove(child)
            except ValueError:
                pass

    entries.sort(key=lambda e: e["task_index"])
    parent_agent._emit("delegate_end", count=len(entries),
                       duration=round(time.time() - started, 2))
    return {
        "subagents": len(entries),
        "completed": sum(1 for e in entries if e["status"] == "completed"),
        "duration_seconds": round(time.time() - started, 2),
        "results": entries,
    }


async def _handler(**kw) -> dict[str, Any]:
    return await delegate_task(
        goal=kw.get("goal"), context=kw.get("context"),
        tasks=kw.get("tasks"), parent_agent=kw.get("parent_agent"),
    )


registry.register_toolset("delegation", "Spawn subagents with isolated context")
registry.register(
    name="delegate_task",
    toolset="delegation",
    emoji="🔀",
    needs_agent=True,          # the handler receives parent_agent=<Agent>
    handler=_handler,
    schema={
        "name": "delegate_task",
        "description": (
            "Spawn one or more subagents that work in ISOLATED contexts and report back a "
            "summary. Tasks in one call run in parallel.\n\n"
            "Use it when the work decomposes into independent subtasks, or when a subtask "
            "would flood your context with intermediate data you don't need to keep.\n"
            "Do NOT use it for single-step mechanical work you can do in one or two tool "
            "calls, and never re-delegate your whole goal to one child.\n\n"
            "Each child knows NOTHING about this conversation — put everything it needs "
            f"into its goal and context. Limit: {MAX_CONCURRENT_CHILDREN} tasks per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "description": "The subtasks to run in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "What this subagent should accomplish. Specific and self-contained.",
                            },
                            "context": {
                                "type": "string",
                                "description": ("Background THIS child needs: file paths, error messages, "
                                                "constraints. Each child sees only its own context — repeat "
                                                "shared background in every task that needs it."),
                            },
                        },
                        "required": ["goal"],
                    },
                },
                "goal": {"type": "string", "description": "Shorthand for a single task (instead of `tasks`)."},
                "context": {"type": "string", "description": "Background for the single-task form."},
            },
            "required": [],
        },
    },
)
