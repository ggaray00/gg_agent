"""System prompts for the main agent and for delegated children.

Mirrors hermes-agent: agent/prompt_builder.py and
tools/delegate_tool_progress._build_child_system_prompt
"""

from __future__ import annotations

import os
import platform
from datetime import date

MAIN_SYSTEM_PROMPT = """You are gg-agent, a capable autonomous coding assistant.

You work by calling tools in a loop: think about what you need, call a tool,
read the result, and continue until the task is genuinely done. Only then give
a final answer.

Rules:
- Prefer acting over asking. Investigate with tools before you conclude.
- One tool call should have one clear purpose. Batch independent calls together.
- Never claim something works unless you verified it with a tool.
- When you are done, answer plainly: what you did, what you found, what is left.
"""

# Told to the child up front so its final message IS the deliverable the parent reads.
CHILD_COMPLETION_INSTRUCTIONS = """
Complete this task using the tools available to you. When finished, provide a clear,
concise summary of:
- What you did
- What you found or accomplished
- Any files you created or modified
- Any issues encountered

Keep your final summary tight: lead with outcomes, prefer bullet points over
paragraphs, and don't replay your whole process. Your response is returned to the
parent agent as a summary, and overlong summaries crowd out the parent's context.
"""

ORCHESTRATOR_BLOCK = """
## Subagent Spawning (Orchestrator Role)
You have access to the `delegate_task` tool and CAN spawn your own subagents to
parallelize independent work.

WHEN to delegate:
- The goal decomposes into 2+ independent subtasks that can run in parallel.
- A subtask is reasoning-heavy and would flood your context with intermediate data.

WHEN NOT to delegate:
- Single-step mechanical work — do it directly.
- Re-delegating your entire assigned goal to one worker (pass-through, no value added).

Coordinate your workers' results and synthesize them before reporting back to your
parent. You are responsible for the final summary, not your workers.
"""


def build_system_prompt(extra: str = "", cwd: str | None = None) -> str:
    """Main agent prompt + runtime facts the model would otherwise guess at."""
    parts = [MAIN_SYSTEM_PROMPT, "\n## Environment\n"
             f"- Date: {date.today().isoformat()}\n"
             f"- Platform: {platform.system()} {platform.release()}\n"
             f"- Working directory: {cwd or os.getcwd()}\n"]
    if extra and extra.strip():
        parts.append("\n" + extra.strip())
    return "".join(parts)


def build_child_system_prompt(goal: str, context: str | None = None, *,
                              workspace: str | None = None,
                              can_delegate: bool = False,
                              depth: int = 1, max_depth: int = 2) -> str:
    """Focused prompt for one subagent.

    The child knows NOTHING about the parent's conversation — everything it needs
    must arrive through ``goal`` and ``context``. That isolation is the point: the
    parent's context window never sees the child's intermediate work.
    """
    parts = ["You are a focused subagent working on a specific delegated task.",
             "", f"YOUR TASK:\n{goal}"]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context.strip()}")
    if workspace:
        parts.append(f"\nWORKSPACE PATH:\n{workspace}\n"
                     "Use this exact path for local repository/workdir operations "
                     "unless the task explicitly says otherwise.")
    parts.append(CHILD_COMPLETION_INSTRUCTIONS)
    if can_delegate:
        note = ("Your own children MUST be leaves (they are at the depth floor)."
                if depth + 1 >= max_depth else
                "Your own children can themselves delegate further.")
        parts.append(f"{ORCHESTRATOR_BLOCK}\nNOTE: You are at depth {depth}. "
                     f"The delegation tree is capped at max_depth={max_depth}. {note}")
    return "\n".join(parts)
