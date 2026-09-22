"""OpenAI-client-shaped shim over the Copilot CLI's ACP mode (``copilot --acp --stdio``).

Each request starts a short-lived ACP session over stdio JSON-RPC, sends the
conversation as ONE text prompt, collects the streamed text, and returns a
minimal Chat Completions-shaped object the ``chat_completions`` transport can
normalize. ACP has no function-calling channel, so tool schemas travel INTO the
prompt as text and calls come back OUT as ``<tool_call>{...}</tool_call>`` blocks.

The CLI may ask to read or write files; reads are served inside the working
directory, writes and permission requests are refused (there is no human in
this channel — the agent's own tools are the way to change files).

Mirrors hermes-agent: agent/copilot_acp_client.py, agent/acp_openai_bridge.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 900.0
_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
_PREAMBLE = (
    "You are the model backend for an agent harness, reached over ACP.",
    "IMPORTANT: to use a tool, output ONLY <tool_call>{...}</tool_call> blocks, each holding one JSON "
    'object {"id": "...", "type": "function", "function": {"name": "...", "arguments": "<JSON string>"}}.',
    "If no tool is needed, answer normally.",
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_INITIALIZE_PARAMS = {
    "protocolVersion": 1,
    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": False}},
    "clientInfo": {"name": "gg-agent", "title": "gg-agent", "version": "0.1.0"},
}


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(p if isinstance(p, str) else str(p.get("text") or "")
                         for p in content if isinstance(p, (str, dict))).strip()
    return str(content).strip()


def format_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    sections = list(_PREAMBLE)
    specs = [t.get("function", t) for t in tools or [] if isinstance(t, dict)]
    if specs:
        sections.append("Available tools (OpenAI function schema):\n" + json.dumps(specs, ensure_ascii=False))
    transcript = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        body = _render_content(msg.get("content"))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            body += f"\n<tool_call>{json.dumps({'id': tc.get('id'), 'type': 'function', 'function': fn})}</tool_call>"
        if role == "tool":
            body = f"[result of {msg.get('tool_call_id')}]\n{body}"
        if body.strip():
            transcript.append(f"{_ROLE_LABELS.get(role, 'Context')}:\n{body.strip()}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(sections)


def extract_tool_calls(text: str) -> tuple[list[Any], str]:
    """``<tool_call>`` blocks -> OpenAI-shaped tool calls, plus the text without them."""
    calls = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text or ""):
        try:
            obj = json.loads(match.group(1))
        except ValueError:
            continue
        fn = obj.get("function") if isinstance(obj, dict) else None
        if not isinstance(fn, dict) or not str(fn.get("name") or "").strip():
            continue
        args = fn.get("arguments", "{}")
        calls.append(SimpleNamespace(
            id=str(obj.get("id") or f"acp_call_{len(calls) + 1}"), type="function",
            function=SimpleNamespace(name=fn["name"].strip(),
                                     arguments=args if isinstance(args, str) else json.dumps(args))))
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", text or "").strip() if calls else (text or "").strip()
    return calls, cleaned


class CopilotACPClient:
    """Just enough of ``openai.AsyncOpenAI`` for the chat_completions transport."""

    def __init__(self, *, command: str = "copilot", args: list[str] | None = None, cwd: str | None = None) -> None:
        self.command = command
        self.args = list(args if args is not None else ["--acp", "--stdio"])
        self.cwd = str(Path(cwd or os.getcwd()).resolve())
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list_models)
        self._proc: subprocess.Popen | None = None

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()

    async def _list_models(self) -> Any:
        return SimpleNamespace(data=[])

    async def _create(self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
                      tools: list[dict[str, Any]] | None = None, **_: Any) -> Any:
        prompt = format_prompt(messages or [], tools)
        text, reasoning = await asyncio.to_thread(self._run_prompt, prompt, model)
        tool_calls, content = extract_tool_calls(text)
        message = SimpleNamespace(content=content or None, tool_calls=tool_calls or None,
                                  reasoning_content=reasoning or None, reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=None, model=model or "copilot-acp")

    # ── The ACP session (blocking; runs in a worker thread) ──────────────

    def _run_prompt(self, prompt: str, model: str | None) -> tuple[str, str]:
        try:
            proc = subprocess.Popen(
                [self.command, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=self.cwd)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Could not start `{self.command}`. Install the GitHub Copilot CLI "
                               "(npm install -g @github/copilot) or set GG_COPILOT_ACP_COMMAND.") from exc
        self._proc = proc
        inbox: list[dict[str, Any]] = []
        cond = threading.Condition()
        stderr_tail: deque[str] = deque(maxlen=40)

        def pump_stdout() -> None:
            for line in proc.stdout or ():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                with cond:
                    inbox.append(msg)
                    cond.notify_all()

        def pump_stderr() -> None:
            for line in proc.stderr or ():
                stderr_tail.append(line.rstrip("\n"))

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()
        ids = iter(range(1, 1 << 62))
        text_parts: list[str] = []
        thought_parts: list[str] = []

        def send(obj: dict[str, Any]) -> None:
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        def request(method: str, params: dict[str, Any]) -> Any:
            request_id = next(ids)
            send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            deadline = time.monotonic() + _TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                with cond:
                    if not inbox:
                        cond.wait(timeout=0.2)
                    pending, inbox[:] = list(inbox), []
                for msg in pending:
                    if self._handle_server_message(msg, send, text_parts, thought_parts):
                        continue
                    if msg.get("id") == request_id:
                        if "error" in msg:
                            err = msg["error"] or {}
                            raise RuntimeError(f"Copilot ACP {method} failed: {err.get('message') or err}")
                        return msg.get("result")
                if proc.poll() is not None and not inbox:
                    tail = "\n".join(stderr_tail).strip()
                    raise RuntimeError(f"Copilot ACP process exited early: {tail or proc.returncode}")
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}")

        try:
            request("initialize", _INITIALIZE_PARAMS)
            session = request("session/new", {"cwd": self.cwd, "mcpServers": []}) or {}
            session_id = str(session.get("sessionId") or "")
            if not session_id:
                raise RuntimeError("Copilot ACP did not return a sessionId")
            if model and model != "copilot-acp":
                try:
                    request("session/set_model", {"sessionId": session_id, "modelId": model})
                except Exception as exc:
                    logger.warning("Copilot ACP: model %r not selectable, using the session default: %s", model, exc)
            request("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]})
            return "".join(text_parts), "".join(thought_parts)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._proc = None

    def _handle_server_message(self, msg: dict[str, Any], send, text_parts: list[str],
                               thought_parts: list[str]) -> bool:
        """Consume a server->client notification or request; True when handled."""
        method = msg.get("method")
        if not isinstance(method, str):
            return False
        if method == "session/update":
            update = (msg.get("params") or {}).get("update") or {}
            content = update.get("content") or {}
            chunk = str(content.get("text") or "") if isinstance(content, dict) else ""
            kind = update.get("sessionUpdate")
            if chunk and kind == "agent_message_chunk":
                text_parts.append(chunk)
            elif chunk and kind == "agent_thought_chunk":
                thought_parts.append(chunk)
            return True
        message_id = msg.get("id")
        if message_id is None:
            return True                      # some other notification
        if method == "session/request_permission":
            send({"jsonrpc": "2.0", "id": message_id, "result": {"outcome": {"outcome": "cancelled"}}})
        elif method == "fs/read_text_file":
            try:
                send({"jsonrpc": "2.0", "id": message_id, "result": self._read_file(msg.get("params") or {})})
            except Exception as exc:
                send({"jsonrpc": "2.0", "id": message_id, "error": {"code": -32602, "message": str(exc)}})
        else:
            send({"jsonrpc": "2.0", "id": message_id,
                  "error": {"code": -32601, "message": f"{method} is not supported by gg-agent"}})
        return True

    def _read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(params.get("path") or ""))
        if not path.is_absolute():
            raise PermissionError("ACP file paths must be absolute")
        resolved = path.resolve()
        resolved.relative_to(Path(self.cwd))          # ValueError outside the working directory
        content = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
        line, limit = params.get("line"), params.get("limit")
        if isinstance(line, int) and line > 1:
            end = line - 1 + limit if isinstance(limit, int) and limit > 0 else None
            content = "".join(content.splitlines(keepends=True)[line - 1:end])
        return {"content": content}
