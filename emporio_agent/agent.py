"""Agent loop: Claude decides which tools to call; Python executes them deterministically.

A hand-written loop (instead of the SDK's beta tool runner) keeps every step explicit and
testable: the client is injected, so tests drive the loop with a scripted fake model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .config import Settings
from .data_store import DataStore
from .prompts import build_system_prompt
from .tools import ToolContext, execute_tool, tool_definitions

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side `fallbacks: "default"` (re-runs a safety-classifier refusal on another model).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

SERVICE_UNAVAILABLE_MESSAGE = (
    "Opa, estou com uma instabilidade aqui e não consegui consultar o sistema agora. 😕 "
    "Pode tentar de novo em instantes? Se preferir, fale com a equipe pelo WhatsApp (67) 3321-4500 "
    "(seg. a sex. 9h–18h, sáb. 9h–13h)."
)
REFUSAL_MESSAGE = (
    "Desculpa, não consigo ajudar com isso por aqui. Posso te ajudar com instrumentos, pedidos ou "
    "informações da loja?"
)


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: dict[str, Any]
    is_error: bool


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    failed: bool = False


class StoreAgent:
    def __init__(self, client: Any, settings: Settings, store: DataStore):
        self.client = client
        self.settings = settings
        self.tool_context = ToolContext(store=store, now=settings.now, runtime_dir=settings.runtime_dir)
        self.tools = tool_definitions(store)
        # Frozen per session so the cached prompt prefix (tools + system) stays byte-identical across turns.
        self.system = build_system_prompt(settings.now())
        self.messages: list[dict[str, Any]] = []

    def reply(self, user_text: str) -> AgentReply:
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        tool_calls: list[ToolCall] = []

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self._call_model()

                if response.stop_reason == "refusal":
                    del self.messages[checkpoint:]
                    return AgentReply(REFUSAL_MESSAGE, tool_calls)

                # Keep the full content (thinking + tool_use blocks); the API needs it on the next request.
                self.messages.append({"role": "assistant", "content": response.content})

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason == "tool_use" and tool_uses:
                    self.messages.append({"role": "user", "content": self._run_tools(tool_uses, tool_calls)})
                    continue
                if response.stop_reason == "pause_turn":
                    continue

                text = _text_of(response)
                if response.stop_reason == "max_tokens" or not text:
                    logger.warning("Unexpected end of turn: stop_reason=%s", response.stop_reason)
                    text = text or SERVICE_UNAVAILABLE_MESSAGE
                return AgentReply(text, tool_calls)

            logger.warning("Tool round limit reached")
            del self.messages[checkpoint:]
            return AgentReply(SERVICE_UNAVAILABLE_MESSAGE, tool_calls, failed=True)

        except anthropic.APIError as exc:
            # Roll back this turn so the history never ends on an unanswered user message or tool_use.
            logger.error("Claude API error: %s", exc)
            del self.messages[checkpoint:]
            return AgentReply(SERVICE_UNAVAILABLE_MESSAGE, tool_calls, failed=True)

    def _call_model(self):
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "tools": self.tools,
            "messages": self.messages,
            "output_config": {"effort": self.settings.effort},
            # Second breakpoint on the latest message: caches the growing conversation history too.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.model in FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return self.client.beta.messages.create(**kwargs)

    def _run_tools(self, tool_uses: list, tool_calls: list[ToolCall]) -> list[dict]:
        results = []
        for block in tool_uses:
            result, is_error = execute_tool(self.tool_context, block.name, block.input)
            tool_calls.append(ToolCall(block.name, block.input, result, is_error))
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": is_error,
            })
        return results  # all results in one user message, as the API expects for parallel calls


def _text_of(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
