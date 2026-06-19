from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self.force_offline and self.langchain_agent is None:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                pass
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self._ensure_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        try:
            result = self.langchain_agent.invoke({"messages": session.messages})
            reply_text = ""
            token_count = 0
            if hasattr(result, "content"):
                reply_text = result.content
            elif isinstance(result, dict):
                reply_text = result.get("output", "") or str(result.get("messages", [""])[-1])
            if hasattr(result, "usage_metadata"):
                token_count = result.usage_metadata.get("total_tokens", 0)
            if not token_count:
                token_count = estimate_tokens(reply_text)
            session.token_usage += token_count
            session.messages.append({"role": "assistant", "content": reply_text})
            return {"response": reply_text, "token_usage": token_count}
        except Exception:
            return self._reply_offline(thread_id, message)

    def _ensure_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def token_usage(self, thread_id: str) -> int:
        return self._ensure_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._ensure_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._ensure_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        lower = message.lower()
        if "tên" in lower or "name" in lower:
            reply_text = "Rất vui được gặp bạn! Mình là Baseline Agent, mình chỉ nhớ trong phiên này thôi."
        elif "ở đâu" in lower or "location" in lower or "nơi ở" in lower:
            reply_text = "Mình không có thông tin về nơi ở của bạn trong phiên này."
        elif "nghề" in lower or "làm gì" in lower or "job" in lower:
            reply_text = "Mình chưa có thông tin về nghề nghiệp của bạn."
        elif "thích" in lower or "sở thích" in lower or "preference" in lower:
            reply_text = "Mình ghi nhận sở thích của bạn trong phiên này."
        elif "nhắc" in lower or "nhớ" in lower or "recall" in lower:
            reply_text = "Mình chỉ nhớ được những gì bạn nói trong phiên trò chuyện này."
        else:
            reply_text = "Cảm ơn bạn! Mình đã nhận được tin nhắn của bạn."

        token_count = estimate_tokens(reply_text)
        session.token_usage += token_count
        session.messages.append({"role": "assistant", "content": reply_text})

        return {"response": reply_text, "token_usage": token_count}

    def _maybe_build_langchain_agent(self):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            model = build_chat_model(self.config.model)
            memory = MemorySaver()
            self.langchain_agent = create_react_agent(model, tools=[], checkpointer=memory)
        except ImportError:
            self.langchain_agent = None
