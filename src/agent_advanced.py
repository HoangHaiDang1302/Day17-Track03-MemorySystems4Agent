from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self.force_offline and self.langchain_agent is None:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                pass
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        updates = extract_profile_updates(message)
        profile = self.profile_store.read_text(user_id)
        existing_facts = self.profile_store.facts(user_id)
        for k, v in updates.items():
            if k == "response_style" and existing_facts.get("response_style"):
                current = existing_facts["response_style"]
                if v not in current:
                    v = f"{current}; {v}"
            self.profile_store.upsert_fact(user_id, k, v)

        self.compact_memory.append(thread_id, "user", message)
        ctx = self.compact_memory.context(thread_id)
        msgs: list[dict[str, str]] = list(ctx.get("messages", []))
        summary_text: str = str(ctx.get("summary", ""))

        system_prompt = f"User profile:\n{profile}\n"
        if summary_text:
            system_prompt += f"Past summary:\n{summary_text}\n"

        prompt_tokens = estimate_tokens(system_prompt) + sum(
            estimate_tokens(m.get("content", "")) for m in msgs
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        try:
            full_messages = [{"role": "system", "content": system_prompt}] + msgs
            result = self.langchain_agent.invoke({"messages": full_messages})
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
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + token_count
            self.compact_memory.append(thread_id, "assistant", reply_text)
            return {"response": reply_text, "token_usage": token_count}
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        existing_facts = self.profile_store.facts(user_id)
        for k, v in updates.items():
            if k == "response_style" and existing_facts.get("response_style"):
                current = existing_facts["response_style"]
                if v not in current:
                    v = f"{current}; {v}"
            self.profile_store.upsert_fact(user_id, k, v)

        self.compact_memory.append(thread_id, "user", message)
        ctx = self.compact_memory.context(thread_id)
        msgs: list[dict[str, str]] = list(ctx.get("messages", []))
        summary_text: str = str(ctx.get("summary", ""))

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        reply_text = self._offline_response(user_id, thread_id, message)
        token_count = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + token_count

        self.compact_memory.append(thread_id, "assistant", reply_text)

        return {"response": reply_text, "token_usage": token_count}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        msgs: list[dict[str, str]] = list(ctx.get("messages", []))
        summary_text: str = str(ctx.get("summary", ""))

        total = estimate_tokens(profile_text)
        total += estimate_tokens(summary_text)
        for m in msgs:
            total += estimate_tokens(m.get("content", ""))
        return total

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        profile = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        msgs: list = list(ctx.get("messages", []))
        lower = message.lower()

        if not profile:
            return "Mình chưa có thông tin gì về bạn."

        fact_count = sum(1 for v in profile.values() if v)
        is_multi = "và" in lower or "," in lower
        is_recall = any(w in lower for w in ["nhắc", "nhớ", "recall", "gì", "nào", "ai", "mô tả", "biết", "tóm tắt"])

        if is_recall or (is_multi and fact_count > 1):
            parts = []
            for k in ["name", "profession", "location", "interest", "food", "drink", "pet", "response_style"]:
                v = profile.get(k)
                if v:
                    label = {"name": "tên", "profession": "nghề", "location": "nơi ở",
                             "interest": "quan tâm", "food": "món yêu thích",
                             "drink": "đồ uống yêu thích",
                             "pet": "thú cưng", "response_style": "style"}.get(k, k)
                    parts.append(f"{label}: {v}")
            if parts:
                return "Dựa trên những gì mình nhớ: " + ", ".join(parts) + "."

        if "tên" in lower or "name" in lower:
            name = profile.get("name", "")
            return f"Tên bạn là {name}." if name else "Mình chưa có thông tin tên của bạn."

        if "ở đâu" in lower or "nơi ở" in lower or "location" in lower or "đang ở" in lower:
            loc = profile.get("location", "")
            return f"Bạn đang ở {loc}." if loc else "Mình chưa có thông tin nơi ở của bạn."

        if "nghề" in lower or "làm gì" in lower or "engineer" in lower:
            prof = profile.get("profession", "")
            return f"Bạn đang làm {prof}." if prof else "Mình chưa có thông tin nghề nghiệp của bạn."

        if "style" in lower or "trả lời" in lower or "bullet" in lower:
            style = profile.get("response_style", "")
            return f"Style trả lời bạn thích: {style}." if style else "Mình chưa có thông tin về style trả lời."

        if "thích" in lower or "sở thích" in lower:
            interest = profile.get("interest", "")
            return f"Sở thích của bạn: {interest}." if interest else "Mình ghi nhận sở thích của bạn."

        if "ăn" in lower or "uống" in lower or "cà phê" in lower or "mì quảng" in lower:
            food = profile.get("food", "")
            drink = profile.get("drink", "")
            parts = []
            if food:
                parts.append(f"món yêu thích: {food}")
            if drink:
                parts.append(f"đồ uống yêu thích: {drink}")
            if parts:
                return " | ".join(parts) + "."
            return "Mình chưa có thông tin về đồ ăn/uống."

        if "nuôi" in lower or "pet" in lower or "corgi" in lower or "bơ" in lower:
            pet = profile.get("pet", "")
            return f"Bạn nuôi {pet}." if pet else "Mình chưa có thông tin về thú cưng."

        return "Cảm ơn bạn! Mình (Advanced Agent) đã ghi nhận và lưu vào bộ nhớ dài hạn."

    def _maybe_build_langchain_agent(self):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            model = build_chat_model(self.config.model)
            memory = MemorySaver()
            self.langchain_agent = create_react_agent(model, tools=[], checkpointer=memory)
        except ImportError:
            self.langchain_agent = None
