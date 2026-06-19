from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return f"# Profile: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        p.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"^-\s*\*\*([^*]+)\*\*:\s*(.+)", line)
            if m:
                facts[m.group(1).strip()] = m.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        existing = self.read_text(user_id)
        pattern = rf"^-\s*\*\*{re.escape(key)}\*\*:.*$"
        if re.search(pattern, existing, re.MULTILINE):
            return self.edit_text(
                user_id,
                re.search(pattern, existing, re.MULTILINE).group(),
                f"- **{key}**: {value}",
            )
        new_entry = f"- **{key}**: {value}\n"
        self.write_text(user_id, existing.rstrip() + "\n" + new_entry)
        return True


def _clean_value(val: str) -> str:
    return val.strip().rstrip(".,;:!?").strip()


_STOP = r"(?:\s+chứ|\s+nhưng|\s+nhé|\s+và\s|,|\.|$)"
_LOC_STOP = r"(?:\s+chứ|\s+nhưng|\s+nhé|\s+và\s|\s+vài|\s+để\s|\s+cho\s|,|\.|$)"
_NAME_STOP = r"(?:\s+và\s|,|\.|$)"


def extract_profile_updates(message: str) -> dict[str, str]:
    facts: dict[str, str] = {}

    stripped = message.strip()

    if stripped.endswith("?"):
        return facts

    if re.search(r"(?:nhớ lại|thử nhớ|nhắc lại|nhắc giúp|nhắc lại giúp|của mình là gì|là gì nhỉ|là gì vậy)", stripped, re.IGNORECASE):
        return facts

    neg_prof = re.search(r"không còn làm\s+(.+?)\s+nữa", stripped, re.IGNORECASE)

    name_m = re.search(r"(?:mình tên là|tên(?: mình)? là|có tên là)\s+(.+?)" + _NAME_STOP, stripped, re.IGNORECASE)
    if name_m:
        facts["name"] = _clean_value(name_m.group(1))

    loc_m = re.search(
        r"(?:mình(?: đang)? (?:làm việc\s+)?ở|mình sống(?: ở)? tại|chuyển(?: đến|dọn đến))\s+(.+?)" + _LOC_STOP,
        stripped, re.IGNORECASE,
    )
    if loc_m:
        loc = _clean_value(loc_m.group(1))
        if loc and "hiện tại" not in loc.lower() and len(loc) < 50:
            facts["location"] = loc

    prof_m = re.search(
        r"(?:chuyển sang|đổi sang)\s+(?:làm\s+)?(.+?)" + _STOP,
        stripped, re.IGNORECASE,
    )
    if not prof_m:
        prof_m = re.search(
            r"mình(?: đang)? làm\s+(?:một\s+)?(.+?)(?:\s+cho|\s+tại)" + _STOP,
            stripped, re.IGNORECASE,
        )
    if not prof_m and not neg_prof:
        prof_m = re.search(
            r"(?:hiện là|hiện tại là|đang là|vẫn là|nghề nghiệp[^.]*?là)\s+(.+?)" + _STOP,
            stripped, re.IGNORECASE,
        )
    if prof_m:
        prof = _clean_value(prof_m.group(1))
        if prof and len(prof) < 60:
            facts["profession"] = prof

    drink_m = re.search(
        r"(?:đồ uống|thức uống)(?:\s+yêu thích)?(?:\s+ưa thích)?\s+là\s+(.+?)" + _STOP,
        stripped, re.IGNORECASE,
    )
    if drink_m:
        drink = _clean_value(drink_m.group(1))
        if drink and len(drink) < 50:
            facts["drink"] = drink
        return facts

    food_m = re.search(
        r"món ăn(?:\s+yêu thích)?(?:\s+ưa thích)?\s+là\s+(.+?)" + _STOP,
        stripped, re.IGNORECASE,
    )
    if food_m:
        food = _clean_value(food_m.group(1))
        if food and len(food) < 50:
            facts["food"] = food
        return facts

    interest_m = re.search(
        r"mình thích\s+(.+?)(?:\.|$)",
        stripped, re.IGNORECASE,
    )
    if interest_m:
        interest = _clean_value(interest_m.group(1))
        if interest and len(interest) < 100:
            facts["interest"] = interest

    style_m = re.search(
        r"(?:muốn bạn trả lời|thích bạn trả lời|muốn câu trả lời|thích câu trả lời)\s+(.+?)" + _STOP,
        stripped, re.IGNORECASE,
    )
    if not style_m:
        style_m2 = re.search(r"trả lời\s+(ngắn gòn[^,\.]*)", stripped, re.IGNORECASE)
        if style_m2:
            facts["response_style"] = _clean_value(style_m2.group(1))
    else:
        val = _clean_value(style_m.group(1))
        if "ngắn gọn" in val or "bullet" in val.lower() or len(val) > 5:
            facts["response_style"] = val

    pet_m = re.search(
        r"nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(\w+)\s+tên\s+(\w+)",
        stripped, re.IGNORECASE,
    )
    if pet_m:
        animal = _clean_value(pet_m.group(1))
        pet_name = _clean_value(pet_m.group(2))
        facts["pet"] = f"{animal} tên {pet_name}"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    important = messages[:max_items]
    summary_parts: list[str] = []
    for msg in important:
        role_label = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        truncated = content[:200] + "..." if len(content) > 200 else content
        summary_parts.append(f"[{role_label}]: {truncated}")
    return "\n".join(summary_parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]  # type: ignore

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure(thread_id)
        msgs: list[dict[str, str]] = thread["messages"]  # type: ignore
        msgs.append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    def _maybe_compact(self, thread_id: str) -> None:
        thread = self._ensure(thread_id)
        msgs: list[dict[str, str]] = thread["messages"]  # type: ignore
        total = sum(estimate_tokens(m.get("content", "")) for m in msgs)

        if total > self.threshold_tokens and len(msgs) > self.keep_messages:
            old = msgs[: -self.keep_messages]
            current = msgs[-self.keep_messages :]

            old_summary = summarize_messages(old, max_items=4)
            existing_summary: str = thread.get("summary", "") or ""
            if existing_summary:
                combined = existing_summary + "\n" + old_summary
            else:
                combined = old_summary

            thread["summary"] = combined
            thread["messages"] = current
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        thread = self._ensure(thread_id)
        return int(thread.get("compactions", 0))
