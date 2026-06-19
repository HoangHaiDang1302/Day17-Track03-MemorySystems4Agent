from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,
        compact_keep_messages=3,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.7),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = AdvancedAgent(config=config, force_offline=True).profile_store

    store.write_text("testuser", "# Profile: testuser\n- **name**: Alice\n")
    assert store.read_text("testuser") == "# Profile: testuser\n- **name**: Alice\n"

    store.upsert_fact("testuser", "location", "Hanoi")
    content = store.read_text("testuser")
    assert "**name**: Alice" in content
    assert "**location**: Hanoi" in content

    result = store.edit_text("testuser", "Alice", "Bob")
    assert result is True
    assert "Bob" in store.read_text("testuser")
    assert "Alice" not in store.read_text("testuser")

    size = store.file_size("testuser")
    assert size > 0


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    for i in range(20):
        agent.reply("user1", "thread1", f"Message number {i} with some extra padding text here to accumulate tokens quickly for testing purposes.")

    assert agent.compaction_count("thread1") > 0, "Compaction should have been triggered"

    ctx = agent.compact_memory.context("thread1")
    msgs = ctx.get("messages", [])
    assert isinstance(msgs, list)
    assert len(msgs) <= config.compact_keep_messages
    summary = ctx.get("summary", "")
    assert summary and len(str(summary)) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    r1 = baseline.reply("user1", "thread_a", "Mình tên là DũngCT và mình ở Đà Nẵng.")
    r2 = baseline.reply("user1", "thread_b", "Mình tên gì và ở đâu?")
    assert "DũngCT" not in r2.get("response", "") or "Đà Nẵng" not in r2.get("response", "")

    r3 = advanced.reply("user1", "thread_c", "Mình tên là DũngCT và mình ở Đà Nẵng.")
    r4 = advanced.reply("user1", "thread_d", "Mình tên gì và ở đâu?")
    resp4 = r4.get("response", "")
    assert "DũngCT" in resp4, f"Expected 'DũngCT' in response but got: {resp4}"
    assert "Đà Nẵng" in resp4, f"Expected 'Đà Nẵng' in response but got: {resp4}"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    for i in range(30):
        msg = f"Turn number {i} with a reasonably long message body to accumulate tokens. " * 3
        baseline.reply("user1", "long_thread", msg)
        advanced.reply("user1", "long_thread", msg)

    baseline_prompt = baseline.prompt_token_usage("long_thread")
    advanced_prompt = advanced.prompt_token_usage("long_thread")

    assert advanced.compaction_count("long_thread") > 0, "Advanced should have compacted"

    print(f"\n  Baseline prompt tokens: {baseline_prompt}")
    print(f"  Advanced prompt tokens: {advanced_prompt}")
    print(f"  Compactions: {advanced.compaction_count('long_thread')}")
