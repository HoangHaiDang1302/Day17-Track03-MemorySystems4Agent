from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    lower = answer.lower()
    found = sum(1 for exp in expected if exp.lower() in lower)
    if not expected:
        return 1.0
    return found / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    lower = answer.lower()
    present = sum(1 for exp in expected if exp.lower() in lower)
    coverage = present / len(expected) if expected else 1.0
    length_ok = 1.0 if 10 <= len(answer) <= 500 else 0.5
    return round(0.7 * coverage + 0.3 * length_ok, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_memory_growth = 0
    total_compactions = 0
    recall_count = 0
    conv_count = 0

    for conv in conversations:
        conv_count += 1
        user_id = conv.get("user_id", "unknown")
        thread_id = conv.get("id", f"thread-{conv_count}")
        turns = conv.get("turns", [])
        questions = conv.get("recall_questions", [])

        start_memory = 0
        if hasattr(agent, "memory_file_size"):
            start_memory = agent.memory_file_size(user_id)

        for turn in turns:
            result = agent.reply(user_id, thread_id, turn)
            total_tokens += result.get("token_usage", 0)

        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)

        for q in questions:
            question_text = q.get("question", "")
            expected = q.get("expected_contains", [])
            recall_thread = f"{thread_id}_recall_{recall_count}"
            q_result = agent.reply(user_id, recall_thread, question_text)
            answer = q_result.get("response", "")

            recall = recall_points(answer, expected)
            quality = heuristic_quality(answer, expected)
            total_recall += recall
            total_quality += quality
            recall_count += 1

        if hasattr(agent, "memory_file_size"):
            end_memory = agent.memory_file_size(user_id)
            total_memory_growth += max(0, end_memory - start_memory)

    avg_recall = total_recall / recall_count if recall_count > 0 else 0.0
    avg_quality = total_quality / recall_count if recall_count > 0 else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=total_memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    if not rows:
        return "No data"
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    col_widths = [len(h) for h in headers]
    for row in rows:
        vals = [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            str(row.recall_score),
            str(row.response_quality),
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for i, v in enumerate(vals):
            col_widths[i] = max(col_widths[i], len(v))

    sep = " | ".join("-" * w for w in col_widths)
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines = [sep, header_line, sep]
    for row in rows:
        vals = [
            row.agent_name.ljust(col_widths[0]),
            str(row.agent_tokens_only).rjust(col_widths[1]),
            str(row.prompt_tokens_processed).rjust(col_widths[2]),
            str(row.recall_score).rjust(col_widths[3]),
            str(row.response_quality).rjust(col_widths[4]),
            str(row.memory_growth_bytes).rjust(col_widths[5]),
            str(row.compactions).rjust(col_widths[6]),
        ]
        lines.append(" | ".join(vals))
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    convs_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    if not convs_path.exists():
        print(f"Warning: {convs_path} not found, skipping standard benchmark.")
        convs = []
    else:
        convs = load_conversations(convs_path)

    if not stress_path.exists():
        print(f"Warning: {stress_path} not found, skipping stress benchmark.")
        stress = []
    else:
        stress = load_conversations(stress_path)

    if not convs and not stress:
        print("No benchmark data found.")
        return

    for dataset_name, dataset, is_stress in [
        ("Standard Benchmark", convs, False),
        ("Long-Context Stress Benchmark", stress, True),
    ]:
        if not dataset:
            continue

        print(f"\n{'='*60}")
        print(f"  {dataset_name}")
        print(f"{'='*60}")

        baseline = BaselineAgent(config=config, force_offline=True)
        advanced = AdvancedAgent(config=config, force_offline=True)

        baseline_row = run_agent_benchmark("Baseline", baseline, dataset, config)
        advanced_row = run_agent_benchmark("Advanced", advanced, dataset, config)

        print(format_rows([baseline_row, advanced_row]))


if __name__ == "__main__":
    main()
