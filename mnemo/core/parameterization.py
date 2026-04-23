"""
Memory Parameterization — paper contribution C4.
Converts high-confidence patterns into natural language soft prompts.
"""

import json
import time
import uuid

from mnemo.storage.db import get_connection, transaction

MIN_CONFIDENCE = 0.7
MIN_EVIDENCE = 5
MAX_TOKENS = 1500
CHARS_PER_TOKEN = 4

TEMPLATES = {
    "prefers_language": (
        "Developer preference: Use {extracted_value} by default. "
        "Only suggest alternatives if explicitly asked."
    ),
    "naming_convention": (
        "Naming convention: This project uses {extracted_value}. "
        "Apply consistently to all new code."
    ),
    "architecture_decision": (
        "Architecture decision on record: {extracted_value}. "
        "Do not contradict without explicitly flagging the deviation."
    ),
    "avoided_pattern": (
        "Avoid: The developer has explicitly rejected {extracted_value}. "
        "Do not suggest this pattern."
    ),
    "project_context": ("Project context: {extracted_value}"),
}


def extract_value_from_facts(pattern_type: str, fact_texts: list[str]) -> str:
    """Heuristic extraction: shortest most specific fact as representative value."""
    sorted_facts = sorted(fact_texts, key=len)
    if sorted_facts:
        return sorted_facts[0][:80]
    return "unknown"


def generate_soft_prompts(project: str | None = None) -> str:
    """Generate the soft prompt block for session injection."""
    conn = get_connection()
    if project:
        # Patterns don't have a project column; consolidation scopes by project at creation time.
        # For now, return all qualifying patterns. TODO: add project column to patterns table.
        filter_clause = "WHERE confidence >= ? AND evidence_count >= ?"
    else:
        filter_clause = "WHERE confidence >= ? AND evidence_count >= ?"
    params: list = [MIN_CONFIDENCE, MIN_EVIDENCE]

    patterns = conn.execute(
        f"SELECT * FROM patterns {filter_clause} ORDER BY confidence DESC",
        params,
    ).fetchall()

    prompt_lines: list[str] = []
    total_chars = 0
    max_chars = MAX_TOKENS * CHARS_PER_TOKEN

    for pattern in patterns:
        template = TEMPLATES.get(pattern["pattern_type"])
        if not template:
            continue

        fact_ids = json.loads(pattern["source_fact_ids"] or "[]")[:5]
        if fact_ids:
            placeholders = ",".join("?" * len(fact_ids))
            fact_rows = conn.execute(
                f"SELECT text FROM facts WHERE id IN ({placeholders})", fact_ids
            ).fetchall()
            fact_texts = [r["text"] for r in fact_rows]
        else:
            fact_texts = [pattern["description"]]

        extracted = extract_value_from_facts(pattern["pattern_type"], fact_texts)
        rendered = template.format(extracted_value=extracted)

        if total_chars + len(rendered) > max_chars:
            break

        prompt_lines.append(rendered)
        total_chars += len(rendered) + 1

        with transaction(conn):
            conn.execute(
                """
                INSERT OR REPLACE INTO soft_prompts
                (id, pattern_id, template_key, rendered_text, token_estimate,
                 active, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, TRUE, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    pattern["id"],
                    pattern["pattern_type"],
                    rendered,
                    len(rendered) // CHARS_PER_TOKEN,
                    time.time(),
                    time.time(),
                ),
            )

    return "\n".join(prompt_lines)
