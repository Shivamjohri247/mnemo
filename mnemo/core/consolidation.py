"""
Consolidation: converts episodic facts into semantic patterns.
Pattern types mirror the TEMPLATES in parameterization.py.
"""

import json
import time
import uuid
from collections import defaultdict

from mnemo.storage.db import get_connection, transaction

PATTERN_KEYWORDS = {
    "naming_convention": [
        "naming",
        "convention",
        "snake_case",
        "camelCase",
        "PascalCase",
        "variable names",
        "function names",
        "class names",
    ],
    "avoided_pattern": [
        "avoid",
        "don't use",
        "never",
        "hate",
        "dislike",
        "banned",
        "prohibited",
    ],
    "architecture_decision": [
        "decided",
        "architecture",
        "we use",
        "the system uses",
        "pattern",
        "microservice",
        "monolith",
        "REST",
        "GraphQL",
        "PostgreSQL",
        "Redis",
    ],
    "project_context": [
        "this project",
        "the codebase",
        "our stack",
        "we are building",
        "the app",
    ],
    "prefers_language": [
        "prefer",
        "always use",
        "instead of",
        "not python",
        "not javascript",
        "go over",
        "rust over",
        "typescript",
    ],
}

# Classification priority: check most specific patterns first
_CLASSIFICATION_ORDER = [
    "naming_convention",
    "avoided_pattern",
    "architecture_decision",
    "project_context",
    "prefers_language",
]


def classify_fact(text: str) -> str | None:
    """Keyword-based classifier — checks most specific patterns first."""
    text_lower = text.lower()
    for pattern_type in _CLASSIFICATION_ORDER:
        keywords = PATTERN_KEYWORDS[pattern_type]
        if any(kw in text_lower for kw in keywords):
            return pattern_type
    return None


def run_consolidation_pass(project: str | None = None, min_evidence: int = 3) -> int:
    """
    Group facts by inferred pattern type.
    Facts with the same pattern type become a pattern if min_evidence met.
    """
    conn = get_connection()
    filter_clause = "AND project = ?" if project else ""
    params: list = [project] if project else []

    facts = conn.execute(
        f"""
        SELECT id, text, access_count, importance, confirmations
        FROM facts
        WHERE lifecycle IN ('Active', 'Warm')
        AND importance >= 0.4
        {filter_clause}
        """,
        params,
    ).fetchall()

    grouped: dict[str, list] = defaultdict(list)
    for fact in facts:
        ptype = classify_fact(fact["text"])
        if ptype:
            grouped[ptype].append(dict(fact))

    now = time.time()
    patterns_created = 0

    with transaction(conn):
        for ptype, evidence_facts in grouped.items():
            if len(evidence_facts) < min_evidence:
                continue

            n = len(evidence_facts)
            importances = [f["importance"] for f in evidence_facts]
            # positive_rate: fraction of evidence with importance above 0.5
            positive_rate = sum(1 for i in importances if i >= 0.5) / n
            # confidence: scales with evidence count and consistency
            # High confidence = lots of evidence + consistent signal
            consistency = abs(positive_rate - 0.3) / 0.7  # 0.3+ is considered consistent
            confidence = min(n / 5.0, 1.0) * min(consistency * 2, 1.0)

            pattern_id = str(uuid.uuid4())
            description = f"Pattern: {ptype} (evidence: {n} facts, confidence: {confidence:.2f})"
            fact_ids = json.dumps([f["id"] for f in evidence_facts])

            conn.execute(
                """
                INSERT OR REPLACE INTO patterns
                (id, pattern_type, description, evidence_count, positive_rate,
                 confidence, last_updated, source_fact_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pattern_id, ptype, description, n, positive_rate, confidence, now, fact_ids),
            )
            patterns_created += 1

    return patterns_created
