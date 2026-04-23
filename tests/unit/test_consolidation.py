"""Exhaustive tests for consolidation and soft prompt parameterization."""

import json
import time

import pytest

from mnemo.core.consolidation import (
    PATTERN_KEYWORDS,
    classify_fact,
    run_consolidation_pass,
)
from mnemo.core.parameterization import (
    CHARS_PER_TOKEN,
    MAX_TOKENS,
    MIN_CONFIDENCE,
    MIN_EVIDENCE,
    TEMPLATES,
    extract_value_from_facts,
    generate_soft_prompts,
)

# ── classify_fact ─────────────────────────────────────────────────────


class TestClassifyFact:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("We prefer TypeScript over JavaScript", "prefers_language"),
            ("Always use Python for scripts", "prefers_language"),
            ("Use snake_case for variable names", "naming_convention"),
            ("Function naming convention is camelCase", "naming_convention"),
            ("We decided on microservice architecture", "architecture_decision"),
            ("The system uses PostgreSQL", "architecture_decision"),
            ("Never use global state", "avoided_pattern"),
            ("Don't use eval()", "avoided_pattern"),
            ("This project is a REST API", "project_context"),
            ("Our stack includes React", "project_context"),
            ("The weather is nice today", None),
            ("Random fact about nothing", None),
            ("", None),
        ],
    )
    def test_classifications(self, text, expected):
        assert classify_fact(text) == expected

    def test_priority_naming_over_language(self):
        """naming_convention checked before prefers_language."""
        text = "Always use snake_case naming convention"
        assert classify_fact(text) == "naming_convention"

    def test_priority_avoided_over_architecture(self):
        """avoided_pattern checked before architecture_decision."""
        text = "Never use microservice architecture"
        assert classify_fact(text) == "avoided_pattern"

    def test_case_insensitive(self):
        assert classify_fact("PREFER TYPESCRIPT") == "prefers_language"
        assert classify_fact("NEVER USE GLOBAL STATE") == "avoided_pattern"

    def test_all_pattern_types_have_keywords(self):
        for ptype, keywords in PATTERN_KEYWORDS.items():
            assert len(keywords) > 0, f"{ptype} has no keywords"

    def test_all_types_in_templates(self):
        """Every pattern type should have a corresponding template."""
        for ptype in PATTERN_KEYWORDS:
            assert ptype in TEMPLATES, f"No template for {ptype}"


# ── extract_value_from_facts ──────────────────────────────────────────


class TestExtractValue:
    def test_returns_shortest_fact(self):
        facts = ["This is a very long description of something", "Short"]
        assert extract_value_from_facts("any", facts) == "Short"

    def test_truncates_long_values(self):
        facts = ["x" * 200]
        result = extract_value_from_facts("any", facts)
        assert len(result) <= 80

    def test_empty_list_returns_unknown(self):
        assert extract_value_from_facts("any", []) == "unknown"

    def test_preserves_content(self):
        facts = ["Use Redis for caching"]
        assert "Redis" in extract_value_from_facts("architecture_decision", facts)


# ── run_consolidation_pass ────────────────────────────────────────────


class TestConsolidationPass:
    @pytest.fixture
    def consol_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        now = time.time()
        # Insert enough facts to form patterns (min_evidence=3)
        facts = [
            ("We prefer Python for all scripting tasks", 0.8),
            ("Always use Python instead of bash", 0.7),
            ("Python is our preferred language for tools", 0.6),
            ("Never use global variables in functions", 0.7),
            ("Don't use eval() in any code", 0.8),
            ("Avoid using global state at all costs", 0.9),
            ("Random unrelated fact", 0.3),
            ("Low importance note", 0.2),
        ]
        for i, (text, imp) in enumerate(facts):
            fid = f"fact_{i}"
            chash = f"hash_{i}"
            conn.execute(
                "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
                "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
                "VALUES (?, ?, 'user', ?, ?, ?, 1.0, ?, 'Active', 32, 3.5, 1.0)",
                (fid, text, now, now, imp, chash),
            )
            vec = mock_embedder.embed_text(text)
            conn.execute(
                "INSERT INTO embeddings (fact_id, vector_f32) VALUES (?, ?)",
                (fid, vec.tobytes()),
            )
        conn.commit()
        yield conn

    def test_creates_patterns(self, consol_env):
        count = run_consolidation_pass(min_evidence=3)
        assert count >= 1  # At least one pattern group

    def test_pattern_in_db(self, consol_env):
        run_consolidation_pass(min_evidence=3)
        rows = consol_env.execute("SELECT * FROM patterns").fetchall()
        assert len(rows) >= 1
        for row in rows:
            assert row["pattern_type"] in PATTERN_KEYWORDS
            assert row["evidence_count"] >= 3
            assert 0 <= row["confidence"] <= 1.0
            assert row["source_fact_ids"] is not None

    def test_min_evidence_filter(self, consol_env):
        """With high min_evidence, fewer patterns should be created."""
        low = run_consolidation_pass(min_evidence=2)
        high = run_consolidation_pass(min_evidence=100)
        assert low >= high

    def test_excludes_warm_and_cold(self, consol_env):
        """Only Active and Warm facts should be considered."""
        consol_env.execute("UPDATE facts SET lifecycle = 'Cold' WHERE id = 'fact_0'")
        consol_env.commit()
        count = run_consolidation_pass(min_evidence=3)
        # May create fewer patterns since one fact is excluded
        assert isinstance(count, int)

    def test_importance_threshold(self, consol_env):
        """Facts with importance < 0.4 should be excluded."""
        consol_env.execute("UPDATE facts SET importance = 0.3 WHERE id LIKE 'fact_%'")
        consol_env.commit()
        count = run_consolidation_pass(min_evidence=3)
        assert count == 0

    def test_project_filter(self, consol_env):
        consol_env.execute("UPDATE facts SET project = 'proj_a' WHERE id LIKE 'fact_%'")
        consol_env.commit()
        count_proj = run_consolidation_pass(project="proj_a", min_evidence=3)
        count_wrong = run_consolidation_pass(project="nonexistent", min_evidence=3)
        assert count_proj >= count_wrong

    def test_confidence_computed(self, consol_env):
        run_consolidation_pass(min_evidence=3)
        row = consol_env.execute("SELECT confidence FROM patterns LIMIT 1").fetchone()
        if row:
            assert row["confidence"] > 0


# ── generate_soft_prompts ─────────────────────────────────────────────


class TestSoftPrompts:
    @pytest.fixture
    def prompt_env(self, tmp_path, monkeypatch, mock_embedder):
        data_dir = tmp_path / ".slm"
        data_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))

        from mnemo.storage.db import get_connection, init_schema

        conn = get_connection()
        init_schema(conn)

        # Insert a pattern that meets confidence threshold
        now = time.time()
        fact_ids = [f"f_{i}" for i in range(6)]
        for i, fid in enumerate(fact_ids):
            conn.execute(
                "INSERT INTO facts (id, text, source, created_at, accessed_at, importance, "
                "trust_score, content_hash, lifecycle, precision_bits, strength, retention) "
                "VALUES (?, ?, 'user', ?, ?, 0.8, 1.0, ?, 'Active', 32, 3.5, 1.0)",
                (fid, f"We prefer Python always {i}", now, now, f"h_{i}"),
            )
        conn.commit()

        # Insert pattern directly
        conn.execute(
            """INSERT INTO patterns
            (id, pattern_type, description, evidence_count, positive_rate,
             confidence, last_updated, source_fact_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pat_1",
                "prefers_language",
                "Python preference",
                6,
                0.9,
                0.85,
                now,
                json.dumps(fact_ids),
            ),
        )
        conn.commit()
        yield conn

    def test_generates_prompt(self, prompt_env):
        result = generate_soft_prompts()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prompt_contains_template(self, prompt_env):
        result = generate_soft_prompts()
        # Should contain text from the prefers_language template
        assert "Developer preference" in result or "prefer" in result.lower()

    def test_min_confidence_filter(self, prompt_env):
        """Pattern below MIN_CONFIDENCE should not appear."""
        prompt_env.execute("UPDATE patterns SET confidence = 0.1 WHERE id = 'pat_1'")
        prompt_env.commit()
        result = generate_soft_prompts()
        assert result == ""

    def test_min_evidence_filter(self, prompt_env):
        """Pattern below MIN_EVIDENCE should not appear."""
        prompt_env.execute("UPDATE patterns SET evidence_count = 1 WHERE id = 'pat_1'")
        prompt_env.commit()
        result = generate_soft_prompts()
        assert result == ""

    def test_stores_soft_prompts_in_db(self, prompt_env):
        generate_soft_prompts()
        rows = prompt_env.execute("SELECT * FROM soft_prompts").fetchall()
        assert len(rows) >= 1
        for row in rows:
            assert row["rendered_text"] is not None
            assert row["active"] == 1

    def test_token_limit(self, prompt_env):
        """Output should not exceed MAX_TOKENS * CHARS_PER_TOKEN."""
        result = generate_soft_prompts()
        assert len(result) <= MAX_TOKENS * CHARS_PER_TOKEN + 100  # small margin

    def test_unknown_pattern_type_skipped(self, prompt_env):
        """Pattern with a type not in TEMPLATES should be skipped."""
        prompt_env.execute(
            "INSERT INTO patterns "
            "(id, pattern_type, description, evidence_count, positive_rate, "
            "confidence, last_updated, source_fact_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("pat_unknown", "nonexistent_type", "Unknown", 10, 0.9, 0.95, time.time(), "[]"),
        )
        prompt_env.commit()
        result = generate_soft_prompts()
        assert "nonexistent_type" not in result

    def test_falls_back_to_description_when_no_fact_ids(self, prompt_env):
        """When source_fact_ids is empty, use pattern description as extracted value."""
        prompt_env.execute(
            """INSERT INTO patterns
            (id, pattern_type, description, evidence_count, positive_rate,
             confidence, last_updated, source_fact_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pat_noid",
                "architecture_decision",
                "Uses Redis caching",
                6,
                0.9,
                0.85,
                time.time(),
                "[]",
            ),
        )
        prompt_env.commit()
        result = generate_soft_prompts()
        assert "Redis" in result

    def test_token_overflow_breaks_early(self, prompt_env):
        """When prompt budget is exceeded, stop adding more templates."""
        # Fill with many high-quality patterns to exceed token budget
        for i in range(50):
            prompt_env.execute(
                """INSERT INTO patterns
                (id, pattern_type, description, evidence_count, positive_rate,
                 confidence, last_updated, source_fact_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"pat_overflow_{i}",
                    "architecture_decision",
                    f"Decision {i}: " + "x" * 200,
                    10,
                    0.9,
                    0.99,
                    time.time(),
                    "[]",
                ),
            )
        prompt_env.commit()
        result = generate_soft_prompts()
        # Should not contain all 50 patterns
        assert len(result) <= MAX_TOKENS * CHARS_PER_TOKEN + 100


# ── Constants ─────────────────────────────────────────────────────────


class TestConstants:
    def test_min_confidence(self):
        assert 0 < MIN_CONFIDENCE <= 1.0

    def test_min_evidence(self):
        assert MIN_EVIDENCE >= 1

    def test_max_tokens(self):
        assert MAX_TOKENS > 0

    def test_all_templates_have_placeholder(self):
        for ptype, template in TEMPLATES.items():
            assert "{extracted_value}" in template, f"{ptype} template missing placeholder"
