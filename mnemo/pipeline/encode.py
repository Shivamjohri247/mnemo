"""
Encode pipeline: raw text → dedup check → entity extraction → embedding → store.
"""

import hashlib
import time
import uuid

from mnemo.storage.db import get_connection, transaction
from mnemo.storage.embeddings import get_embedder

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def extract_entities(text: str) -> list[dict]:
    """Returns list of {name, type, canonical_name} dicts."""
    doc = _get_nlp()(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"):
            entities.append(
                {
                    "name": ent.text,
                    "type": ent.label_,
                    "canonical_name": ent.text.lower().strip(),
                }
            )
    return entities


def store_fact(
    text: str,
    source: str = "user",
    project: str | None = None,
    importance: float = 0.5,
    trust_score: float = 1.0,
    valid_from: float | None = None,
    valid_until: float | None = None,
) -> str | None:
    """
    Store a new fact. Returns fact_id, or None if duplicate.
    ADD/UPDATE/SUPERSEDE/NOOP decision based on content_hash.
    """
    conn = get_connection()
    chash = content_hash(text)

    # NOOP check
    existing = conn.execute("SELECT id FROM facts WHERE content_hash = ?", (chash,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE facts SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), existing["id"]),
        )
        conn.commit()
        return None

    fact_id = str(uuid.uuid4())
    now = time.time()

    embedder = get_embedder()
    vector = embedder.embed_text(text)

    with transaction(conn):
        conn.execute(
            """
            INSERT INTO facts (
                id, text, source, project, created_at, accessed_at,
                importance, trust_score, content_hash, lifecycle, precision_bits,
                valid_from, valid_until, strength, retention
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Active', 32, ?, ?, 3.5, 1.0)
            """,
            (
                fact_id,
                text,
                source,
                project,
                now,
                now,
                importance,
                trust_score,
                chash,
                valid_from,
                valid_until,
            ),
        )

        conn.execute(
            "INSERT INTO embeddings (fact_id, vector_f32) VALUES (?, ?)",
            (fact_id, vector.tobytes()),
        )

        entities = extract_entities(text)
        mentioned_ids: list[str] = []
        for ent in entities:
            conn.execute(
                """
                INSERT OR IGNORE INTO entities (id, name, type, canonical_name)
                VALUES (?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), ent["name"], ent["type"], ent["canonical_name"]),
            )
            entity_row = conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ? AND type = ?",
                (ent["canonical_name"], ent["type"]),
            ).fetchone()
            if entity_row:
                conn.execute(
                    "INSERT OR IGNORE INTO entity_mentions (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, entity_row["id"]),
                )
                mentioned_ids.append(entity_row["id"])

        # Populate co-occurrence edges between all entities in this fact
        if len(mentioned_ids) >= 2:
            now_ts = time.time()
            for i, eid_a in enumerate(mentioned_ids):
                for eid_b in mentioned_ids[i + 1 :]:
                    for src, dst in [(eid_a, eid_b), (eid_b, eid_a)]:
                        conn.execute(
                            """
                            INSERT INTO entity_edges (from_entity, to_entity, relation, weight, last_updated)
                            VALUES (?, ?, 'co-occurs', 1.0, ?)
                            ON CONFLICT(from_entity, to_entity) DO UPDATE SET
                                weight = weight + 1.0,
                                last_updated = excluded.last_updated
                            """,
                            (src, dst, now_ts),
                        )

    return fact_id
