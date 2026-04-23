"""
4-Channel Retrieval with Reciprocal Rank Fusion.

Channels:
  1. Semantic  — sqlite-vec KNN cosine similarity (weight 1.2)
  2. BM25      — SQLite FTS5 keyword matching (weight 1.0)
  3. Temporal  — Recency-weighted timestamp scoring (weight 1.0)
  4. Entity    — Knowledge graph traversal (weight 1.0)

Fusion: Weighted RRF with k=15.
"""

import logging
import time

from mnemo.storage.db import get_connection
from mnemo.storage.embeddings import embed_text

logger = logging.getLogger(__name__)

RRF_K = 15

CHANNEL_WEIGHTS = {
    "semantic": 1.2,
    "bm25": 1.0,
    "temporal": 1.0,
    "entity": 1.0,
}

TOP_K_PER_CHANNEL = 20
FINAL_TOP_K = 10


def semantic_channel(query: str, project: str | None, k: int) -> list[str]:
    """sqlite-vec KNN over float32 embeddings."""
    conn = get_connection()
    vec = embed_text(query)

    filter_clause = "AND f.project = ?" if project else ""
    params: list = [vec.tobytes(), k]
    if project:
        params.insert(1, project)

    rows = conn.execute(
        f"""
        SELECT f.id
        FROM embeddings e
        JOIN facts f ON e.fact_id = f.id
        WHERE f.lifecycle != 'Forgotten'
        {filter_clause}
        ORDER BY vec_distance_cosine(e.vector_f32, ?) ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [r["id"] for r in rows]


def bm25_channel(query: str, project: str | None, k: int) -> list[str]:
    """SQLite FTS5 BM25 full-text search."""
    conn = get_connection()
    # Wrap each word in double quotes to escape FTS5 operators (NOT, OR, AND)
    # while allowing FTS5 implicit AND matching between words.
    safe_query = " ".join(f'"{word.replace(chr(34), chr(34) + chr(34))}"' for word in query.split())

    filter_clause = "AND f.project = ?" if project else ""
    params: list = [safe_query, k]
    if project:
        params.insert(1, project)

    try:
        rows = conn.execute(
            f"""
            SELECT f.id
            FROM facts_fts fts
            JOIN facts f ON fts.id = f.id
            WHERE facts_fts MATCH ?
            AND f.lifecycle != 'Forgotten'
            {filter_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [r["id"] for r in rows]
    except Exception:
        logger.debug("BM25 search failed for query: %s", query, exc_info=True)
        return []


def temporal_channel(query: str, project: str | None, k: int) -> list[str]:
    """Recency-weighted scoring: score = 1 / (1 + hours_since_access)."""
    conn = get_connection()
    now = time.time()

    filter_clause = "AND project = ?" if project else ""
    params: list = [now, k]
    if project:
        params.insert(0, project)

    rows = conn.execute(
        f"""
        SELECT id,
               1.0 / (1.0 + (? - accessed_at) / 3600.0) AS recency_score
        FROM facts
        WHERE lifecycle != 'Forgotten'
        {filter_clause}
        ORDER BY recency_score DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [r["id"] for r in rows]


_nlp = None


def _get_nlp():
    """Lazy-load and cache spacy NLP model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.debug("spacy model en_core_web_sm not available")
            return None
    return _nlp


def entity_channel(query: str, project: str | None, k: int) -> list[str]:
    """Knowledge graph traversal: extract entities from query → direct + 1-hop."""
    conn = get_connection()

    nlp = _get_nlp()
    if nlp is None:
        return []

    try:
        doc = nlp(query)
        query_entities = [ent.text.lower().strip() for ent in doc.ents]
    except Exception:
        logger.debug("Entity extraction failed for query: %s", query, exc_info=True)
        return []

    if not query_entities:
        return []

    placeholders = ",".join("?" * len(query_entities))
    project_filter = "AND f.project = ?" if project else ""
    params: list = query_entities + ([project] if project else []) + [k]

    direct = conn.execute(
        f"""
        SELECT DISTINCT em.fact_id
        FROM entity_mentions em
        JOIN entities e ON em.entity_id = e.id
        JOIN facts f ON em.fact_id = f.id
        WHERE e.canonical_name IN ({placeholders})
        AND f.lifecycle != 'Forgotten'
        {project_filter}
        LIMIT ?
        """,
        params,
    ).fetchall()
    direct_ids = [r["fact_id"] for r in direct]

    if direct_ids:
        entity_placeholders = ",".join("?" * len(direct_ids))
        entity_ids = conn.execute(
            f"""
            SELECT DISTINCT em.entity_id
            FROM entity_mentions em
            WHERE em.fact_id IN ({entity_placeholders})
            """,
            direct_ids,
        ).fetchall()

        hop_entity_ids = [r["entity_id"] for r in entity_ids]
        if hop_entity_ids:
            edge_placeholders = ",".join("?" * len(hop_entity_ids))
            related = conn.execute(
                f"""
                SELECT to_entity FROM entity_edges
                WHERE from_entity IN ({edge_placeholders})
                ORDER BY weight DESC LIMIT 20
                """,
                hop_entity_ids,
            ).fetchall()

            all_entity_ids = hop_entity_ids + [r["to_entity"] for r in related]
            all_placeholders = ",".join("?" * len(all_entity_ids))
            hop_project_filter = "AND f.project = ?" if project else ""
            hop_params: list = all_entity_ids + ([project] if project else []) + [k]
            hop_facts = conn.execute(
                f"""
                SELECT DISTINCT em.fact_id
                FROM entity_mentions em
                JOIN facts f ON em.fact_id = f.id
                WHERE em.entity_id IN ({all_placeholders})
                AND f.lifecycle != 'Forgotten'
                {hop_project_filter}
                LIMIT ?
                """,
                hop_params,
            ).fetchall()
            return direct_ids + [r["fact_id"] for r in hop_facts if r["fact_id"] not in direct_ids]

    return direct_ids


def rrf_score(rank: int, weight: float, k: int = RRF_K) -> float:
    """Paper equation (11): score(m) = Σ w_c · 1/(k + rank_c(m))."""
    return weight * (1.0 / (k + rank))


def fuse(channel_results: dict[str, list[str]]) -> list[str]:
    """Weighted Reciprocal Rank Fusion across all channels."""
    scores: dict[str, float] = {}
    for channel, fact_ids in channel_results.items():
        w = CHANNEL_WEIGHTS.get(channel, 1.0)
        for rank, fid in enumerate(fact_ids):
            scores[fid] = scores.get(fid, 0.0) + rrf_score(rank, w)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def recall(
    query: str,
    project: str | None = None,
    top_k: int = FINAL_TOP_K,
    channels: list[str] | None = None,
) -> list[dict]:
    """Main retrieval: run channels, fuse with RRF, return top-k facts."""
    active_channels = channels or list(CHANNEL_WEIGHTS.keys())
    k = TOP_K_PER_CHANNEL

    channel_results: dict[str, list[str]] = {}
    if "semantic" in active_channels:
        channel_results["semantic"] = semantic_channel(query, project, k)
    if "bm25" in active_channels:
        channel_results["bm25"] = bm25_channel(query, project, k)
    if "temporal" in active_channels:
        channel_results["temporal"] = temporal_channel(query, project, k)
    if "entity" in active_channels:
        channel_results["entity"] = entity_channel(query, project, k)

    fused_ids = fuse(channel_results)[:top_k]

    conn = get_connection()
    if not fused_ids:
        return []

    placeholders = ",".join("?" * len(fused_ids))
    rows = conn.execute(f"SELECT * FROM facts WHERE id IN ({placeholders})", fused_ids).fetchall()

    id_to_row = {r["id"]: dict(r) for r in rows}
    results = [id_to_row[fid] for fid in fused_ids if fid in id_to_row]

    now = time.time()
    with conn:
        for fid in fused_ids[:3]:
            conn.execute(
                """
                UPDATE facts
                SET accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (now, fid),
            )

    return results
