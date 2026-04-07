"""Entity-relation-evidence helpers for the scientific graph layer."""
from __future__ import annotations

from difflib import SequenceMatcher
import re
from collections import Counter, defaultdict
from typing import Any

from db import database as db


VALID_ENTITY_TYPES = {
    "concept",
    "method",
    "task",
    "dataset",
    "metric",
    "artifact",
    "material",
    "gene",
    "protein",
    "disease",
    "organism",
    "theory",
}

VALID_RELATIONS = {
    "uses",
    "builds_on",
    "evaluated_on",
    "measured_by",
    "compares_with",
    "applied_to",
    "improves_over",
    "part_of",
    "studies",
    "predicts",
    "treats",
    "interacts_with",
    "derived_from",
    "related_to",
}

METHOD_QUALIFIER_PATTERNS = (
    r"\(ours\)",
    r"\[ours\]",
    r"\(our method\)",
    r"\(proposed\)",
)

TRAILING_MARKERS_PATTERN = re.compile(r"[\*\u2020\u2021]+$")
WHITESPACE_PATTERN = re.compile(r"\s+")
SEPARATOR_PATTERN = re.compile(r"[^a-z0-9]+")
WORD_PATTERN = re.compile(r"[a-z0-9]+")


def canonicalize_entity_name(entity_type: str | None, name: str) -> str:
    """Convert an entity mention into a stable display name."""
    value = (name or "").strip()
    if not value:
        return ""

    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = TRAILING_MARKERS_PATTERN.sub("", value).strip()

    if normalize_entity_type(entity_type) in {"method", "artifact"}:
        for pattern in METHOD_QUALIFIER_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    value = WHITESPACE_PATTERN.sub(" ", value).strip(" -_")
    return value


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for canonical matching."""
    value = WHITESPACE_PATTERN.sub(" ", (name or "").strip().lower())
    value = SEPARATOR_PATTERN.sub("_", value)
    value = value.strip("_")
    return value


def _tokenize_words(name: str) -> list[str]:
    return WORD_PATTERN.findall((name or "").lower())


def _acronym(name: str) -> str:
    words = _tokenize_words(name)
    if len(words) <= 1:
        return ""
    return "".join(word[0] for word in words if word)


def _entity_support_count(entity_id: str) -> int:
    row = db.fetchone(
        """SELECT
             COALESCE((SELECT COUNT(*) FROM paper_entity_mentions WHERE entity_id=?), 0) +
             COALESCE((SELECT COUNT(*) FROM graph_relations WHERE subject_entity_id=? OR object_entity_id=?), 0)
             AS c""",
        (entity_id, entity_id, entity_id),
    )
    return row["c"] if row else 0


def normalize_entity_type(entity_type: str | None) -> str:
    """Normalize entity type to a supported value."""
    value = normalize_entity_name(entity_type or "")
    return value if value in VALID_ENTITY_TYPES else "concept"


def normalize_predicate(predicate: str | None) -> str:
    """Normalize a relation predicate."""
    value = normalize_entity_name(predicate or "")
    return value if value in VALID_RELATIONS else "related_to"


def make_entity_id(entity_type: str, canonical_name: str) -> str:
    """Create a stable entity identifier."""
    norm_type = normalize_entity_type(entity_type)
    display_name = canonicalize_entity_name(norm_type, canonical_name)
    norm_name = normalize_entity_name(display_name)
    return f"{norm_type}:{norm_name}" if norm_name else f"{norm_type}:unknown"


def ensure_entity_resolution(entity_id: str, canonical_entity_id: str | None = None, status: str = "canonical"):
    """Ensure an entity has a resolution row."""
    canonical = canonical_entity_id or entity_id
    db.execute(
        """INSERT INTO entity_resolutions (entity_id, canonical_entity_id, status)
           VALUES (?, ?, ?)
           ON CONFLICT(entity_id) DO UPDATE SET
             canonical_entity_id=excluded.canonical_entity_id,
             status=excluded.status,
             updated_at=CURRENT_TIMESTAMP""",
        (entity_id, canonical, status),
    )
    db.commit()


def backfill_entity_resolutions():
    """Ensure all graph entities have a self-resolution row."""
    rows = db.fetchall(
        """SELECT ge.id
           FROM graph_entities ge
           LEFT JOIN entity_resolutions er ON er.entity_id = ge.id
           WHERE er.entity_id IS NULL"""
    )
    for row in rows:
        db.execute(
            """INSERT INTO entity_resolutions (entity_id, canonical_entity_id, status)
               VALUES (?, ?, 'canonical')""",
            (row["id"], row["id"]),
        )
    if rows:
        db.commit()


def get_canonical_entity_id(entity_id: str) -> str:
    """Resolve an entity ID through the resolution map."""
    row = db.fetchone(
        "SELECT canonical_entity_id FROM entity_resolutions WHERE entity_id=?",
        (entity_id,),
    )
    return row["canonical_entity_id"] if row else entity_id


def _unique_list(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = (value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def upsert_entity(entity: dict) -> str:
    """Insert or update a graph entity and return its ID."""
    raw_name = (entity.get("canonical_name") or entity.get("name") or "").strip()
    entity_type = normalize_entity_type(entity.get("entity_type"))
    canonical_name = canonicalize_entity_name(entity_type, raw_name)
    entity_id = entity.get("id") or make_entity_id(entity_type, canonical_name)
    aliases = _unique_list(([raw_name] if raw_name and raw_name != canonical_name else []) + entity.get("aliases", []))
    normalized_name = normalize_entity_name(canonical_name)

    existing = db.fetchone("SELECT aliases FROM graph_entities WHERE id=?", (entity_id,))
    existing_aliases = db._load_json(existing.get("aliases"), []) if existing else []
    merged_aliases = _unique_list(existing_aliases + aliases)

    db.execute(
        """INSERT INTO graph_entities
           (id, canonical_name, entity_type, normalized_name, description, aliases, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             canonical_name=excluded.canonical_name,
             entity_type=excluded.entity_type,
             normalized_name=excluded.normalized_name,
             description=COALESCE(excluded.description, graph_entities.description),
             aliases=excluded.aliases,
             metadata=excluded.metadata,
             updated_at=CURRENT_TIMESTAMP""",
        (
            entity_id,
            canonical_name or entity_id,
            entity_type,
            normalized_name,
            entity.get("description"),
            db._dump_json(merged_aliases),
            db._dump_json(entity.get("metadata", {})),
        ),
    )
    db.commit()
    ensure_entity_resolution(entity_id)
    return entity_id


def _merge_entity_buckets(items: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], dict] = {}
    for item in items:
        entity_type = normalize_entity_type(item.get("entity_type"))
        canonical_name = canonicalize_entity_name(entity_type, item.get("name") or item.get("canonical_name") or "")
        if not canonical_name:
            continue
        key = (entity_type, canonical_name)
        bucket = buckets.setdefault(key, {
            "name": canonical_name,
            "entity_type": entity_type,
            "description": item.get("description"),
            "aliases": [],
            "mention_role": item.get("mention_role"),
            "confidence": item.get("confidence", 1.0),
            "evidence_location": item.get("evidence_location"),
            "source_text": item.get("source_text"),
            "metadata": item.get("metadata", {}),
        })
        bucket["aliases"] = _unique_list(bucket["aliases"] + item.get("aliases", []) + [item.get("name", "")])
        if not bucket.get("description"):
            bucket["description"] = item.get("description")
        bucket["confidence"] = max(bucket.get("confidence", 0.0), item.get("confidence", 1.0))
        bucket["mention_role"] = bucket.get("mention_role") or item.get("mention_role")
        bucket["evidence_location"] = bucket.get("evidence_location") or item.get("evidence_location")
        bucket["source_text"] = bucket.get("source_text") or item.get("source_text")
    return list(buckets.values())


def _merge_relation_buckets(items: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str, str, str], dict] = {}
    for item in items:
        subject_type = normalize_entity_type(item.get("subject_type"))
        object_type = normalize_entity_type(item.get("object_type"))
        subject = canonicalize_entity_name(subject_type, item.get("subject") or "")
        obj = canonicalize_entity_name(object_type, item.get("object") or "")
        predicate = normalize_predicate(item.get("predicate"))
        if not subject or not obj:
            continue
        key = (subject_type, subject, predicate, object_type, obj)
        bucket = buckets.setdefault(key, {
            "subject": subject,
            "subject_type": subject_type,
            "predicate": predicate,
            "object": obj,
            "object_type": object_type,
            "confidence": item.get("confidence", 1.0),
            "evidence_location": item.get("evidence_location"),
            "source_text": item.get("source_text"),
        })
        bucket["confidence"] = max(bucket.get("confidence", 0.0), item.get("confidence", 1.0))
        bucket["evidence_location"] = bucket.get("evidence_location") or item.get("evidence_location")
        bucket["source_text"] = bucket.get("source_text") or item.get("source_text")
    return list(buckets.values())


def merge_graph_payloads(*payloads: dict | None) -> dict:
    """Merge multiple graph payloads into one deduplicated structure."""
    entity_items: list[dict] = []
    relation_items: list[dict] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        entity_items.extend(payload.get("entities", []))
        relation_items.extend(payload.get("relations", []))
    return {
        "entities": _merge_entity_buckets(entity_items),
        "relations": _merge_relation_buckets(relation_items),
    }


def score_entity_merge_candidate(entity_a: dict, entity_b: dict) -> tuple[float, str] | None:
    """Return a heuristic merge score and rationale for two entities."""
    type_a = normalize_entity_type(entity_a.get("entity_type"))
    type_b = normalize_entity_type(entity_b.get("entity_type"))
    if type_a != type_b:
        return None

    name_a = canonicalize_entity_name(type_a, entity_a.get("canonical_name") or entity_a.get("name") or "")
    name_b = canonicalize_entity_name(type_b, entity_b.get("canonical_name") or entity_b.get("name") or "")
    if not name_a or not name_b or name_a == name_b:
        if name_a and name_a == name_b:
            return 1.0, "Canonical names already match after normalization."
        return None

    norm_a = normalize_entity_name(name_a)
    norm_b = normalize_entity_name(name_b)
    aliases_a = {normalize_entity_name(alias) for alias in db._load_json(entity_a.get("aliases"), []) if alias}
    aliases_b = {normalize_entity_name(alias) for alias in db._load_json(entity_b.get("aliases"), []) if alias}
    aliases_a.add(norm_a)
    aliases_b.add(norm_b)

    if norm_a == norm_b:
        return 0.99, "Normalized names match exactly."
    if norm_a in aliases_b or norm_b in aliases_a:
        return 0.97, "One entity name appears in the other entity's aliases."
    if aliases_a & aliases_b:
        return 0.95, "Entities share normalized aliases."

    tokens_a = set(_tokenize_words(name_a))
    tokens_b = set(_tokenize_words(name_b))
    if not tokens_a or not tokens_b:
        return None

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    acronym_a = _acronym(name_a)
    acronym_b = _acronym(name_b)
    acronym_match = bool(acronym_a and acronym_b and acronym_a == acronym_b)
    substring_match = norm_a in norm_b or norm_b in norm_a

    score = max(jaccard, ratio)
    reasons = []
    if jaccard >= 0.75:
        reasons.append(f"high token overlap ({jaccard:.2f})")
    if ratio >= 0.90:
        reasons.append(f"strong string similarity ({ratio:.2f})")
    if acronym_match:
        score = max(score, 0.88)
        reasons.append(f"matching acronym ({acronym_a})")
    if substring_match and abs(len(norm_a) - len(norm_b)) <= 8:
        score = max(score, 0.86)
        reasons.append("one normalized name contains the other")

    if score < 0.84:
        return None

    if not reasons:
        reasons.append(f"combined similarity {score:.2f}")
    return score, "; ".join(reasons)


def _merge_candidate_order(entity_a: dict, entity_b: dict) -> tuple[dict, dict]:
    """Pick which entity should act as the primary side of a merge candidate."""
    support_a = entity_a.get("support_count", 0)
    support_b = entity_b.get("support_count", 0)
    if support_a != support_b:
        return (entity_a, entity_b) if support_a > support_b else (entity_b, entity_a)

    len_a = len(entity_a.get("canonical_name") or "")
    len_b = len(entity_b.get("canonical_name") or "")
    if len_a != len_b:
        return (entity_a, entity_b) if len_a <= len_b else (entity_b, entity_a)
    return (entity_a, entity_b) if entity_a["id"] <= entity_b["id"] else (entity_b, entity_a)


def _candidate_block_keys(entity: dict) -> set[str]:
    entity_type = normalize_entity_type(entity.get("entity_type"))
    name = canonicalize_entity_name(entity_type, entity.get("canonical_name") or "")
    norm_name = normalize_entity_name(name)
    tokens = _tokenize_words(name)
    keys = {f"name:{norm_name}"}
    if tokens:
        keys.add(f"first:{tokens[0]}")
        keys.add(f"last:{tokens[-1]}")
    acronym = _acronym(name)
    if acronym:
        keys.add(f"acro:{acronym}")
    for alias in db._load_json(entity.get("aliases"), []):
        alias_norm = normalize_entity_name(alias)
        if alias_norm:
            keys.add(f"alias:{alias_norm}")
    return keys


def list_merge_candidates(status: str = "pending", limit: int = 100, entity_type: str | None = None) -> list[dict]:
    """List merge candidates with display metadata."""
    table_exists = db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_merge_candidates'"
    )
    if not table_exists:
        return []

    sql = """
        SELECT emc.*, 
               p.canonical_name AS primary_name, p.entity_type AS primary_type,
               c.canonical_name AS candidate_name, c.entity_type AS candidate_type
        FROM entity_merge_candidates emc
        JOIN graph_entities p ON p.id = emc.primary_entity_id
        JOIN graph_entities c ON c.id = emc.candidate_entity_id
        WHERE emc.status = ?
    """
    params: list[Any] = [status]
    if entity_type:
        sql += " AND emc.entity_type = ?"
        params.append(normalize_entity_type(entity_type))
    sql += " ORDER BY emc.similarity_score DESC, emc.updated_at DESC LIMIT ?"
    params.append(limit)
    return db.fetchall(sql, tuple(params))


def _collect_entity_context(entity_id: str, limit: int = 5) -> dict:
    """Collect node, paper, and evidence context for one entity."""
    canonical_entity_id = get_canonical_entity_id(entity_id)

    node_rows = db.fetchall(
        """SELECT tn.id AS node_id, tn.name AS node_name, COUNT(*) AS mention_count
           FROM paper_entity_mentions pem
           LEFT JOIN taxonomy_nodes tn ON tn.id = pem.node_id
           WHERE pem.entity_id = ?
           GROUP BY tn.id, tn.name
           ORDER BY mention_count DESC, node_name
           LIMIT ?""",
        (entity_id, limit),
    )

    paper_rows = db.fetchall(
        """SELECT p.id AS paper_id, p.title, pem.mention_role, pem.evidence_location, pem.source_text
           FROM paper_entity_mentions pem
           JOIN papers p ON p.id = pem.paper_id
           WHERE pem.entity_id = ?
           ORDER BY p.published_date DESC, p.id DESC
           LIMIT ?""",
        (entity_id, limit),
    )

    relation_rows = db.fetchall(
        """SELECT gr.predicate, raw_s.canonical_name AS raw_subject_name, raw_o.canonical_name AS raw_object_name,
                  s.canonical_name AS subject_name, o.canonical_name AS object_name,
                  gr.evidence_location, gr.source_text
           FROM graph_relations gr
           LEFT JOIN entity_resolutions ers ON ers.entity_id = gr.subject_entity_id
           LEFT JOIN entity_resolutions ero ON ero.entity_id = gr.object_entity_id
           LEFT JOIN graph_entities s ON s.id = ers.canonical_entity_id
           LEFT JOIN graph_entities o ON o.id = ero.canonical_entity_id
           JOIN graph_entities raw_s ON raw_s.id = gr.subject_entity_id
           JOIN graph_entities raw_o ON raw_o.id = gr.object_entity_id
           WHERE gr.subject_entity_id = ? OR gr.object_entity_id = ?
           ORDER BY gr.id DESC
           LIMIT ?""",
        (entity_id, entity_id, limit),
    )

    return {
        "entity_id": entity_id,
        "canonical_entity_id": canonical_entity_id,
        "nodes": node_rows,
        "papers": paper_rows,
        "relations": relation_rows,
        "support_count": _entity_support_count(entity_id),
    }


def get_merge_candidate_context(candidate_id: int) -> dict | None:
    """Return a merge candidate plus supporting node/paper/relation context."""
    table_exists = db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_merge_candidates'"
    )
    if not table_exists:
        return None

    row = db.fetchone(
        """SELECT emc.*, 
                  p.canonical_name AS primary_name, p.entity_type AS primary_type, p.aliases AS primary_aliases,
                  c.canonical_name AS candidate_name, c.entity_type AS candidate_type, c.aliases AS candidate_aliases
           FROM entity_merge_candidates emc
           JOIN graph_entities p ON p.id = emc.primary_entity_id
           JOIN graph_entities c ON c.id = emc.candidate_entity_id
           WHERE emc.id = ?""",
        (candidate_id,),
    )
    if not row:
        return None
    row["primary_aliases"] = db._load_json(row.get("primary_aliases"), [])
    row["candidate_aliases"] = db._load_json(row.get("candidate_aliases"), [])
    row["primary_context"] = _collect_entity_context(row["primary_entity_id"])
    row["candidate_context"] = _collect_entity_context(row["candidate_entity_id"])
    return row


def list_merge_candidates_with_context(status: str = "pending", limit: int = 100, entity_type: str | None = None) -> list[dict]:
    """List merge candidates and attach lightweight review context."""
    rows = list_merge_candidates(status=status, limit=limit, entity_type=entity_type)
    results = []
    for row in rows:
        if row.get("id") is not None:
            ctx = get_merge_candidate_context(row["id"])
            if ctx is not None:
                results.append(ctx)
    return results


def refresh_merge_candidates(entity_type: str | None = None, min_score: float = 0.84, max_entities_per_type: int = 500) -> dict:
    """Generate heuristic merge candidates for unresolved canonical entities."""
    backfill_entity_resolutions()

    sql = """
        SELECT ge.id, ge.canonical_name, ge.entity_type, ge.aliases,
               COALESCE((SELECT COUNT(*) FROM paper_entity_mentions pem WHERE pem.entity_id = ge.id), 0) +
               COALESCE((SELECT COUNT(*) FROM graph_relations gr WHERE gr.subject_entity_id = ge.id OR gr.object_entity_id = ge.id), 0)
               AS support_count
        FROM graph_entities ge
        JOIN entity_resolutions er ON er.entity_id = ge.id
        WHERE er.canonical_entity_id = ge.id
    """
    params: list[Any] = []
    if entity_type:
        sql += " AND ge.entity_type = ?"
        params.append(normalize_entity_type(entity_type))
    sql += " ORDER BY ge.entity_type, support_count DESC, ge.canonical_name LIMIT ?"
    params.append(max_entities_per_type if entity_type else max_entities_per_type * max(len(VALID_ENTITY_TYPES), 1))
    rows = db.fetchall(sql, tuple(params))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["entity_type"]].append(row)

    created = 0
    compared_pairs: set[tuple[str, str]] = set()
    for current_type, entities in by_type.items():
        blocks: dict[str, list[dict]] = defaultdict(list)
        for entity in entities:
            for key in _candidate_block_keys(entity):
                blocks[key].append(entity)

        candidate_pairs: set[tuple[str, str]] = set()
        for block_entities in blocks.values():
            if len(block_entities) < 2:
                continue
            for i, entity_a in enumerate(block_entities):
                for entity_b in block_entities[i + 1:]:
                    pair = tuple(sorted((entity_a["id"], entity_b["id"])))
                    candidate_pairs.add(pair)

        for entity_a_id, entity_b_id in sorted(candidate_pairs):
            if (entity_a_id, entity_b_id) in compared_pairs:
                continue
            compared_pairs.add((entity_a_id, entity_b_id))
            entity_a = next(entity for entity in entities if entity["id"] == entity_a_id)
            entity_b = next(entity for entity in entities if entity["id"] == entity_b_id)
            scored = score_entity_merge_candidate(entity_a, entity_b)
            if not scored:
                continue
            score, rationale = scored
            if score < min_score:
                continue
            primary, candidate = _merge_candidate_order(entity_a, entity_b)
            db.execute(
                """INSERT INTO entity_merge_candidates
                   (entity_type, primary_entity_id, candidate_entity_id, similarity_score, rationale, generated_by)
                   VALUES (?, ?, ?, ?, ?, 'heuristic')
                   ON CONFLICT(primary_entity_id, candidate_entity_id) DO UPDATE SET
                     similarity_score=excluded.similarity_score,
                     rationale=excluded.rationale,
                     updated_at=CURRENT_TIMESTAMP""",
                (current_type, primary["id"], candidate["id"], score, rationale),
            )
            created += 1
    if created:
        db.commit()
    return {"candidates_upserted": created, "types_scanned": len(by_type), "pairs_compared": len(compared_pairs)}


def decide_merge_candidate(candidate_id: int, decision: str, note: str = "") -> dict | None:
    """Accept or reject a merge candidate, keeping an auditable trail."""
    if decision not in {"accepted", "rejected"}:
        raise ValueError("decision must be 'accepted' or 'rejected'")

    row = db.fetchone(
        """SELECT * FROM entity_merge_candidates WHERE id=?""",
        (candidate_id,),
    )
    if not row:
        return None

    db.execute(
        """UPDATE entity_merge_candidates
           SET status=?, decision_note=?, updated_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (decision, note, candidate_id),
    )

    if decision == "accepted":
        primary_id = get_canonical_entity_id(row["primary_entity_id"])
        candidate_entity_id = row["candidate_entity_id"]

        db.execute(
            """UPDATE entity_resolutions
               SET canonical_entity_id=?, status='merged', updated_at=CURRENT_TIMESTAMP
               WHERE entity_id=? OR canonical_entity_id=?""",
            (primary_id, candidate_entity_id, candidate_entity_id),
        )

        primary = db.fetchone("SELECT aliases FROM graph_entities WHERE id=?", (primary_id,))
        candidate = db.fetchone("SELECT canonical_name, aliases FROM graph_entities WHERE id=?", (candidate_entity_id,))
        merged_aliases = _unique_list(
            db._load_json(primary.get("aliases"), []) +
            db._load_json(candidate.get("aliases"), []) +
            [candidate.get("canonical_name", "")]
        )
        db.execute(
            "UPDATE graph_entities SET aliases=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (db._dump_json(merged_aliases), primary_id),
        )

        # Invalidate cached summaries so future reads recompute with the new canonical mapping.
        db.execute("DELETE FROM node_graph_summaries")
        db.execute("DELETE FROM node_summaries")

    db.commit()
    return db.fetchone("SELECT * FROM entity_merge_candidates WHERE id=?", (candidate_id,))


def build_structured_graph_payload_from_records(
    methods: list[dict],
    results: list[dict],
    claims: list[dict],
    insight: dict | None = None,
) -> dict:
    """Build graph entities and relations from already-structured paper records."""
    entities: list[dict] = []
    relations: list[dict] = []

    def add_entity(name: str, entity_type: str, **extra: Any):
        canonical_name = canonicalize_entity_name(entity_type, name)
        if not canonical_name:
            return
        entities.append({
            "name": canonical_name,
            "entity_type": normalize_entity_type(entity_type),
            "description": extra.get("description"),
            "aliases": extra.get("aliases", []),
            "mention_role": extra.get("mention_role"),
            "confidence": extra.get("confidence", 0.92),
            "evidence_location": extra.get("evidence_location"),
            "source_text": extra.get("source_text"),
            "metadata": extra.get("metadata", {}),
        })

    def add_relation(subject: str, subject_type: str, predicate: str, obj: str, object_type: str, **extra: Any):
        if not subject or not obj:
            return
        relations.append({
            "subject": canonicalize_entity_name(subject_type, subject),
            "subject_type": normalize_entity_type(subject_type),
            "predicate": normalize_predicate(predicate),
            "object": canonicalize_entity_name(object_type, obj),
            "object_type": normalize_entity_type(object_type),
            "confidence": extra.get("confidence", 0.9),
            "evidence_location": extra.get("evidence_location"),
            "source_text": extra.get("source_text"),
        })

    for method in methods:
        add_entity(
            method.get("name", ""),
            "method",
            description=method.get("description"),
            aliases=method.get("builds_on", []),
            mention_role="proposed" if method.get("first_paper_id") else "used",
            confidence=0.96,
        )
        for dep in method.get("builds_on", []):
            add_entity(dep, "method", mention_role="referenced", confidence=0.85)
            add_relation(method.get("name", ""), "method", "builds_on", dep, "method", confidence=0.93)

    for result in results:
        method_name = result.get("method_name") or ""
        dataset_name = result.get("dataset_name") or ""
        metric_name = result.get("metric_name") or ""
        if method_name:
            add_entity(method_name, "method", mention_role="evaluated", confidence=0.94, evidence_location=result.get("evidence_location"))
        if dataset_name:
            add_entity(dataset_name, "dataset", mention_role="benchmark", confidence=0.94, evidence_location=result.get("evidence_location"))
        if metric_name:
            add_entity(metric_name, "metric", mention_role="metric", confidence=0.94, evidence_location=result.get("evidence_location"))
        if method_name and dataset_name:
            add_relation(method_name, "method", "evaluated_on", dataset_name, "dataset",
                         confidence=0.95, evidence_location=result.get("evidence_location"))
        if method_name and metric_name:
            add_relation(method_name, "method", "measured_by", metric_name, "metric",
                         confidence=0.93, evidence_location=result.get("evidence_location"))

    for claim in claims:
        method_name = claim.get("method_name") or ""
        dataset_name = claim.get("dataset_name") or ""
        metric_name = claim.get("metric_name") or ""
        location = claim.get("evidence_location")
        if method_name and dataset_name:
            add_relation(method_name, "method", "evaluated_on", dataset_name, "dataset",
                         confidence=0.86, evidence_location=location, source_text=claim.get("claim_text"))
        if method_name and metric_name:
            add_relation(method_name, "method", "measured_by", metric_name, "metric",
                         confidence=0.84, evidence_location=location, source_text=claim.get("claim_text"))
        if claim.get("claim_type") == "method" and method_name:
            add_entity(method_name, "method", mention_role="method_claim", confidence=0.82, source_text=claim.get("claim_text"))

    if insight:
        work_type = (insight.get("work_type") or "").strip()
        if work_type:
            add_entity(work_type.replace("_", " "), "concept", mention_role="work_type", confidence=0.78)
        for text in insight.get("key_findings", []):
            add_entity(text, "concept", mention_role="finding", confidence=0.72)
        for text in insight.get("limitations", []):
            add_entity(text, "concept", mention_role="limitation", confidence=0.68)
        for text in insight.get("open_questions", []):
            add_entity(text, "concept", mention_role="open_question", confidence=0.68)

    return merge_graph_payloads({"entities": entities, "relations": relations})


def clear_paper_graph(paper_id: str):
    """Remove extracted graph evidence for one paper so it can be rebuilt."""
    db.execute("DELETE FROM graph_relations WHERE paper_id=?", (paper_id,))
    db.execute("DELETE FROM paper_entity_mentions WHERE paper_id=?", (paper_id,))
    db.commit()


def insert_entity_mention(mention: dict, commit: bool = True) -> int:
    """Insert one entity mention row."""
    cur = db.execute(
        """INSERT INTO paper_entity_mentions
           (paper_id, node_id, entity_id, mention_text, mention_role, confidence, evidence_location, source_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            mention["paper_id"],
            mention.get("node_id"),
            mention["entity_id"],
            mention.get("mention_text"),
            mention.get("mention_role"),
            mention.get("confidence", 1.0),
            mention.get("evidence_location"),
            mention.get("source_text"),
        ),
    )
    if commit:
        db.commit()
    return cur.lastrowid


def insert_relation(relation: dict, commit: bool = True) -> int:
    """Insert one relation evidence row."""
    cur = db.execute(
        """INSERT INTO graph_relations
           (paper_id, node_id, subject_entity_id, predicate, object_entity_id,
            confidence, evidence_location, source_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            relation["paper_id"],
            relation.get("node_id"),
            relation["subject_entity_id"],
            normalize_predicate(relation.get("predicate")),
            relation["object_entity_id"],
            relation.get("confidence", 1.0),
            relation.get("evidence_location"),
            relation.get("source_text"),
        ),
    )
    if commit:
        db.commit()
    return cur.lastrowid


def store_paper_graph(paper_id: str, node_ids: list[str], graph: dict) -> dict:
    """Store extracted entities and relations for a paper."""
    clear_paper_graph(paper_id)
    assigned_nodes = node_ids or [None]

    entities_by_name: dict[tuple[str, str], str] = {}
    entity_count = 0
    relation_count = 0

    for entity in graph.get("entities", []):
        entity_type = normalize_entity_type(entity.get("entity_type"))
        canonical_name = canonicalize_entity_name(entity_type, entity.get("canonical_name") or entity.get("name") or "")
        if not canonical_name:
            continue
        entity_id = upsert_entity({
            "name": canonical_name,
            "entity_type": entity_type,
            "description": entity.get("description"),
            "aliases": entity.get("aliases", []),
            "metadata": entity.get("metadata", {}),
        })
        entities_by_name[(entity_type, canonical_name)] = entity_id
        entities_by_name[(entity_type, canonical_name.lower())] = entity_id
        entity_count += 1

        for node_id in assigned_nodes:
            insert_entity_mention({
                "paper_id": paper_id,
                "node_id": node_id,
                "entity_id": entity_id,
                "mention_text": entity.get("mention_text") or canonical_name,
                "mention_role": entity.get("mention_role"),
                "confidence": entity.get("confidence", 1.0),
                "evidence_location": entity.get("evidence_location"),
                "source_text": entity.get("source_text"),
            }, commit=False)

    def resolve_entity_id(name: str, entity_type: str | None = None) -> str:
        norm_type = normalize_entity_type(entity_type)
        key = (norm_type, name)
        lowered = (norm_type, name.lower())
        entity_id = entities_by_name.get(key) or entities_by_name.get(lowered)
        if entity_id:
            return entity_id
        placeholder_id = upsert_entity({
            "name": name,
            "entity_type": norm_type,
            "description": None,
            "aliases": [],
            "metadata": {"auto_created": True},
        })
        entities_by_name[key] = placeholder_id
        entities_by_name[lowered] = placeholder_id
        return placeholder_id

    for relation in graph.get("relations", []):
        subject = (relation.get("subject") or "").strip()
        obj = (relation.get("object") or "").strip()
        if not subject or not obj:
            continue
        subject_entity_id = resolve_entity_id(subject, relation.get("subject_type"))
        object_entity_id = resolve_entity_id(obj, relation.get("object_type"))

        for node_id in assigned_nodes:
            insert_relation({
                "paper_id": paper_id,
                "node_id": node_id,
                "subject_entity_id": subject_entity_id,
                "predicate": relation.get("predicate"),
                "object_entity_id": object_entity_id,
                "confidence": relation.get("confidence", 1.0),
                "evidence_location": relation.get("evidence_location"),
                "source_text": relation.get("source_text"),
            }, commit=False)
            relation_count += 1

    db.commit()
    return {"entities": entity_count, "relations": relation_count}


def build_structured_graph_payload_for_paper(paper_id: str) -> dict:
    """Build graph payload from already stored structured paper data."""
    methods = db.fetchall(
        """SELECT name, category, description, key_innovation, first_paper_id, builds_on
           FROM methods WHERE first_paper_id=?""",
        (paper_id,),
    )
    for method in methods:
        method["builds_on"] = db._load_json(method.get("builds_on"), [])

    results = db.fetchall(
        """SELECT method_name, dataset_name, metric_name, evidence_location
           FROM results WHERE paper_id=?""",
        (paper_id,),
    )
    claims = db.fetchall(
        """SELECT claim_text, claim_type, method_name, dataset_name, metric_name, evidence_location
           FROM claims WHERE paper_id=?""",
        (paper_id,),
    )
    insight = db.get_paper_insight(paper_id)
    return build_structured_graph_payload_from_records(methods, results, claims, insight)


def backfill_graph_from_structured_data(limit: int | None = None, overwrite: bool = False) -> dict:
    """Backfill graph evidence for papers that already have structured extraction."""
    from db import opportunity_engine as opp
    from db import taxonomy as tax

    sql = (
        "SELECT p.id FROM papers p "
        "LEFT JOIN paper_entity_mentions pem ON pem.paper_id = p.id "
        "WHERE p.status IN ('extracted', 'reasoned') "
    )
    params: list[Any] = []
    if not overwrite:
        sql += "GROUP BY p.id HAVING COUNT(pem.id) = 0 "
    else:
        sql += "GROUP BY p.id "
    sql += "ORDER BY p.updated_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    processed = 0
    total_entities = 0
    total_relations = 0
    touched_nodes: set[str] = set()

    for row in rows:
        paper_id = row["id"]
        node_rows = db.fetchall(
            "SELECT node_id FROM paper_taxonomy WHERE paper_id=? ORDER BY node_id",
            (paper_id,),
        )
        node_ids = [item["node_id"] for item in node_rows]
        payload = build_structured_graph_payload_for_paper(paper_id)
        if not payload["entities"] and not payload["relations"]:
            continue
        stats = store_paper_graph(paper_id, node_ids, payload)
        processed += 1
        total_entities += stats["entities"]
        total_relations += stats["relations"]
        touched_nodes.update(node_ids)

    for node_id in sorted(touched_nodes):
        ensure_node_graph_summary(node_id, force=True)
        opp.ensure_node_opportunities(node_id, force=True)
        tax.ensure_node_summary(node_id, force=True)

    return {
        "papers_processed": processed,
        "entities_written": total_entities,
        "relations_written": total_relations,
        "nodes_updated": len(touched_nodes),
    }


def get_paper_graph(paper_id: str) -> dict:
    """Return graph evidence for one paper."""
    entities = db.fetchall(
        """SELECT DISTINCT ge.id, ge.canonical_name, ge.entity_type, ge.description, ge.aliases,
                  raw.canonical_name AS raw_name, raw.id AS raw_entity_id,
                  pem.mention_role, pem.confidence, pem.evidence_location, pem.source_text
           FROM paper_entity_mentions pem
           JOIN entity_resolutions er ON er.entity_id = pem.entity_id
           JOIN graph_entities ge ON ge.id = er.canonical_entity_id
           JOIN graph_entities raw ON raw.id = pem.entity_id
           WHERE pem.paper_id=?
           ORDER BY ge.entity_type, ge.canonical_name""",
        (paper_id,),
    )
    for entity in entities:
        entity["aliases"] = db._load_json(entity.get("aliases"), [])

    relations = db.fetchall(
        """SELECT gr.id, gr.predicate, gr.confidence, gr.evidence_location, gr.source_text,
                  s.canonical_name AS subject_name, s.entity_type AS subject_type,
                  o.canonical_name AS object_name, o.entity_type AS object_type,
                  raw_s.canonical_name AS raw_subject_name,
                  raw_o.canonical_name AS raw_object_name
           FROM graph_relations gr
           JOIN entity_resolutions ers ON ers.entity_id = gr.subject_entity_id
           JOIN entity_resolutions ero ON ero.entity_id = gr.object_entity_id
           JOIN graph_entities s ON s.id = ers.canonical_entity_id
           JOIN graph_entities o ON o.id = ero.canonical_entity_id
           JOIN graph_entities raw_s ON raw_s.id = gr.subject_entity_id
           JOIN graph_entities raw_o ON raw_o.id = gr.object_entity_id
           WHERE gr.paper_id=?
           ORDER BY gr.predicate, subject_name, object_name""",
        (paper_id,),
    )
    return {"entities": entities, "relations": relations}


def summarize_graph_rows(entity_rows: list[dict], relation_rows: list[dict]) -> dict:
    """Build a compact graph summary from entity and relation rows."""
    entity_counter: dict[tuple[str, str], dict] = {}
    type_counter = Counter()
    for row in entity_rows:
        key = (row["entity_type"], row["canonical_name"])
        bucket = entity_counter.setdefault(key, {
            "name": row["canonical_name"],
            "entity_type": row["entity_type"],
            "paper_ids": set(),
            "mentions": 0,
        })
        bucket["paper_ids"].add(row["paper_id"])
        bucket["mentions"] += row.get("mention_count", 1)
        type_counter[row["entity_type"]] += row.get("mention_count", 1)

    relation_counter: dict[tuple[str, str, str], dict] = {}
    for row in relation_rows:
        key = (row["subject_name"], row["predicate"], row["object_name"])
        bucket = relation_counter.setdefault(key, {
            "subject": row["subject_name"],
            "predicate": row["predicate"],
            "object": row["object_name"],
            "paper_ids": set(),
            "count": 0,
        })
        bucket["paper_ids"].add(row["paper_id"])
        bucket["count"] += row.get("relation_count", 1)

    top_entities = sorted(
        (
            {
                "name": bucket["name"],
                "entity_type": bucket["entity_type"],
                "paper_count": len(bucket["paper_ids"]),
                "mention_count": bucket["mentions"],
            }
            for bucket in entity_counter.values()
        ),
        key=lambda item: (-item["paper_count"], -item["mention_count"], item["name"]),
    )[:12]

    top_relations = sorted(
        (
            {
                "subject": bucket["subject"],
                "predicate": bucket["predicate"],
                "object": bucket["object"],
                "paper_count": len(bucket["paper_ids"]),
                "relation_count": bucket["count"],
            }
            for bucket in relation_counter.values()
        ),
        key=lambda item: (-item["paper_count"], -item["relation_count"], item["predicate"], item["subject"], item["object"]),
    )[:12]

    top_entity_types = [
        {"entity_type": entity_type, "mention_count": count}
        for entity_type, count in type_counter.most_common(8)
    ]

    return {
        "top_entities": top_entities,
        "top_relations": top_relations,
        "top_entity_types": top_entity_types,
        "entity_count": len(entity_counter),
        "relation_count": len(relation_counter),
    }


def get_node_graph_snapshot(node_id: str) -> dict:
    """Aggregate graph evidence for a taxonomy node."""
    entity_rows = db.fetchall(
        """SELECT ge.canonical_name, ge.entity_type, pem.paper_id, COUNT(*) as mention_count
           FROM paper_entity_mentions pem
           JOIN entity_resolutions er ON er.entity_id = pem.entity_id
           JOIN graph_entities ge ON ge.id = er.canonical_entity_id
           JOIN taxonomy_nodes t ON pem.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY ge.canonical_name, ge.entity_type, pem.paper_id""",
        (node_id, node_id),
    )

    relation_rows = db.fetchall(
        """SELECT s.canonical_name AS subject_name, gr.predicate, o.canonical_name AS object_name,
                  gr.paper_id, COUNT(*) as relation_count
           FROM graph_relations gr
           JOIN entity_resolutions ers ON ers.entity_id = gr.subject_entity_id
           JOIN entity_resolutions ero ON ero.entity_id = gr.object_entity_id
           JOIN graph_entities s ON s.id = ers.canonical_entity_id
           JOIN graph_entities o ON o.id = ero.canonical_entity_id
           JOIN taxonomy_nodes t ON gr.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY subject_name, gr.predicate, object_name, gr.paper_id""",
        (node_id, node_id),
    )

    summary = summarize_graph_rows(entity_rows, relation_rows)
    summary["paper_count"] = db.fetchone(
        """SELECT COUNT(DISTINCT pem.paper_id) as c
           FROM paper_entity_mentions pem
           JOIN taxonomy_nodes t ON pem.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'""",
        (node_id, node_id),
    )["c"]
    summary["generated_from_papers"] = [
        row["paper_id"] for row in db.fetchall(
            """SELECT DISTINCT pem.paper_id
               FROM paper_entity_mentions pem
               JOIN taxonomy_nodes t ON pem.node_id = t.id
               WHERE t.id = ? OR t.id LIKE ? || '.%'
               ORDER BY pem.paper_id DESC
               LIMIT 20""",
            (node_id, node_id),
        )
    ]
    return summary


def upsert_node_graph_summary(node_id: str, summary: dict):
    """Cache a node graph summary."""
    db.execute(
        """INSERT INTO node_graph_summaries
           (node_id, top_entities, top_relations, top_entity_types, generated_from_papers,
            paper_count, entity_count, relation_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(node_id) DO UPDATE SET
             top_entities=excluded.top_entities,
             top_relations=excluded.top_relations,
             top_entity_types=excluded.top_entity_types,
             generated_from_papers=excluded.generated_from_papers,
             paper_count=excluded.paper_count,
             entity_count=excluded.entity_count,
             relation_count=excluded.relation_count,
             generated_at=CURRENT_TIMESTAMP,
             updated_at=CURRENT_TIMESTAMP""",
        (
            node_id,
            db._dump_json(summary.get("top_entities", [])),
            db._dump_json(summary.get("top_relations", [])),
            db._dump_json(summary.get("top_entity_types", [])),
            db._dump_json(summary.get("generated_from_papers", [])),
            summary.get("paper_count", 0),
            summary.get("entity_count", 0),
            summary.get("relation_count", 0),
        ),
    )
    db.commit()


def get_node_graph_summary(node_id: str) -> dict | None:
    """Load a cached node graph summary."""
    row = db.fetchone("SELECT * FROM node_graph_summaries WHERE node_id=?", (node_id,))
    if not row:
        return None
    for key in ("top_entities", "top_relations", "top_entity_types", "generated_from_papers"):
        row[key] = db._load_json(row.get(key), [])
    return row


def ensure_node_graph_summary(node_id: str, force: bool = False) -> dict | None:
    """Generate or refresh a node graph summary."""
    existing = get_node_graph_summary(node_id)
    if existing and not force:
        return existing

    summary = get_node_graph_snapshot(node_id)
    if summary["paper_count"] == 0:
        return existing
    upsert_node_graph_summary(node_id, summary)
    return get_node_graph_summary(node_id)
