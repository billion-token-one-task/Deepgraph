"""Entity-relation-evidence helpers for the scientific graph layer."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from db import database as db


VALID_ENTITY_TYPES = {
    "concept",
    "method",
    "task",
    "dataset",
    "metric",
    "artifact",
    "empirical_result",
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

GENERIC_ENTITY_NAMES = {
    "model",
    "models",
    "method",
    "methods",
    "system",
    "systems",
    "approach",
    "approaches",
    "framework",
    "task",
    "tasks",
    "dataset",
    "datasets",
    "benchmark",
    "benchmarks",
    "metric",
    "metrics",
    "performance",
    "evaluation",
    "results",
    "analysis",
}


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


def _is_sentence_like_entity(name: str) -> bool:
    text = (name or "").strip()
    if not text:
        return True
    words = _tokenize_words(text)
    if len(words) > 10:
        return True
    if text.endswith((".", "?", "!")) and len(words) > 4:
        return True
    return False


def is_high_value_entity(entity_type: str | None, name: str) -> bool:
    """Filter generic words and claim-like sentences out of the reusable graph."""
    entity_type = normalize_entity_type(entity_type)
    canonical = canonicalize_entity_name(entity_type, name)
    if not canonical:
        return False
    norm = normalize_entity_name(canonical)
    if norm in {normalize_entity_name(x) for x in GENERIC_ENTITY_NAMES}:
        return False
    if entity_type == "concept" and _is_sentence_like_entity(canonical):
        return False
    return True


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
        if not canonical_name or not is_high_value_entity(entity_type, canonical_name):
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
        if (
            not subject
            or not obj
            or not is_high_value_entity(subject_type, subject)
            or not is_high_value_entity(object_type, obj)
        ):
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


def list_merge_candidates(status: str = "pending", limit: int = 100, entity_type: str | None = None) -> list[dict]:
    """List merge candidates with display metadata."""
    if not db.table_exists("entity_merge_candidates"):
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
    if not db.table_exists("entity_merge_candidates"):
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
        if not canonical_name or not is_high_value_entity(entity_type, canonical_name):
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
        if (
            not subject
            or not obj
            or not is_high_value_entity(subject_type, subject)
            or not is_high_value_entity(object_type, obj)
        ):
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

    # Paper overview sentences are useful as claims/signals, but they are not
    # reusable graph entities. Avoid turning findings/open questions into nodes.

    return merge_graph_payloads({"entities": entities, "relations": relations})


def clear_paper_graph(paper_id: str):
    """Remove extracted graph evidence for one paper so it can be rebuilt."""
    db.execute("DELETE FROM graph_relations WHERE paper_id=?", (paper_id,))
    db.execute("DELETE FROM paper_entity_mentions WHERE paper_id=?", (paper_id,))
    db.commit()


def insert_entity_mention(mention: dict, commit: bool = True) -> int:
    """Insert one entity mention row."""
    mid = db.insert_returning_id(
        """INSERT INTO paper_entity_mentions
           (paper_id, node_id, entity_id, mention_text, mention_role, confidence, evidence_location, source_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
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
    return mid


def insert_relation(relation: dict, commit: bool = True) -> int:
    """Insert one relation evidence row."""
    rid = db.insert_returning_id(
        """INSERT INTO graph_relations
           (paper_id, node_id, subject_entity_id, predicate, object_entity_id,
            confidence, evidence_location, source_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
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
    return rid


def store_paper_graph(paper_id: str, node_ids: list[str], graph: dict) -> dict:
    """Store extracted entities and relations for a paper."""
    clear_paper_graph(paper_id)
    # Store each paper-level graph fact once. Multi-node taxonomy assignment is
    # already represented in paper_taxonomy; copying every entity/relation to
    # each node inflates node overlap and bridge signals.
    assigned_nodes = [node_ids[0]] if node_ids else [None]

    entities_by_name: dict[tuple[str, str], str] = {}
    entity_count = 0
    relation_count = 0

    for entity in graph.get("entities", []):
        entity_type = normalize_entity_type(entity.get("entity_type"))
        canonical_name = canonicalize_entity_name(entity_type, entity.get("canonical_name") or entity.get("name") or "")
        if not canonical_name or not is_high_value_entity(entity_type, canonical_name):
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
        if not is_high_value_entity(relation.get("subject_type"), subject):
            continue
        if not is_high_value_entity(relation.get("object_type"), obj):
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
