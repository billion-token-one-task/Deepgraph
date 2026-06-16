"""Agentic codebase scout: LLM-driven GitHub search with verification loops."""
from __future__ import annotations

import base64
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx

from agents.llm_client import call_llm_json
from agents.stage_prompts import prompt_block
from db import database as db

GITHUB_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)
ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
TITLE_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "on", "with", "to", "via", "from", "by",
    "using", "through", "into", "over", "under", "between", "across", "about", "as", "at", "is",
    "are", "be", "based", "large", "language", "model", "models", "llm", "llms",
}

GITHUB_API = "https://api.github.com"
ARXIV_API = "https://export.arxiv.org/api/query"
MAX_SEARCH_QUERIES = 5
MAX_CODE_SEARCHES = 3
MAX_PAPER_GITHUB_QUERIES = 4
MAX_CANDIDATES = 16
MAX_AGENT_ROUNDS = 3
MAX_AUTO_VERIFY_PAPER_CANDIDATES = 5
_GITHUB_API_DISABLED_REASON = ""
COMMON_ENTRYPOINTS = (
    "train.py",
    "main.py",
    "run.py",
    "evaluate.py",
    "eval.py",
    "scripts/train.py",
    "src/train.py",
    "training/train.py",
)

SEARCH_PLAN_SYSTEM = prompt_block("code_scout") + """

You are an active research engineer hunting for the best open-source baseline repository.
Start from paper-linked repositories and baseline-method papers already extracted below.
Only design broader GitHub searches when those paper anchors are missing or clearly unsuitable.

Return JSON:
{
  "search_queries": ["multi-agent llm evaluation harness stars:>100", "repo:langchain-ai/langgraph", ...],
  "code_search_queries": ["gsm8k evaluate.py repo:EleutherAI/lm-evaluation-harness", ...],
  "must_find": ["what capability the repo must support"],
  "avoid": ["toy repos, course homework, unrelated domains"],
  "notes": "brief strategy"
}

Rules:
- Prefer repos explicitly linked from supporting papers or baseline-method papers.
- Prefer queries tied to named baselines, benchmarks, frameworks, or seminal papers in the plan.
- Include at least 3 repository search queries and 1 code search when baselines are technical.
- Use GitHub qualifiers when helpful: stars:>N, language:python, topic:llm, repo:owner/name.
- If the experiment is niche, search for the closest evaluation harness rather than an exact method match."""


DELIBERATE_SYSTEM = prompt_block("code_scout") + """

You reviewed real GitHub candidates gathered from live search (metadata + README + entrypoint hints).
Pick the best execution substrate for the locked experiment contract.

Return JSON:
{
  "action": "pick" | "search_more" | "scratch",
  "codebase": {
    "url": "https://github.com/owner/repo or scratch",
    "name": "short name",
    "reason": "why this repo fits the baselines/datasets",
    "setup_commands": ["pip install ..."],
    "main_train_file": "path/to/train.py",
    "main_eval_command": "python ...",
    "expected_baseline_metric": "optional"
  },
  "alternatives": [{"url": "...", "name": "...", "reason": "..."}],
  "additional_queries": ["only when action=search_more"],
  "confidence": 0.0
}

Rules:
- Prefer repos that already run the named baselines on the named datasets.
- Prefer paper-linked repos and repos from baseline-method papers over generic search hits.
- Only choose scratch when no candidate can plausibly host a real benchmark harness.
- If choosing scratch, set real_benchmark_runner=true in reason text and still provide train.py path train.py.
- main_train_file must match an entrypoint_hint when one exists.
- action=search_more only if the candidate pool is clearly off-target; suggest sharper queries."""


def _github_token() -> str | None:
    for key in ("DEEPGRAPH_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DeepGraph-CodeScout",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(client: httpx.Client, path: str, *, params: dict | None = None) -> dict | list | None:
    global _GITHUB_API_DISABLED_REASON
    if _GITHUB_API_DISABLED_REASON:
        return None
    try:
        response = client.get(f"{GITHUB_API}{path}", headers=_github_headers(), params=params or {})
        if response.status_code == 404:
            return None
        if response.status_code in {403, 429}:
            _GITHUB_API_DISABLED_REASON = f"HTTP {response.status_code}"
            print(
                f"[SCOUT] GitHub API disabled for this process after {_GITHUB_API_DISABLED_REASON}; "
                "falling back to paper-linked repos or scratch.",
                flush=True,
            )
            return None
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"[SCOUT] GitHub API {path} failed: {exc}", flush=True)
        return None


def search_github_repositories(query: str, *, per_page: int = 8) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    with httpx.Client(timeout=20.0) as client:
        payload = _github_get(
            client,
            "/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": per_page},
        )
    items = (payload or {}).get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        full_name = str(item.get("full_name") or "").strip()
        if not full_name or "/" not in full_name:
            continue
        results.append(
            {
                "full_name": full_name,
                "url": str(item.get("html_url") or f"https://github.com/{full_name}"),
                "description": str(item.get("description") or "")[:400],
                "stars": int(item.get("stargazers_count") or 0),
                "updated_at": str(item.get("updated_at") or ""),
                "topics": list(item.get("topics") or [])[:8],
                "source_query": query,
                "source_kind": "repository_search",
            }
        )
    return results


def search_github_code(query: str, *, per_page: int = 6) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    with httpx.Client(timeout=20.0) as client:
        payload = _github_get(
            client,
            "/search/code",
            params={"q": query, "per_page": per_page},
        )
    items = (payload or {}).get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        repo = item.get("repository") if isinstance(item.get("repository"), dict) else {}
        full_name = str(repo.get("full_name") or "").strip()
        if not full_name:
            continue
        path = str(item.get("path") or "")
        results.append(
            {
                "full_name": full_name,
                "url": str(repo.get("html_url") or f"https://github.com/{full_name}"),
                "description": f"code hit: {path}",
                "stars": int(repo.get("stargazers_count") or 0),
                "updated_at": "",
                "topics": [],
                "source_query": query,
                "source_kind": "code_search",
                "matched_path": path,
            }
        )
    return results


def _probe_entrypoints(client: httpx.Client, full_name: str) -> list[str]:
    owner, repo = full_name.split("/", 1)
    found: list[str] = []
    for path in COMMON_ENTRYPOINTS:
        payload = _github_get(client, f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}")
        if isinstance(payload, dict) and payload.get("type") == "file":
            found.append(path)
    return found


def _fetch_readme_excerpt(client: httpx.Client, full_name: str, *, limit: int = 1800) -> str:
    owner, repo = full_name.split("/", 1)
    payload = _github_get(client, f"/repos/{owner}/{repo}/readme")
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if not isinstance(content, str):
        return ""
    try:
        raw = base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return ""
    return raw[:limit]


def enrich_repository(candidate: dict) -> dict:
    full_name = str(candidate.get("full_name") or "").strip()
    if not full_name or "/" not in full_name:
        return dict(candidate)
    owner, repo = full_name.split("/", 1)
    enriched = dict(candidate)
    with httpx.Client(timeout=20.0) as client:
        meta = _github_get(client, f"/repos/{owner}/{repo}")
        if isinstance(meta, dict):
            enriched["description"] = str(meta.get("description") or enriched.get("description") or "")[:500]
            enriched["stars"] = int(meta.get("stargazers_count") or enriched.get("stars") or 0)
            enriched["updated_at"] = str(meta.get("updated_at") or enriched.get("updated_at") or "")
            enriched["topics"] = list(meta.get("topics") or enriched.get("topics") or [])[:10]
            enriched["default_branch"] = str(meta.get("default_branch") or "main")
            enriched["language"] = str(meta.get("language") or "")
        enriched["entrypoint_hints"] = _probe_entrypoints(client, full_name)
        enriched["readme_excerpt"] = _fetch_readme_excerpt(client, full_name)
    return enriched


def _normalize_github_repo_url(url: str) -> str | None:
    match = GITHUB_URL_RE.search(str(url or "").strip())
    if not match:
        return None
    owner = match.group("owner").strip(" .")
    repo = match.group("repo").strip(" .")
    if not owner or not repo:
        return None
    return f"https://github.com/{owner}/{repo}"


def _extract_github_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in GITHUB_URL_RE.finditer(str(text or "")):
        normalized = _normalize_github_repo_url(match.group(0))
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(normalized)
    return urls


def _paper_ids_from_insight(parsed: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for field in ("supporting_papers", "source_paper_ids", "source_papers"):
        values = parsed.get(field)
        if isinstance(values, str):
            try:
                import json

                values = json.loads(values)
            except Exception:
                values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            paper_id = str(value or "").strip()
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            ids.append(paper_id)
    return ids


def _fetch_paper_record(paper_id: str) -> dict | None:
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return None
    row = db.fetchone(
        """
        SELECT id, title, abstract, full_text, appendix_text, pdf_url
        FROM papers
        WHERE id=? OR arxiv_base_id=?
        LIMIT 1
        """,
        (paper_id, paper_id),
    )
    return dict(row) if row else None


def _baseline_names(plan: dict) -> list[str]:
    names: list[str] = []
    for baseline in plan.get("baselines") or []:
        if isinstance(baseline, dict):
            name = str(baseline.get("name") or baseline.get("method") or "").strip()
        else:
            name = str(baseline or "").strip()
        if name:
            names.append(name)
    return names


def _match_baseline_method_papers(baseline_names: list[str]) -> list[dict]:
    if not baseline_names:
        return []
    method_rows = db.fetchall(
        """
        SELECT name, first_paper_id
        FROM methods
        WHERE COALESCE(first_paper_id, '') != ''
        ORDER BY LENGTH(name) DESC
        """
    )
    matches: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for baseline in baseline_names:
        baseline_lower = baseline.lower()
        for row in method_rows:
            method_name = str(row.get("name") or "").strip()
            paper_id = str(row.get("first_paper_id") or "").strip()
            if not method_name or not paper_id:
                continue
            method_lower = method_name.lower()
            if method_lower not in baseline_lower and baseline_lower not in method_lower:
                continue
            key = (baseline, paper_id)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "baseline_name": baseline,
                    "method_name": method_name,
                    "paper_id": paper_id,
                }
            )
    return matches


def _candidate_from_repo_url(
    url: str,
    *,
    source_kind: str,
    source_query: str,
    paper_id: str = "",
    paper_title: str = "",
    baseline_name: str = "",
    method_name: str = "",
) -> dict | None:
    normalized = _normalize_github_repo_url(url)
    if not normalized:
        return None
    full_name = normalized.split("github.com/", 1)[-1].strip("/")
    description_bits = []
    if paper_title:
        description_bits.append(f"paper: {paper_title[:160]}")
    if baseline_name:
        description_bits.append(f"baseline: {baseline_name}")
    if method_name and method_name != baseline_name:
        description_bits.append(f"method: {method_name}")
    return {
        "full_name": full_name,
        "url": normalized,
        "description": " | ".join(description_bits) or source_query,
        "stars": 0,
        "updated_at": "",
        "topics": [],
        "source_query": source_query,
        "source_kind": source_kind,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "baseline_name": baseline_name,
        "method_name": method_name,
    }


def _is_arxiv_id(paper_id: str) -> bool:
    base = str(paper_id or "").strip().split("v")[0]
    return bool(ARXIV_ID_RE.match(base))


def _arxiv_base_id(paper_id: str) -> str:
    return str(paper_id or "").strip().split("v")[0]


def _title_keywords(title: str, *, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+_.-]*", str(title or ""))
    picked: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in TITLE_STOPWORDS or len(word) <= 2:
            continue
        picked.append(word)
        if len(picked) >= max_words:
            break
    return " ".join(picked)


def _fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    base = _arxiv_base_id(arxiv_id)
    if not _is_arxiv_id(base):
        return None
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(ARXIV_API, params={"id_list": base, "max_results": 1})
            response.raise_for_status()
            text = response.text
    except Exception as exc:
        print(f"[SCOUT] arXiv metadata fetch failed for {base}: {exc}", flush=True)
        return None

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    entry = root.find("atom:entry", ARXIV_ATOM_NS)
    if entry is None:
        return None
    title = (entry.findtext("atom:title", default="", namespaces=ARXIV_ATOM_NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ARXIV_ATOM_NS) or "").strip()
    authors = [
        (author.findtext("atom:name", default="", namespaces=ARXIV_ATOM_NS) or "").strip()
        for author in entry.findall("atom:author", ARXIV_ATOM_NS)
    ]
    authors = [name for name in authors if name]
    comment = (entry.findtext("{http://arxiv.org/schemas/atom}comment", default="") or "").strip()
    return {
        "id": base,
        "title": title,
        "abstract": summary,
        "full_text": "\n".join(part for part in (summary, comment) if part),
        "authors": authors,
    }


def _paper_record_blob(paper: dict) -> str:
    author_blob = ""
    authors = paper.get("authors")
    if isinstance(authors, list):
        author_blob = " ".join(str(name) for name in authors if name)
    parts = [
        str(paper.get(field) or "")
        for field in ("title", "abstract", "full_text", "appendix_text", "pdf_url")
    ]
    if author_blob:
        parts.append(author_blob)
    return "\n".join(parts)


def _resolve_paper_record(paper_id: str) -> dict | None:
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return None
    paper = _fetch_paper_record(paper_id)
    if paper:
        return paper
    if _is_arxiv_id(paper_id):
        arxiv_meta = _fetch_arxiv_metadata(paper_id)
        if arxiv_meta:
            return arxiv_meta
    return None


def _github_queries_for_paper(
    paper: dict,
    *,
    baseline_name: str = "",
    method_name: str = "",
) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    paper_id = str(paper.get("id") or "").strip()

    def add(query: str) -> None:
        query = " ".join(str(query or "").split())
        if not query:
            return
        key = query.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(query)

    if _is_arxiv_id(paper_id):
        add(_arxiv_base_id(paper_id))
    title_terms = _title_keywords(str(paper.get("title") or ""))
    if title_terms:
        add(f'"{title_terms}" in:name,description,readme language:python')
        add(f"{title_terms} official implementation stars:>1")
    if baseline_name:
        add(f'"{baseline_name}" language:python stars:>1')
    if method_name and method_name.lower() != baseline_name.lower():
        add(f'"{method_name}" language:python stars:>1')
    authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
    if authors and title_terms:
        surname = str(authors[0]).split()[-1]
        if surname:
            add(f"{surname} {title_terms.split()[0]} github")
    return queries[:MAX_PAPER_GITHUB_QUERIES]


def _search_github_for_paper(
    paper: dict,
    *,
    baseline_name: str = "",
    method_name: str = "",
    source_kind: str,
) -> list[dict]:
    enriched = dict(paper)
    paper_id = str(enriched.get("id") or "").strip()
    if _is_arxiv_id(paper_id):
        arxiv_meta = _fetch_arxiv_metadata(paper_id)
        if arxiv_meta:
            for key, value in arxiv_meta.items():
                enriched.setdefault(key, value)

    found: list[dict] = []
    seen_urls: set[str] = set()

    def add_candidate(row: dict | None) -> None:
        if not row:
            return
        key = str(row.get("url") or "").lower()
        if not key or key in seen_urls:
            return
        seen_urls.add(key)
        found.append(row)

    for url in _extract_github_urls(_paper_record_blob(enriched)):
        add_candidate(
            _candidate_from_repo_url(
                url,
                source_kind=source_kind if source_kind != "baseline_paper" else "baseline_paper",
                source_query=f"paper_text:{paper_id or baseline_name or method_name}",
                paper_id=paper_id,
                paper_title=str(enriched.get("title") or ""),
                baseline_name=baseline_name,
                method_name=method_name,
            )
        )

    text_urls = len(found)
    if text_urls == 0:
        for query in _github_queries_for_paper(
            enriched,
            baseline_name=baseline_name,
            method_name=method_name,
        ):
            for row in search_github_repositories(query, per_page=4):
                add_candidate(
                    {
                        **row,
                        "source_kind": "paper_github_search",
                        "source_query": f"github_search:{query}",
                        "paper_id": paper_id,
                        "paper_title": str(enriched.get("title") or ""),
                        "baseline_name": baseline_name,
                        "method_name": method_name,
                    }
                )
    else:
        print(
            f"[SCOUT] Paper {paper_id or baseline_name}: extracted {text_urls} GitHub URL(s) from text",
            flush=True,
        )
    return found


def search_paper_repositories_complete(parsed: dict, plan: dict) -> list[dict]:
    """Complete paper-first repository search: text links, arXiv metadata, GitHub search."""
    gathered: list[dict] = []
    seen_papers: set[str] = set()

    def ingest_paper(paper_id: str, *, baseline_name: str = "", method_name: str = "") -> None:
        paper_id = str(paper_id or "").strip()
        if not paper_id:
            return
        key = paper_id.lower()
        if key in seen_papers:
            return
        paper = _resolve_paper_record(paper_id)
        if not paper:
            print(f"[SCOUT] Could not resolve paper record for {paper_id}", flush=True)
            return
        seen_papers.add(key)
        source_kind = "baseline_paper" if baseline_name else "supporting_paper"
        rows = _search_github_for_paper(
            paper,
            baseline_name=baseline_name,
            method_name=method_name,
            source_kind=source_kind,
        )
        if rows:
            print(
                f"[SCOUT] Paper {paper.get('id') or paper_id}: found {len(rows)} repo candidate(s)",
                flush=True,
            )
        gathered.extend(rows)

    for paper_id in _paper_ids_from_insight(parsed):
        ingest_paper(paper_id)

    for match in _match_baseline_method_papers(_baseline_names(plan)):
        ingest_paper(
            match["paper_id"],
            baseline_name=match["baseline_name"],
            method_name=match["method_name"],
        )

    return _dedupe_candidates(gathered)


def _try_auto_pick_verified_paper_repo(candidates: list[dict]) -> dict | None:
    for row in candidates[:MAX_AUTO_VERIFY_PAPER_CANDIDATES]:
        url = str(row.get("url") or "").strip()
        if not url or url == "scratch":
            continue
        codebase = {
            "url": url,
            "name": str(row.get("full_name") or url.rsplit("/", 1)[-1]),
            "reason": str(row.get("description") or "paper-linked repository"),
            "main_train_file": "train.py",
            "main_eval_command": "python train.py",
        }
        verify = _verify_codebase_download(codebase)
        if verify.get("ok"):
            picked = verify.get("codebase") or codebase
            print(
                f"[SCOUT] Auto-selected verified paper repo: {picked.get('url')} "
                f"entry={picked.get('main_train_file')}",
                flush=True,
            )
            return picked
        print(f"[SCOUT] Paper repo verify failed for {url}: {verify.get('error')}", flush=True)
    return None


def collect_paper_linked_repositories(parsed: dict, plan: dict) -> list[dict]:
    """Backward-compatible alias for the complete paper-first repository search."""
    return search_paper_repositories_complete(parsed, plan)


def _candidate_sort_key(row: dict) -> tuple:
    kind = str(row.get("source_kind") or "")
    if kind == "baseline_paper":
        priority = 0
    elif kind in {"supporting_paper", "paper_link"}:
        priority = 1
    elif kind == "paper_github_search":
        priority = 2
    else:
        priority = 3
    return (priority, -int(row.get("stars") or 0), str(row.get("full_name") or ""))


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in candidates:
        key = str(row.get("full_name") or row.get("url") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=_candidate_sort_key)
    return out


def _execute_search_plan(plan: dict) -> list[dict]:
    gathered: list[dict] = []
    for query in (plan.get("search_queries") or [])[:MAX_SEARCH_QUERIES]:
        gathered.extend(search_github_repositories(str(query)))
    for query in (plan.get("code_search_queries") or [])[:MAX_CODE_SEARCHES]:
        gathered.extend(search_github_code(str(query)))
    return _dedupe_candidates(gathered)[:MAX_CANDIDATES]


def _format_candidate_dossier(candidates: list[dict]) -> str:
    lines = ["# GitHub Candidate Pool"]
    if not candidates:
        lines.append("(empty — previous searches returned nothing useful)")
        return "\n".join(lines)
    for idx, row in enumerate(candidates, start=1):
        hints = ", ".join(row.get("entrypoint_hints") or []) or "none detected"
        lines.append(f"\n## Candidate {idx}: {row.get('full_name')}")
        lines.append(f"- URL: {row.get('url')}")
        lines.append(f"- Stars: {row.get('stars', 0)} | Language: {row.get('language', '?')}")
        lines.append(f"- Source: {row.get('source_kind')} via `{row.get('source_query')}`")
        if row.get("paper_id"):
            lines.append(f"- Paper: {row.get('paper_id')} — {row.get('paper_title', '')}")
        if row.get("baseline_name"):
            lines.append(f"- Baseline anchor: {row.get('baseline_name')}")
        if row.get("matched_path"):
            lines.append(f"- Code hit: {row.get('matched_path')}")
        lines.append(f"- Description: {row.get('description', '')}")
        if row.get("topics"):
            lines.append(f"- Topics: {', '.join(row.get('topics') or [])}")
        lines.append(f"- Entrypoint hints: {hints}")
        readme = str(row.get("readme_excerpt") or "").strip()
        if readme:
            lines.append(f"- README excerpt:\n{readme[:900]}")
    return "\n".join(lines)


def _format_paper_repo_section(paper_candidates: list[dict]) -> str:
    lines = ["# Paper-Linked Code Repositories (highest priority)"]
    if not paper_candidates:
        lines.append("(none extracted from supporting papers or baseline-method papers yet)")
        return "\n".join(lines)
    for row in paper_candidates:
        bits = [f"- {row.get('url')} [{row.get('source_kind')}]"]
        if row.get("paper_id"):
            bits.append(f"paper={row.get('paper_id')}")
        if row.get("baseline_name"):
            bits.append(f"baseline={row.get('baseline_name')}")
        if row.get("paper_title"):
            bits.append(f"title={str(row.get('paper_title'))[:120]}")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def build_scout_context(parsed: dict, plan: dict, *, paper_candidates: list[dict] | None = None) -> str:
    method = parsed.get("proposed_method", {}) if isinstance(parsed.get("proposed_method"), dict) else {}
    node_ids = parsed.get("source_node_ids", []) if isinstance(parsed.get("source_node_ids"), list) else []

    context_parts = ["# Method to Implement"]
    context_parts.append(f"Name: {method.get('name', 'Unknown')}")
    context_parts.append(f"Type: {method.get('type', 'unknown')}")
    context_parts.append(f"Summary: {method.get('one_line', '')}")
    if method.get("definition"):
        context_parts.append(f"Definition: {str(method['definition'])[:800]}")

    context_parts.append("\n# Experimental Plan")
    if plan.get("baselines"):
        context_parts.append("Baselines:")
        for baseline in plan["baselines"][:8]:
            name = baseline.get("name", baseline) if isinstance(baseline, dict) else str(baseline)
            model = baseline.get("model", "") if isinstance(baseline, dict) else ""
            context_parts.append(f"  - {name} {model}".strip())
    if plan.get("datasets"):
        context_parts.append("Datasets:")
        for dataset in plan["datasets"][:8]:
            name = dataset.get("name", dataset) if isinstance(dataset, dict) else str(dataset)
            context_parts.append(f"  - {name}")
    if plan.get("model_targets"):
        context_parts.append("Model targets:")
        for target in plan["model_targets"][:4]:
            if isinstance(target, dict):
                context_parts.append(f"  - {target.get('hf_model') or target.get('name') or target.get('model')}")
    context_parts.append(f"Resource class: {parsed.get('resource_class', 'cpu')}")
    context_parts.append(f"Generated runner supported: {plan.get('generated_runner_supported')}")

    context_parts.append("\n# Research Area")
    context_parts.append(f"Taxonomy nodes: {', '.join(str(x) for x in node_ids[:6])}")
    if parsed.get("problem_statement"):
        context_parts.append("\n# Problem")
        context_parts.append(str(parsed["problem_statement"])[:500])

    if node_ids:
        graph_methods = db.fetchall(
            """
            SELECT DISTINCT ge.canonical_name, ge.description
            FROM graph_entities ge
            JOIN paper_entity_mentions pem ON pem.entity_id = ge.id
            WHERE ge.entity_type = 'method'
              AND pem.node_id IN ({})
            ORDER BY ge.canonical_name
            LIMIT 15
            """.format(",".join("?" * len(node_ids))),
            tuple(node_ids),
        )
        if graph_methods:
            context_parts.append("\n# Known Methods in This Area (knowledge graph)")
            for row in graph_methods:
                desc = f" — {row['description'][:100]}" if row.get("description") else ""
                context_parts.append(f"  - {row['canonical_name']}{desc}")

    paper_candidates = paper_candidates if paper_candidates is not None else search_paper_repositories_complete(parsed, plan)
    context_parts.append("\n" + _format_paper_repo_section(paper_candidates))
    return "\n".join(context_parts)


def _verify_codebase_download(codebase: dict) -> dict:
    from agents.experiment_forge import (
        _codebase_has_expected_entrypoint,
        _code_dir_has_content,
        _download_repo_archive,
        _candidate_train_entrypoints,
        repair_codebase_entrypoint,
    )

    url = str(codebase.get("url") or "").strip()
    if not url or url == "scratch":
        return {"ok": True, "codebase": dict(codebase), "note": "scratch selected"}

    with tempfile.TemporaryDirectory(prefix="deepgraph-scout-") as tmp:
        code_dir = Path(tmp) / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        git_bin = shutil.which("git")
        clone_ok = False
        if git_bin:
            try:
                import subprocess

                subprocess.run(
                    [git_bin, "clone", "--depth", "1", url, str(code_dir)],
                    timeout=90,
                    capture_output=True,
                    check=True,
                )
                clone_ok = _code_dir_has_content(code_dir)
            except Exception as exc:
                print(f"[SCOUT] Clone verify failed for {url}: {exc}", flush=True)
        if not clone_ok:
            if code_dir.exists():
                shutil.rmtree(code_dir, ignore_errors=True)
                code_dir.mkdir(parents=True, exist_ok=True)
            clone_ok = _download_repo_archive(url, code_dir)
        if not clone_ok or not _code_dir_has_content(code_dir):
            return {"ok": False, "error": f"Could not download repository {url}"}

        repaired = repair_codebase_entrypoint(code_dir, dict(codebase))
        if not _codebase_has_expected_entrypoint(code_dir, repaired):
            hints = [p.relative_to(code_dir).as_posix() for p in _candidate_train_entrypoints(code_dir)]
            if hints:
                repaired["main_train_file"] = hints[0]
                repaired["main_eval_command"] = f"python {hints[0]}"
            else:
                return {
                    "ok": False,
                    "error": f"Repository {url} has no train/eval entrypoint",
                    "entrypoint_hints": [],
                }
        return {"ok": True, "codebase": repaired, "note": "download and entrypoint verified"}


def _normalize_pick(raw: dict) -> dict:
    from agents.experiment_forge import _normalize_codebase_metadata, _scratch_codebase

    codebase = raw.get("codebase") if isinstance(raw.get("codebase"), dict) else {}
    url = str(codebase.get("url") or "").strip()
    if not url:
        return _scratch_codebase(reason="agentic scout returned empty codebase")
    if url == "scratch":
        scratch = _scratch_codebase(reason=str(codebase.get("reason") or "agent selected scratch after search"))
        scratch.update({k: v for k, v in codebase.items() if k not in scratch or v})
        scratch["url"] = "scratch"
        scratch.setdefault("real_benchmark_runner", True)
        return _normalize_codebase_metadata(scratch)
    normalized = _normalize_codebase_metadata(codebase)
    normalized["url"] = url
    return normalized


def scout_codebase_agentic(insight: dict) -> dict:
    """LLM-driven multi-round GitHub search, deliberation, and verification."""
    from agents.discovery_metadata import infer_resource_class
    from agents.experiment_forge import _ensure_real_benchmark_plan, _parse_insight_fields, _scratch_codebase

    parsed = _parse_insight_fields(insight)
    method = parsed.get("proposed_method", {}) if isinstance(parsed.get("proposed_method"), dict) else {}
    plan = _ensure_real_benchmark_plan(
        parsed,
        method,
        parsed.get("experimental_plan", {}),
        parsed.get("resource_class") or infer_resource_class(parsed),
    )
    context = build_scout_context(parsed, plan)
    paper_candidates = search_paper_repositories_complete(parsed, plan)
    if paper_candidates:
        print(f"[SCOUT] Paper-first search found {len(paper_candidates)} repo candidate(s)", flush=True)
    auto_picked = _try_auto_pick_verified_paper_repo(paper_candidates)
    if auto_picked:
        return auto_picked

    candidates: list[dict] = list(paper_candidates)
    last_failure = ""

    for round_idx in range(MAX_AGENT_ROUNDS):
        print(f"[SCOUT] Agent round {round_idx + 1}/{MAX_AGENT_ROUNDS}", flush=True)
        plan_prompt = build_scout_context(parsed, plan, paper_candidates=paper_candidates)
        if candidates:
            plan_prompt += "\n\n# Prior Candidate Pool\n" + _format_candidate_dossier(candidates[:6])
        if last_failure:
            plan_prompt += f"\n\n# Previous Verification Failure\n{last_failure}"

        search_plan, _ = call_llm_json(SEARCH_PLAN_SYSTEM, plan_prompt, temperature=0.2)
        if not isinstance(search_plan, dict):
            search_plan = {}
        if round_idx == 0 or len(candidates) <= len(paper_candidates):
            new_candidates = _execute_search_plan(search_plan)
            candidates = _dedupe_candidates(candidates + new_candidates)[:MAX_CANDIDATES]
            print(f"[SCOUT] Gathered {len(candidates)} GitHub candidates", flush=True)

        enriched = [enrich_repository(row) for row in candidates]
        dossier = build_scout_context(parsed, plan, paper_candidates=paper_candidates) + "\n\n" + _format_candidate_dossier(enriched)
        if last_failure:
            dossier += f"\n\n# Previous Verification Failure\n{last_failure}\nPick another repo or justify scratch."

        decision, _ = call_llm_json(DELIBERATE_SYSTEM, dossier, temperature=0.1)
        if not isinstance(decision, dict):
            decision = {}

        action = str(decision.get("action") or "pick").strip().lower()
        if action == "search_more":
            extra = decision.get("additional_queries") or search_plan.get("search_queries") or []
            extra_plan = {
                "search_queries": list(extra)[:MAX_SEARCH_QUERIES],
                "code_search_queries": list(decision.get("code_search_queries") or [])[:MAX_CODE_SEARCHES],
            }
            candidates = _dedupe_candidates(candidates + _execute_search_plan(extra_plan))[:MAX_CANDIDATES]
            continue

        codebase = _normalize_pick(decision)
        if action == "scratch" or codebase.get("url") == "scratch":
            print("[SCOUT] Agent chose scratch after active search", flush=True)
            return codebase

        verify = _verify_codebase_download(codebase)
        if verify.get("ok"):
            picked = verify.get("codebase") or codebase
            print(f"[SCOUT] Verified repo: {picked.get('url')} entry={picked.get('main_train_file')}", flush=True)
            return picked

        last_failure = str(verify.get("error") or "verification failed")
        rejected_url = codebase.get("url")
        candidates = [row for row in candidates if str(row.get("url") or "").strip() != str(rejected_url or "").strip()]
        alts = decision.get("alternatives") if isinstance(decision.get("alternatives"), list) else []
        for alt in alts:
            if not isinstance(alt, dict):
                continue
            url = str(alt.get("url") or "").strip()
            if url and url != "scratch" and "github.com" in url:
                name = url.rstrip("/").split("github.com/")[-1]
                candidates.append(
                    {
                        "full_name": name,
                        "url": url,
                        "description": str(alt.get("reason") or "LLM alternative"),
                        "stars": 0,
                        "source_kind": "llm_alternative",
                    }
                )
        candidates = _dedupe_candidates(candidates)

    print("[SCOUT] Agentic scout exhausted rounds; falling back to scratch", flush=True)
    return _scratch_codebase(reason=last_failure or "agentic scout could not verify a GitHub repository")
