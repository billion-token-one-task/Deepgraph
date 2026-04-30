"""DeepGraph central configuration."""
import os
import sys
import shutil
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


PROFILE_DEFAULTS = {
    "machine_learning": {
        "root_node_id": "ml",
        "arxiv_categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.MA", "stat.ML"],
        "subtitle": "A plain-language map of what each research area is doing and where the gaps are.",
    },
    "open_science": {
        "root_node_id": "science",
        "arxiv_categories": [
            "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "stat.ML",
            "math.OC", "math.PR", "math.ST",
            "physics.bio-ph", "physics.comp-ph", "physics.data-an", "physics.geo-ph", "physics.med-ph",
            "cond-mat.mtrl-sci", "cond-mat.stat-mech",
            "q-bio.BM", "q-bio.GN", "q-bio.NC", "q-bio.QM",
            "q-fin.ST",
            "eess.AS", "eess.IV", "eess.SP", "eess.SY",
        ],
        "subtitle": "Open scientific opportunity mapping across machine learning, computation, biology, medicine, physics, and more.",
    },
}


PROFILE = _env_str("DEEPGRAPH_PROFILE", "machine_learning")
PROFILE_SETTINGS = PROFILE_DEFAULTS.get(PROFILE, PROFILE_DEFAULTS["machine_learning"])

# Paths
PROJECT_ROOT = Path(__file__).parent


def _load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ (does not override existing exports)."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val


_load_dotenv_file(PROJECT_ROOT / ".env")

DB_PATH = Path(_env_str("DEEPGRAPH_DB_PATH", str(PROJECT_ROOT / "deepgraph.db")))
# PostgreSQL: when DEEPGRAPH_DATABASE_URL is set, the app uses psycopg3 + schema_postgres.sql; otherwise SQLite (DEEPGRAPH_DB_PATH).
DATABASE_URL = _env_str("DEEPGRAPH_DATABASE_URL", "")
WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_WORKSPACE_DIR", str(PROJECT_ROOT / "workspace")))
PDF_CACHE_DIR = Path(_env_str("DEEPGRAPH_PDF_CACHE_DIR", str(WORKSPACE_DIR / "pdfs")))

# App
APP_NAME = _env_str("DEEPGRAPH_APP_NAME", "DeepGraph")
APP_SUBTITLE = _env_str("DEEPGRAPH_APP_SUBTITLE", PROFILE_SETTINGS["subtitle"])
ROOT_NODE_ID = _env_str("DEEPGRAPH_ROOT_NODE_ID", PROFILE_SETTINGS["root_node_id"])

# LLM — default: MiniMax only (OpenAI-compatible proxies off unless DEEPGRAPH_LLM_USE_TABCODE=1)
LLM_USE_TABCODE = _env_bool("DEEPGRAPH_LLM_USE_TABCODE", False)
# OpenAI-compatible gateway protocol: "responses" or "chat_completions"
LLM_PROTOCOL = _env_str("DEEPGRAPH_LLM_PROTOCOL", "responses").strip().lower()
LLM_BASE_URL = _env_str("DEEPGRAPH_LLM_BASE_URL", "https://api2.tabcode.cc/openai")
LLM_API_KEY = _env_str(
    "DEEPGRAPH_LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
)
LLM_MODEL = _env_str("DEEPGRAPH_LLM_MODEL", "gpt-5.4")
LLM_RPM = _env_int("DEEPGRAPH_LLM_RPM", 0)
LLM_SECONDARY_ENABLED = _env_bool("DEEPGRAPH_LLM_SECONDARY_ENABLED", False)
LLM_SECONDARY_PROTOCOL = _env_str("DEEPGRAPH_LLM_SECONDARY_PROTOCOL", LLM_PROTOCOL).strip().lower()
LLM_SECONDARY_BASE_URL = _env_str("DEEPGRAPH_LLM_SECONDARY_BASE_URL", "")
LLM_SECONDARY_API_KEY = _env_str("DEEPGRAPH_LLM_SECONDARY_API_KEY", "")
LLM_SECONDARY_MODEL = _env_str("DEEPGRAPH_LLM_SECONDARY_MODEL", LLM_MODEL)
LLM_SECONDARY_RPM = _env_int("DEEPGRAPH_LLM_SECONDARY_RPM", 0)
LLM_REASONING_EFFORT = _env_str("DEEPGRAPH_LLM_REASONING_EFFORT", "medium")
LLM_CONNECT_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_LLM_CONNECT_TIMEOUT_SECONDS", 30)
LLM_REQUEST_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_LLM_REQUEST_TIMEOUT_SECONDS", 300)
LLM_TRANSIENT_RETRIES = _env_int("DEEPGRAPH_LLM_TRANSIENT_RETRIES", 2)
LLM_TRANSIENT_BACKOFF_SECONDS = _env_int("DEEPGRAPH_LLM_TRANSIENT_BACKOFF_SECONDS", 5)
LLM_TRANSIENT_COOLDOWN_SECONDS = _env_int("DEEPGRAPH_LLM_TRANSIENT_COOLDOWN_SECONDS", 180)
LLM_MAX_INPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_INPUT_TOKENS", 900_000)
LLM_MAX_OUTPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_OUTPUT_TOKENS", 32_000)
# Provenance / feedback loop: bump when changing insight prompts for A/B analysis
PROMPT_VERSION = _env_str("DEEPGRAPH_PROMPT_VERSION", "insight_v1")

# MiniMax (Chat Completions API)
MINIMAX_API_KEY = _env_str("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = _env_str("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
MINIMAX_MODEL = _env_str("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")
MINIMAX_RPM = _env_int("MINIMAX_RPM", 18)

# arXiv discovery
ARXIV_CATEGORIES = _split_csv(os.getenv("DEEPGRAPH_ARXIV_CATEGORIES")) or PROFILE_SETTINGS["arxiv_categories"]
ARXIV_MAX_RESULTS_PER_QUERY = _env_int("DEEPGRAPH_ARXIV_MAX_RESULTS_PER_QUERY", 100)

# PDF text: GROBID (TEI XML) preferred for scientific PDFs; see docker-compose.grobid.yml
GROBID_BASE_URL = _env_str("DEEPGRAPH_GROBID_URL", "http://127.0.0.1:8070")
GROBID_REQUEST_TIMEOUT = _env_int("DEEPGRAPH_GROBID_TIMEOUT", 300)
# auto = GROBID then PyMuPDF fallback; grobid = GROBID only; pymupdf = legacy text layer only
PDF_TEXT_BACKEND = _env_str("DEEPGRAPH_PDF_TEXT_BACKEND", "auto").lower()

# Pipeline
PIPELINE_CONCURRENCY = _env_int("DEEPGRAPH_PIPELINE_CONCURRENCY", 30)
PIPELINE_SLEEP_BETWEEN_PAPERS = _env_int("DEEPGRAPH_PIPELINE_SLEEP_BETWEEN_PAPERS", 1)
PIPELINE_INCREMENTAL_INSIGHT_EVERY = _env_int("DEEPGRAPH_INCREMENTAL_INSIGHT_EVERY", 20)
# Claim/result grounding: drop rows below this score before DB insert (0 = keep all)
GROUNDING_MIN_STORE_SCORE = _env_float("DEEPGRAPH_GROUNDING_MIN_STORE_SCORE", 0.0)
# Contradiction detection: if True, only "verified" performance claims (strict cite-and-verify)
CONTRADICTION_REQUIRES_GROUNDED = _env_bool("DEEPGRAPH_CONTRADICTION_REQUIRES_GROUNDED", False)
BACKFILL_GRAPH_ON_START = _env_bool("DEEPGRAPH_BACKFILL_GRAPH_ON_START", True)
REFRESH_MERGE_CANDIDATES_ON_START = _env_bool("DEEPGRAPH_REFRESH_MERGE_CANDIDATES_ON_START", True)
PAPER_CLUSTER_MIN_PAPERS = _env_int("DEEPGRAPH_PAPER_CLUSTER_MIN_PAPERS", 10)

# Discovery (Tier 1 / Tier 2 insight generation)
DISCOVERY_TIER1_CANDIDATES = _env_int("DEEPGRAPH_TIER1_CANDIDATES", 5)
DISCOVERY_TIER2_PROBLEMS = _env_int("DEEPGRAPH_TIER2_PROBLEMS", 8)
DISCOVERY_TIER2_PAPERS = _env_int("DEEPGRAPH_TIER2_PAPERS", 5)
DISCOVERY_MIN_TIER2_BACKLOG = _env_int("DEEPGRAPH_DISCOVERY_MIN_TIER2_BACKLOG", 3)
DISCOVERY_AUTO_TRIGGER_PAPERS = _env_int("DEEPGRAPH_DISCOVERY_AUTO_TRIGGER", 200)
# Bulk/on-demand full pass: wider SQL signals + more LLM slots
DISCOVERY_BULK_TIER1_CANDIDATES = _env_int("DEEPGRAPH_BULK_TIER1_CANDIDATES", 12)
DISCOVERY_BULK_TIER1_OVERLAPS = _env_int("DEEPGRAPH_BULK_TIER1_OVERLAPS", 80)
DISCOVERY_BULK_TIER1_PATTERNS = _env_int("DEEPGRAPH_BULK_TIER1_PATTERNS", 60)
DISCOVERY_BULK_TIER2_PROBLEMS = _env_int("DEEPGRAPH_BULK_TIER2_PROBLEMS", 15)
DISCOVERY_BULK_TIER2_PLATEAUS = _env_int("DEEPGRAPH_BULK_TIER2_PLATEAUS", 35)
DISCOVERY_BULK_TIER2_LIMIT_NODES = _env_int("DEEPGRAPH_BULK_TIER2_LIMIT_NODES", 30)
EVOSCI_VERIFY_TIMEOUT = _env_int("DEEPGRAPH_EVOSCI_VERIFY_TIMEOUT", 900)

# SciForge Experiment Validation
EXPERIMENT_TIME_BUDGET = _env_int("SCIFORGE_TIME_BUDGET", 300)
EXPERIMENT_MAX_ITERATIONS = _env_int("SCIFORGE_MAX_ITERATIONS", 100)
EXPERIMENT_REPRODUCTION_ITERS = _env_int("SCIFORGE_REPRODUCTION_ITERS", 3)
EXPERIMENT_PROXY_DATA_FRACTION = float(_env_str("SCIFORGE_PROXY_DATA_FRACTION", "0.15"))
EXPERIMENT_PROXY_MAX_EPOCHS = _env_int("SCIFORGE_PROXY_MAX_EPOCHS", 10)
EXPERIMENT_EARLY_STOP_THRESHOLD = float(_env_str("SCIFORGE_EARLY_STOP_THRESHOLD", "0.20"))
EXPERIMENT_REFUTE_MIN_ITERS = _env_int("SCIFORGE_REFUTE_MIN_ITERS", 30)
EXPERIMENT_WORKDIR = Path(_env_str("SCIFORGE_WORKDIR", str(Path.home() / "sciforge_runs")))
IDEA_WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_IDEA_WORKSPACE_DIR", str(Path.home() / "deepgraph_ideas")))
RUNTIME_PYTHON = _env_str("DEEPGRAPH_RUNTIME_PYTHON", sys.executable)
MLFLOW_TRACKING_URI = _env_str("DEEPGRAPH_MLFLOW_TRACKING_URI", "")
CODEX_EXEC_ENABLED = _env_bool("DEEPGRAPH_CODEX_EXEC_ENABLED", True)
CODEX_CLI_PATH = _env_str(
    "DEEPGRAPH_CODEX_CLI_PATH",
    shutil.which("codex") or "",
)
CODEX_MODEL = _env_str("DEEPGRAPH_CODEX_MODEL", "")
CODEX_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_CODEX_TIMEOUT_SECONDS", 900)

# GPU scheduling / experiment lanes
GPU_MODE = _env_str("DEEPGRAPH_GPU_MODE", "single_host")
GPU_WORKER_SLOTS = _env_int("DEEPGRAPH_GPU_WORKER_SLOTS", 4)
GPU_VISIBLE_DEVICES = _split_csv(os.getenv("DEEPGRAPH_GPU_VISIBLE_DEVICES")) or [str(i) for i in range(max(1, GPU_WORKER_SLOTS))]
GPU_DEFAULT_MODEL = _env_str("DEEPGRAPH_GPU_DEFAULT_MODEL", "H20")
GPU_DEFAULT_VRAM_GB = _env_int("DEEPGRAPH_GPU_DEFAULT_VRAM_GB", 96)
GPU_POLL_SECONDS = _env_int("DEEPGRAPH_GPU_POLL_SECONDS", 10)
GPU_JOB_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_GPU_JOB_TIMEOUT_SECONDS", 14400)
GPU_REMOTE_SSH_HOST = _env_str("DEEPGRAPH_GPU_REMOTE_SSH_HOST", "")
GPU_REMOTE_SSH_PORT = _env_int("DEEPGRAPH_GPU_REMOTE_SSH_PORT", 22)
GPU_REMOTE_SSH_USER = _env_str("DEEPGRAPH_GPU_REMOTE_SSH_USER", "")
GPU_REMOTE_SSH_PASSWORD = _env_str("DEEPGRAPH_GPU_REMOTE_SSH_PASSWORD", "")
GPU_REMOTE_BASE_DIR = _env_str("DEEPGRAPH_GPU_REMOTE_BASE_DIR", "/root/deepgraph-remote-worker")
GPU_REMOTE_PYTHON = _env_str("DEEPGRAPH_GPU_REMOTE_PYTHON", "python")
# When using SSH GPU workers, the synced repo often has no preinstalled deps; without this the
# remote process exits at import time and GPUs stay idle (nvidia-smi shows ~0 MiB).
GPU_REMOTE_AUTO_PIP_INSTALL = _env_bool("DEEPGRAPH_GPU_REMOTE_AUTO_PIP_INSTALL", True)
GPU_REMOTE_SETUP_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_GPU_REMOTE_SETUP_TIMEOUT_SECONDS", 3600)

# Mechanism-first discovery
IDEA_EVIDENCE_MIN_NON_NUMERIC = _env_int("DEEPGRAPH_IDEA_EVIDENCE_MIN_NON_NUMERIC", 2)

# Manuscript / submission bundle
MANUSCRIPT_LATEX_TEMPLATE = _env_str("DEEPGRAPH_MANUSCRIPT_LATEX_TEMPLATE", "article")
SUBMISSION_BUNDLE_FORMATS = _split_csv(os.getenv("DEEPGRAPH_SUBMISSION_BUNDLE_FORMATS")) or ["conference", "journal"]
MANUSCRIPT_WORKDIR = Path(_env_str("DEEPGRAPH_MANUSCRIPT_WORKDIR", str(Path.home() / "deepgraph_manuscripts")))
# PaperOrchestra is the only supported manuscript backend.
MANUSCRIPT_BACKEND = _env_str("DEEPGRAPH_MANUSCRIPT_BACKEND", "paper_orchestra").lower()

# PaperOrchestra full §4 pipeline (S2 + parallel plot/lit + AgentReview loop)
SEMANTIC_SCHOLAR_API_KEY = _env_str("DEEPGRAPH_SEMANTIC_SCHOLAR_API_KEY", "")
# Optional: shell command template for PaperBanana-style diagrams; must write output image path.
# Example: python /path/to/PaperBanana/run.py --out {output} --spec '{spec}'
PAPERBANANA_CMD = _env_str("DEEPGRAPH_PAPERBANANA_CMD", "")
PAPERORCHESTRA_REFINEMENT_ITERS = _env_int("DEEPGRAPH_PAPERORCHESTRA_REFINEMENT_ITERS", 4)
PGVECTOR_EMBEDDING_DIM = _env_int("DEEPGRAPH_PGVECTOR_EMBEDDING_DIM", 1536)
PIPELINE_EVENT_POLL_SECONDS = _env_int("DEEPGRAPH_PIPELINE_EVENT_POLL_SECONDS", 5)

# Auto Research orchestration
AUTO_RESEARCH_ENABLED = _env_bool("DEEPGRAPH_AUTO_RESEARCH_ENABLED", True)
AUTO_RESEARCH_INTERVAL_SECONDS = _env_int("DEEPGRAPH_AUTO_RESEARCH_INTERVAL_SECONDS", 300)
AUTO_RESEARCH_MAX_ACTIVE = _env_int("DEEPGRAPH_AUTO_RESEARCH_MAX_ACTIVE", 1)

# Web
WEB_HOST = _env_str("DEEPGRAPH_WEB_HOST", "0.0.0.0")
WEB_PORT = _env_int("DEEPGRAPH_WEB_PORT", 8080)
