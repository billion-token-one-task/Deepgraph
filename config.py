"""DeepGraph central configuration."""
import os
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
WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_WORKSPACE_DIR", str(PROJECT_ROOT / "workspace")))
PDF_CACHE_DIR = Path(_env_str("DEEPGRAPH_PDF_CACHE_DIR", str(WORKSPACE_DIR / "pdfs")))

# App
APP_NAME = _env_str("DEEPGRAPH_APP_NAME", "DeepGraph")
APP_SUBTITLE = _env_str("DEEPGRAPH_APP_SUBTITLE", PROFILE_SETTINGS["subtitle"])
ROOT_NODE_ID = _env_str("DEEPGRAPH_ROOT_NODE_ID", PROFILE_SETTINGS["root_node_id"])

# LLM — default: MiniMax only (tabcode/OpenAI-compatible proxies off unless DEEPGRAPH_LLM_USE_TABCODE=1)
LLM_USE_TABCODE = _env_bool("DEEPGRAPH_LLM_USE_TABCODE", False)
LLM_BASE_URL = _env_str("DEEPGRAPH_LLM_BASE_URL", "https://api2.tabcode.cc/openai")
LLM_API_KEY = _env_str(
    "DEEPGRAPH_LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
)
LLM_MODEL = _env_str("DEEPGRAPH_LLM_MODEL", "gpt-5.4")
LLM_MAX_INPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_INPUT_TOKENS", 900_000)
LLM_MAX_OUTPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_OUTPUT_TOKENS", 32_000)

# MiniMax (Chat Completions API)
MINIMAX_API_KEY = _env_str("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = _env_str("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
MINIMAX_MODEL = _env_str("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")
MINIMAX_RPM = _env_int("MINIMAX_RPM", 18)

# arXiv discovery
ARXIV_CATEGORIES = _split_csv(os.getenv("DEEPGRAPH_ARXIV_CATEGORIES")) or PROFILE_SETTINGS["arxiv_categories"]
ARXIV_MAX_RESULTS_PER_QUERY = _env_int("DEEPGRAPH_ARXIV_MAX_RESULTS_PER_QUERY", 100)

# Pipeline
PIPELINE_CONCURRENCY = _env_int("DEEPGRAPH_PIPELINE_CONCURRENCY", 30)
PIPELINE_SLEEP_BETWEEN_PAPERS = _env_int("DEEPGRAPH_PIPELINE_SLEEP_BETWEEN_PAPERS", 1)
BACKFILL_GRAPH_ON_START = _env_bool("DEEPGRAPH_BACKFILL_GRAPH_ON_START", True)
REFRESH_MERGE_CANDIDATES_ON_START = _env_bool("DEEPGRAPH_REFRESH_MERGE_CANDIDATES_ON_START", True)
PAPER_CLUSTER_MIN_PAPERS = _env_int("DEEPGRAPH_PAPER_CLUSTER_MIN_PAPERS", 10)

# Discovery (Tier 1 / Tier 2 insight generation)
DISCOVERY_TIER1_CANDIDATES = _env_int("DEEPGRAPH_TIER1_CANDIDATES", 5)
DISCOVERY_TIER2_PROBLEMS = _env_int("DEEPGRAPH_TIER2_PROBLEMS", 8)
DISCOVERY_TIER2_PAPERS = _env_int("DEEPGRAPH_TIER2_PAPERS", 5)
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

# Web
WEB_HOST = _env_str("DEEPGRAPH_WEB_HOST", "0.0.0.0")
WEB_PORT = _env_int("DEEPGRAPH_WEB_PORT", 8080)
