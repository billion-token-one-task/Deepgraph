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
DB_PATH = Path(_env_str("DEEPGRAPH_DB_PATH", str(PROJECT_ROOT / "deepgraph.db")))
WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_WORKSPACE_DIR", str(PROJECT_ROOT / "workspace")))
PDF_CACHE_DIR = Path(_env_str("DEEPGRAPH_PDF_CACHE_DIR", str(WORKSPACE_DIR / "pdfs")))

# App
APP_NAME = _env_str("DEEPGRAPH_APP_NAME", "DeepGraph")
APP_SUBTITLE = _env_str("DEEPGRAPH_APP_SUBTITLE", PROFILE_SETTINGS["subtitle"])
ROOT_NODE_ID = _env_str("DEEPGRAPH_ROOT_NODE_ID", PROFILE_SETTINGS["root_node_id"])

# LLM
LLM_BASE_URL = _env_str("DEEPGRAPH_LLM_BASE_URL", "https://api.tabcode.cc/openai")
LLM_API_KEY = _env_str("DEEPGRAPH_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = _env_str("DEEPGRAPH_LLM_MODEL", "gpt-5.4")
LLM_MAX_INPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_INPUT_TOKENS", 900_000)
LLM_MAX_OUTPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_OUTPUT_TOKENS", 32_000)

# arXiv discovery
ARXIV_CATEGORIES = _split_csv(os.getenv("DEEPGRAPH_ARXIV_CATEGORIES")) or PROFILE_SETTINGS["arxiv_categories"]
ARXIV_MAX_RESULTS_PER_QUERY = _env_int("DEEPGRAPH_ARXIV_MAX_RESULTS_PER_QUERY", 100)

# Pipeline
PIPELINE_CONCURRENCY = _env_int("DEEPGRAPH_PIPELINE_CONCURRENCY", 4)
PIPELINE_SLEEP_BETWEEN_PAPERS = _env_int("DEEPGRAPH_PIPELINE_SLEEP_BETWEEN_PAPERS", 1)

# Web
WEB_HOST = _env_str("DEEPGRAPH_WEB_HOST", "0.0.0.0")
WEB_PORT = _env_int("DEEPGRAPH_WEB_PORT", 8080)
