"""DeepGraph central configuration."""
import os
import sys
import shutil
import tomllib
from pathlib import Path

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


def _load_toml_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


_load_dotenv_file(PROJECT_ROOT / ".env")
CONFIG_TOML_PATH = Path(os.getenv("DEEPGRAPH_CONFIG_TOML", str(PROJECT_ROOT / "deepgraph.toml")))
CONFIG_TOML = _load_toml_file(CONFIG_TOML_PATH)


def _toml_get(path: str | None, default):
    if not path:
        return default
    cur = CONFIG_TOML
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _expand_config_refs(value: str) -> str:
    return (
        value.replace("{project_root}", str(PROJECT_ROOT))
        .replace("{repo_root}", str(PROJECT_ROOT))
        .replace("{home}", str(Path.home()))
    )


def _env_str(name: str, default: str, toml_path: str | None = None) -> str:
    default_value = _toml_get(toml_path, default)
    if isinstance(default_value, str) and not default_value.strip() and str(default).strip():
        default_value = default
    value = os.getenv(name)
    selected = value.strip() if value and value.strip() else str(default_value)
    return _expand_config_refs(selected)


def _env_int(name: str, default: int, toml_path: str | None = None) -> int:
    default_value = _toml_get(toml_path, default)
    value = os.getenv(name)
    if not value:
        try:
            return int(default_value)
        except (TypeError, ValueError):
            return default
    try:
        return int(value)
    except ValueError:
        try:
            return int(default_value)
        except (TypeError, ValueError):
            return default


def _env_bool(name: str, default: bool, toml_path: str | None = None) -> bool:
    default_value = _toml_get(toml_path, default)
    value = os.getenv(name)
    if value is None:
        if isinstance(default_value, bool):
            return default_value
        return str(default_value).strip().lower() in {"1", "true", "yes", "on"}
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, toml_path: str | None = None) -> float:
    default_value = _toml_get(toml_path, default)
    value = os.getenv(name)
    if not value:
        try:
            return float(default_value)
        except (TypeError, ValueError):
            return default
    try:
        return float(value.strip())
    except ValueError:
        try:
            return float(default_value)
        except (TypeError, ValueError):
            return default


def _split_csv(value: str | list | tuple | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
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


PROFILE = _env_str("DEEPGRAPH_PROFILE", "machine_learning", "profile.active")
PROFILE_SETTINGS = PROFILE_DEFAULTS.get(PROFILE, PROFILE_DEFAULTS["machine_learning"])

# Paths
DB_PATH = Path(_env_str("DEEPGRAPH_DB_PATH", str(PROJECT_ROOT / "deepgraph.db"), "paths.db_path")).expanduser()
# PostgreSQL: when DEEPGRAPH_DATABASE_URL is set, the app uses psycopg3 + schema_postgres.sql; otherwise SQLite (DEEPGRAPH_DB_PATH).
DATABASE_URL = _env_str("DEEPGRAPH_DATABASE_URL", "", "database.url")
WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_WORKSPACE_DIR", str(PROJECT_ROOT / "workspace"), "paths.workspace_dir")).expanduser()
PDF_CACHE_DIR = Path(_env_str("DEEPGRAPH_PDF_CACHE_DIR", str(WORKSPACE_DIR / "pdfs"), "paths.pdf_cache_dir")).expanduser()

# App
APP_NAME = _env_str("DEEPGRAPH_APP_NAME", "DeepGraph", "app.name")
APP_SUBTITLE = _env_str("DEEPGRAPH_APP_SUBTITLE", PROFILE_SETTINGS["subtitle"], "app.subtitle")
ROOT_NODE_ID = _env_str("DEEPGRAPH_ROOT_NODE_ID", PROFILE_SETTINGS["root_node_id"], "app.root_node_id")

# LLM — default: MiniMax only (OpenAI-compatible proxies off unless DEEPGRAPH_LLM_USE_TABCODE=1)
LLM_USE_TABCODE = _env_bool("DEEPGRAPH_LLM_USE_TABCODE", False, "llm.use_tabcode")
# OpenAI-compatible gateway protocol: "responses" or "chat_completions"
LLM_PROTOCOL = _env_str("DEEPGRAPH_LLM_PROTOCOL", "responses", "llm.protocol").strip().lower()
LLM_BASE_URL = _env_str("DEEPGRAPH_LLM_BASE_URL", "https://api2.tabcode.cc/openai", "llm.base_url")
LLM_API_KEY = _env_str(
    "DEEPGRAPH_LLM_API_KEY",
    os.getenv("OPENAI_API_KEY", ""),
    "llm.api_key",
)
LLM_MODEL = _env_str("DEEPGRAPH_LLM_MODEL", "gpt-5.4", "llm.model")
LLM_RPM = _env_int("DEEPGRAPH_LLM_RPM", 0, "llm.rpm")
LLM_SECONDARY_ENABLED = _env_bool("DEEPGRAPH_LLM_SECONDARY_ENABLED", False, "llm.secondary.enabled")
LLM_SECONDARY_PROTOCOL = _env_str("DEEPGRAPH_LLM_SECONDARY_PROTOCOL", LLM_PROTOCOL, "llm.secondary.protocol").strip().lower()
LLM_SECONDARY_BASE_URL = _env_str("DEEPGRAPH_LLM_SECONDARY_BASE_URL", "", "llm.secondary.base_url")
LLM_SECONDARY_API_KEY = _env_str("DEEPGRAPH_LLM_SECONDARY_API_KEY", "", "llm.secondary.api_key")
LLM_SECONDARY_MODEL = _env_str("DEEPGRAPH_LLM_SECONDARY_MODEL", LLM_MODEL, "llm.secondary.model")
LLM_SECONDARY_RPM = _env_int("DEEPGRAPH_LLM_SECONDARY_RPM", 0, "llm.secondary.rpm")
# JSON list of additional OpenAI-compatible routes. Each item may include:
# name, base_url, api_key, model, protocol, rpm, enabled, stream_chat_completions.
LLM_EXTRA_PROVIDERS_JSON = _env_str("DEEPGRAPH_LLM_EXTRA_PROVIDERS_JSON", "", "llm.extra_providers_json")
LLM_REASONING_EFFORT = _env_str("DEEPGRAPH_LLM_REASONING_EFFORT", "medium", "llm.reasoning_effort")
LLM_CONNECT_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_LLM_CONNECT_TIMEOUT_SECONDS", 30, "llm.connect_timeout_seconds")
LLM_REQUEST_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_LLM_REQUEST_TIMEOUT_SECONDS", 300, "llm.request_timeout_seconds")
LLM_TRANSIENT_RETRIES = _env_int("DEEPGRAPH_LLM_TRANSIENT_RETRIES", 2, "llm.transient_retries")
LLM_TRANSIENT_BACKOFF_SECONDS = _env_int("DEEPGRAPH_LLM_TRANSIENT_BACKOFF_SECONDS", 5, "llm.transient_backoff_seconds")
LLM_TRANSIENT_COOLDOWN_SECONDS = _env_int("DEEPGRAPH_LLM_TRANSIENT_COOLDOWN_SECONDS", 180, "llm.transient_cooldown_seconds")
LLM_MAX_INPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_INPUT_TOKENS", 900_000, "llm.max_input_tokens")
LLM_MAX_OUTPUT_TOKENS = _env_int("DEEPGRAPH_LLM_MAX_OUTPUT_TOKENS", 32_000, "llm.max_output_tokens")
# Provenance / feedback loop: bump when changing insight prompts for A/B analysis
PROMPT_VERSION = _env_str("DEEPGRAPH_PROMPT_VERSION", "insight_v1", "prompts.version")

# MiniMax (Chat Completions API)
MINIMAX_API_KEY = _env_str("MINIMAX_API_KEY", "", "minimax.api_key")
MINIMAX_BASE_URL = _env_str("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1", "minimax.base_url")
MINIMAX_MODEL = _env_str("MINIMAX_MODEL", "MiniMax-M2.7-highspeed", "minimax.model")
MINIMAX_RPM = _env_int("MINIMAX_RPM", 18, "minimax.rpm")

# arXiv discovery
ARXIV_CATEGORIES = _split_csv(os.getenv("DEEPGRAPH_ARXIV_CATEGORIES") or _toml_get("arxiv.categories", None)) or PROFILE_SETTINGS["arxiv_categories"]
ARXIV_MAX_RESULTS_PER_QUERY = _env_int("DEEPGRAPH_ARXIV_MAX_RESULTS_PER_QUERY", 100, "arxiv.max_results_per_query")

# PDF text: GROBID (TEI XML) preferred for scientific PDFs; see docker-compose.grobid.yml
GROBID_BASE_URL = _env_str("DEEPGRAPH_GROBID_URL", "http://127.0.0.1:8070", "grobid.base_url")
GROBID_REQUEST_TIMEOUT = _env_int("DEEPGRAPH_GROBID_TIMEOUT", 300, "grobid.request_timeout_seconds")
# auto = GROBID then PyMuPDF fallback; grobid = GROBID only; pymupdf = legacy text layer only
PDF_TEXT_BACKEND = _env_str("DEEPGRAPH_PDF_TEXT_BACKEND", "auto", "pdf.text_backend").lower()

# Pipeline
PIPELINE_CONCURRENCY = _env_int("DEEPGRAPH_PIPELINE_CONCURRENCY", 30, "pipeline.concurrency")
PIPELINE_SLEEP_BETWEEN_PAPERS = _env_int("DEEPGRAPH_PIPELINE_SLEEP_BETWEEN_PAPERS", 1, "pipeline.sleep_between_papers")
PIPELINE_INCREMENTAL_INSIGHT_EVERY = _env_int("DEEPGRAPH_INCREMENTAL_INSIGHT_EVERY", 20, "pipeline.incremental_insight_every")
PIPELINE_MAX_RETRYABLE_FAILURES = _env_int("DEEPGRAPH_PIPELINE_MAX_RETRYABLE_FAILURES", 12, "pipeline.max_retryable_failures")
# Claim/result grounding: drop rows below this score before DB insert (0 = keep all)
GROUNDING_MIN_STORE_SCORE = _env_float("DEEPGRAPH_GROUNDING_MIN_STORE_SCORE", 0.0, "graph.grounding_min_store_score")
# Contradiction detection: if True, only "verified" performance claims (strict cite-and-verify)
CONTRADICTION_REQUIRES_GROUNDED = _env_bool("DEEPGRAPH_CONTRADICTION_REQUIRES_GROUNDED", False, "graph.contradiction_requires_grounded")
BACKFILL_GRAPH_ON_START = _env_bool("DEEPGRAPH_BACKFILL_GRAPH_ON_START", True, "graph.backfill_on_start")
REFRESH_MERGE_CANDIDATES_ON_START = _env_bool("DEEPGRAPH_REFRESH_MERGE_CANDIDATES_ON_START", True, "graph.refresh_merge_candidates_on_start")
PAPER_CLUSTER_MIN_PAPERS = _env_int("DEEPGRAPH_PAPER_CLUSTER_MIN_PAPERS", 10, "graph.paper_cluster_min_papers")

# Discovery (Tier 1 / Tier 2 insight generation)
DISCOVERY_TIER1_CANDIDATES = _env_int("DEEPGRAPH_TIER1_CANDIDATES", 5, "discovery.tier1_candidates")
DISCOVERY_TIER2_PROBLEMS = _env_int("DEEPGRAPH_TIER2_PROBLEMS", 8, "discovery.tier2_problems")
DISCOVERY_TIER2_PAPERS = _env_int("DEEPGRAPH_TIER2_PAPERS", 5, "discovery.tier2_papers")
DISCOVERY_MIN_TIER2_BACKLOG = _env_int("DEEPGRAPH_DISCOVERY_MIN_TIER2_BACKLOG", 3, "discovery.min_tier2_backlog")
DISCOVERY_AUTO_TRIGGER_PAPERS = _env_int("DEEPGRAPH_DISCOVERY_AUTO_TRIGGER", 200, "discovery.auto_trigger_papers")
# Bulk/on-demand full pass: wider SQL signals + more LLM slots
DISCOVERY_BULK_TIER1_CANDIDATES = _env_int("DEEPGRAPH_BULK_TIER1_CANDIDATES", 12, "discovery.bulk_tier1_candidates")
DISCOVERY_BULK_TIER1_OVERLAPS = _env_int("DEEPGRAPH_BULK_TIER1_OVERLAPS", 80, "discovery.bulk_tier1_overlaps")
DISCOVERY_BULK_TIER1_PATTERNS = _env_int("DEEPGRAPH_BULK_TIER1_PATTERNS", 60, "discovery.bulk_tier1_patterns")
DISCOVERY_BULK_TIER2_PROBLEMS = _env_int("DEEPGRAPH_BULK_TIER2_PROBLEMS", 15, "discovery.bulk_tier2_problems")
DISCOVERY_BULK_TIER2_PLATEAUS = _env_int("DEEPGRAPH_BULK_TIER2_PLATEAUS", 35, "discovery.bulk_tier2_plateaus")
DISCOVERY_BULK_TIER2_LIMIT_NODES = _env_int("DEEPGRAPH_BULK_TIER2_LIMIT_NODES", 30, "discovery.bulk_tier2_limit_nodes")
EVOSCI_VERIFY_TIMEOUT = _env_int("DEEPGRAPH_EVOSCI_VERIFY_TIMEOUT", 900, "experiment.evosci_verify_timeout")

# SciForge Experiment Validation
EXPERIMENT_TIME_BUDGET = _env_int("SCIFORGE_TIME_BUDGET", 300, "experiment.time_budget_seconds")
EXPERIMENT_MAX_ITERATIONS = _env_int("SCIFORGE_MAX_ITERATIONS", 100, "experiment.max_iterations")
EXPERIMENT_REPRODUCTION_ITERS = _env_int("SCIFORGE_REPRODUCTION_ITERS", 3, "experiment.reproduction_iters")
EXPERIMENT_PROXY_DATA_FRACTION = _env_float("SCIFORGE_PROXY_DATA_FRACTION", 0.15, "experiment.proxy_data_fraction")
EXPERIMENT_PROXY_MAX_EPOCHS = _env_int("SCIFORGE_PROXY_MAX_EPOCHS", 10, "experiment.proxy_max_epochs")
EXPERIMENT_EARLY_STOP_THRESHOLD = _env_float("SCIFORGE_EARLY_STOP_THRESHOLD", 0.20, "experiment.early_stop_threshold")
EXPERIMENT_REFUTE_MIN_ITERS = _env_int("SCIFORGE_REFUTE_MIN_ITERS", 30, "experiment.refute_min_iters")
EXPERIMENT_PLATEAU_PATIENCE = _env_int("SCIFORGE_PLATEAU_PATIENCE", 12, "experiment.plateau_patience")
# Real benchmark policy: by default DeepGraph must run real datasets/models, not
# synthetic proxy scaffolds. Set the synthetic fallback flag only for smoke tests.
EXPERIMENT_REQUIRE_REAL_BENCHMARK = _env_bool("DEEPGRAPH_REQUIRE_REAL_BENCHMARK", True, "experiment.require_real_benchmark")
EXPERIMENT_ALLOW_SYNTHETIC_FALLBACK = _env_bool("DEEPGRAPH_ALLOW_SYNTHETIC_FALLBACK", False, "experiment.allow_synthetic_fallback")
EXPERIMENT_REAL_LLM_MODEL = _env_str("DEEPGRAPH_REAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct", "experiment.real_llm_model")
EXPERIMENT_REAL_BENCHMARK_DATASET = _env_str("DEEPGRAPH_REAL_BENCHMARK_DATASET", "openai/gsm8k", "experiment.real_benchmark_dataset")
EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG = _env_str("DEEPGRAPH_REAL_BENCHMARK_DATASET_CONFIG", "main", "experiment.real_benchmark_dataset_config")
EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES = _env_int("DEEPGRAPH_REAL_BENCHMARK_MAX_EXAMPLES", 64, "experiment.real_benchmark_max_examples")
EXPERIMENT_REAL_BENCHMARK_SEEDS = _env_int("DEEPGRAPH_REAL_BENCHMARK_SEEDS", 3, "experiment.real_benchmark_seeds")
EXPERIMENT_REAL_BENCHMARK_TIME_BUDGET = _env_int("DEEPGRAPH_REAL_BENCHMARK_TIME_BUDGET", 3600, "experiment.real_benchmark_time_budget_seconds")
EXPERIMENT_WATCHDOG_ENABLED = _env_bool("DEEPGRAPH_EXPERIMENT_WATCHDOG_ENABLED", True, "experiment.watchdog_enabled")
EXPERIMENT_WATCHDOG_USE_LLM = _env_bool("DEEPGRAPH_EXPERIMENT_WATCHDOG_USE_LLM", False, "experiment.watchdog_use_llm")
# Closed-loop validation should prove the real runner and compare methods on a
# bounded real-data slice. Full publication evidence is scheduled separately and
# must opt in with DEEPGRAPH_BENCHMARK_FULL_RUN=1.
EXPERIMENT_VALIDATION_BENCHMARK_MAX_EXAMPLES = _env_int(
    "DEEPGRAPH_VALIDATION_BENCHMARK_MAX_EXAMPLES",
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
    "experiment.validation_benchmark_max_examples",
)
EXPERIMENT_VALIDATION_BENCHMARK_SEEDS = _env_int(
    "DEEPGRAPH_VALIDATION_BENCHMARK_SEEDS",
    EXPERIMENT_REAL_BENCHMARK_SEEDS,
    "experiment.validation_benchmark_seeds",
)
EXPERIMENT_VALIDATION_BENCHMARK_METHODS = _env_str(
    "DEEPGRAPH_VALIDATION_BENCHMARK_METHODS",
    "Vanilla Direct Answering,Always-Reason Chain-of-Thought,CGGR,CGGR/no_counterfactual_delta",
    "experiment.validation_benchmark_methods",
)
# After baseline reproduction fails (crash / no metric), run Codex or LLM repair rounds before giving up.
REPRODUCTION_REPAIR_MAX_ROUNDS = _env_int("DEEPGRAPH_REPRODUCTION_REPAIR_MAX_ROUNDS", 8, "experiment.reproduction_repair_max_rounds")
ALLOW_SMOKE_EXPERIMENT_VALIDATION = _env_bool("DEEPGRAPH_ALLOW_SMOKE_EXPERIMENT_VALIDATION", False, "experiment.allow_smoke_validation")
EXPERIMENT_WORKDIR = Path(_env_str("SCIFORGE_WORKDIR", str(Path.home() / "sciforge_runs"), "paths.experiment_workdir")).expanduser()
IDEA_WORKSPACE_DIR = Path(_env_str("DEEPGRAPH_IDEA_WORKSPACE_DIR", str(Path.home() / "deepgraph_ideas"), "paths.idea_workspace_dir")).expanduser()
RUNTIME_PYTHON = _env_str("DEEPGRAPH_RUNTIME_PYTHON", sys.executable, "runtime.python")
MLFLOW_TRACKING_URI = _env_str("DEEPGRAPH_MLFLOW_TRACKING_URI", "", "tracking.mlflow_uri")
CODEX_EXEC_ENABLED = _env_bool("DEEPGRAPH_CODEX_EXEC_ENABLED", True, "codex.exec_enabled")
CODEX_CLI_PATH = _env_str(
    "DEEPGRAPH_CODEX_CLI_PATH",
    shutil.which("codex") or "",
    "codex.cli_path",
)
CODEX_MODEL = _env_str("DEEPGRAPH_CODEX_MODEL", "", "codex.model")
CODEX_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_CODEX_TIMEOUT_SECONDS", 900, "codex.timeout_seconds")

# GPU scheduling / experiment lanes
GPU_WORKER_SLOTS = _env_int("DEEPGRAPH_GPU_WORKER_SLOTS", 4, "gpu.worker_slots")
GPU_VISIBLE_DEVICES = _split_csv(os.getenv("DEEPGRAPH_GPU_VISIBLE_DEVICES") or _toml_get("gpu.visible_devices", None)) or [str(i) for i in range(max(1, GPU_WORKER_SLOTS))]
GPU_DEFAULT_MODEL = _env_str("DEEPGRAPH_GPU_DEFAULT_MODEL", "H20", "gpu.default_model")
GPU_DEFAULT_VRAM_GB = _env_int("DEEPGRAPH_GPU_DEFAULT_VRAM_GB", 96, "gpu.default_vram_gb")
GPU_POLL_SECONDS = _env_int("DEEPGRAPH_GPU_POLL_SECONDS", 10, "gpu.poll_seconds")
GPU_STALE_RECOVERY_POLL_SECONDS = _env_int("DEEPGRAPH_GPU_STALE_RECOVERY_POLL_SECONDS", 120, "gpu.stale_recovery_poll_seconds")
GPU_JOB_TIMEOUT_SECONDS = _env_int("DEEPGRAPH_GPU_JOB_TIMEOUT_SECONDS", 14400, "gpu.job_timeout_seconds")

# Mechanism-first discovery
IDEA_EVIDENCE_MIN_NON_NUMERIC = _env_int("DEEPGRAPH_IDEA_EVIDENCE_MIN_NON_NUMERIC", 2, "idea.evidence_min_non_numeric")

# Manuscript / submission bundle
MANUSCRIPT_LATEX_TEMPLATE = _env_str("DEEPGRAPH_MANUSCRIPT_LATEX_TEMPLATE", "iclr2026", "manuscript.latex_template")
SUBMISSION_BUNDLE_FORMATS = _split_csv(os.getenv("DEEPGRAPH_SUBMISSION_BUNDLE_FORMATS") or _toml_get("manuscript.submission_bundle_formats", None)) or ["conference"]
MANUSCRIPT_WORKDIR = Path(_env_str("DEEPGRAPH_MANUSCRIPT_WORKDIR", str(Path.home() / "deepgraph_manuscripts"), "paths.manuscript_workdir")).expanduser()
# PaperOrchestra is the only supported manuscript backend.
MANUSCRIPT_BACKEND = _env_str("DEEPGRAPH_MANUSCRIPT_BACKEND", "paper_orchestra", "manuscript.backend").lower()
MANUSCRIPT_ALLOW_NEGATIVE_RESULTS = _env_bool(
    "DEEPGRAPH_MANUSCRIPT_ALLOW_NEGATIVE_RESULTS", True, "manuscript.allow_negative_results"
)
MANUSCRIPT_RELAX_COMPLETENESS_FOR_NEGATIVE_RESULTS = _env_bool(
    "DEEPGRAPH_MANUSCRIPT_RELAX_COMPLETENESS_FOR_NEGATIVE_RESULTS",
    True,
    "manuscript.relax_completeness_for_negative_results",
)
REFERENCE_PDF_CORPUS_DIR = Path(
    _env_str("DEEPGRAPH_REFERENCE_PDF_CORPUS_DIR", str(PROJECT_ROOT.parent.parent / "workspace" / "pdfs"), "paths.reference_pdf_corpus_dir")
).expanduser()
ICLR2026_TEMPLATE_DIR = Path(
    _env_str("DEEPGRAPH_ICLR2026_TEMPLATE_DIR", str(PROJECT_ROOT / "third_party" / "iclr2026" / "iclr2026"), "paths.iclr2026_template_dir")
).expanduser()
ICLR2026_TEMPLATE_FILES = _split_csv(_toml_get("manuscript.iclr2026_template_files", None)) or [
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    "math_commands.tex",
    "natbib.sty",
    "fancyhdr.sty",
]

# D2 (#13): paths for the four additional top-tier venue templates.
# Each adapter consults its own ``_TEMPLATE_DIR`` + ``_TEMPLATE_FILES`` pair
# so a venue can be swapped to a different upstream snapshot via env vars
# without touching adapter code.
NEURIPS2024_TEMPLATE_DIR = Path(
    _env_str("DEEPGRAPH_NEURIPS2024_TEMPLATE_DIR", str(PROJECT_ROOT / "third_party" / "neurips2024"))
)
NEURIPS2024_TEMPLATE_FILES = ["neurips_2024.sty", "README.md"]

ICML2024_TEMPLATE_DIR = Path(
    _env_str("DEEPGRAPH_ICML2024_TEMPLATE_DIR", str(PROJECT_ROOT / "third_party" / "icml2024"))
)
ICML2024_TEMPLATE_FILES = [
    "icml2024.sty",
    "icml2024.bst",
    "fancyhdr.sty",
    "algorithm.sty",
    "algorithmic.sty",
    "README.md",
]

ACL_ARR_TEMPLATE_DIR = Path(
    _env_str("DEEPGRAPH_ACL_ARR_TEMPLATE_DIR", str(PROJECT_ROOT / "third_party" / "acl_arr"))
)
ACL_ARR_TEMPLATE_FILES = ["acl.sty", "acl_natbib.bst", "README.md"]

CVPR2024_TEMPLATE_DIR = Path(
    _env_str("DEEPGRAPH_CVPR2024_TEMPLATE_DIR", str(PROJECT_ROOT / "third_party" / "cvpr2024"))
)
CVPR2024_TEMPLATE_FILES = ["cvpr.sty", "ieeenat_fullname.bst", "README.md"]

# Manuscript venue routing config (issue #11/#12 D1).
# Set ``DEEPGRAPH_VENUES_CONFIG_PATH`` to point at a custom YAML/JSON file.
VENUES_CONFIG_PATH = Path(
    _env_str(
        "DEEPGRAPH_VENUES_CONFIG_PATH",
        str(PROJECT_ROOT / "manuscript_venues" / "venues_v1.yaml"),
    )
)

# PaperOrchestra full §4 pipeline (S2 + parallel plot/lit + AgentReview loop)
SEMANTIC_SCHOLAR_API_KEY = _env_str("DEEPGRAPH_SEMANTIC_SCHOLAR_API_KEY", "", "paper_orchestra.semantic_scholar_api_key")
# Optional: shell command template for PaperBanana-style diagrams; must write output image path.
# Example: python /path/to/PaperBanana/run.py --out {output} --spec '{spec}'
PAPERBANANA_CMD = _env_str("DEEPGRAPH_PAPERBANANA_CMD", "", "paper_orchestra.paperbanana_cmd")
PAPERORCHESTRA_REFINEMENT_ITERS = _env_int("DEEPGRAPH_PAPERORCHESTRA_REFINEMENT_ITERS", 4, "paper_orchestra.refinement_iters")
PGVECTOR_EMBEDDING_DIM = _env_int("DEEPGRAPH_PGVECTOR_EMBEDDING_DIM", 1536, "database.pgvector_embedding_dim")
PIPELINE_EVENT_POLL_SECONDS = _env_int("DEEPGRAPH_PIPELINE_EVENT_POLL_SECONDS", 5, "pipeline.event_poll_seconds")

# Auto Research orchestration
AUTO_PIPELINE_ENABLED = _env_bool("DEEPGRAPH_AUTO_PIPELINE_ENABLED", False, "auto_research.pipeline_enabled")
AUTO_PIPELINE_BATCH_SIZE = _env_int("DEEPGRAPH_AUTO_PIPELINE_BATCH_SIZE", 100, "auto_research.pipeline_batch_size")
AUTO_PIPELINE_INTERVAL_SECONDS = _env_int("DEEPGRAPH_AUTO_PIPELINE_INTERVAL_SECONDS", 120, "auto_research.pipeline_interval_seconds")
AUTO_PIPELINE_START_DELAY_SECONDS = _env_int("DEEPGRAPH_AUTO_PIPELINE_START_DELAY_SECONDS", 10, "auto_research.pipeline_start_delay_seconds")
AUTO_RESEARCH_ENABLED = _env_bool("DEEPGRAPH_AUTO_RESEARCH_ENABLED", True, "auto_research.enabled")
AUTO_RESEARCH_INTERVAL_SECONDS = _env_int("DEEPGRAPH_AUTO_RESEARCH_INTERVAL_SECONDS", 300, "auto_research.interval_seconds")
AUTO_RESEARCH_MAX_ACTIVE = _env_int("DEEPGRAPH_AUTO_RESEARCH_MAX_ACTIVE", 1, "auto_research.max_active")
# When True: novelty must be 'novel', EvoScientist final_report.md must exist, and EvoSci
# must be installed before experiment forge / validation loop (see agents/evosci_requirements.py).
REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS = _env_bool("DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS", False, "experiment.require_evoscientist")

# Web
WEB_HOST = _env_str("DEEPGRAPH_WEB_HOST", "0.0.0.0", "web.host")
WEB_PORT = _env_int("DEEPGRAPH_WEB_PORT", 8080, "web.port")
