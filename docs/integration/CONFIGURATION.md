# meta-harness-v1 configuration

Non-sensitive policy lives in `deepgraph.toml`. Research direction is supplied
as an Agenda file/API payload. Credentials are referenced by environment or a
secret manager and must never be written into TOML, agenda files, traces, or
artifacts.

## Required policy sections

| Section | Purpose |
|---|---|
| `agenda` | active IDs, explicit backlog policy, token/GPU caps, concurrency and default backend allowlist |
| `portfolio` | transparent best-of-N weights, reserve buckets, kill/park thresholds |
| `resource_grants` | stage caps, TTL and required artifacts |
| `llm_routes` | proposer/evaluator/reviewer provider/model references and fail-closed policy |
| `compute_backends` | enabled registry, heartbeat, artifact root and secret references |
| `scientific_evidence` | canonical states and evidence requirements |
| `harness_evolution` | candidate root/namespace, configurable diff limits and required suites |
| `failure_policy` | truthful terminal states |
| `trace` | trace/artifact locations and provenance fields |

## Secret references

Examples are references, not values:

```toml
provider_ref = "env:DEEPGRAPH_LLM_EVALUATOR"
credential_ref = "env:DEEPGRAPH_SSH_CREDENTIAL"
accounts_manifest_ref = "env:DEEPGRAPH_COLAB_ACCOUNTS_MANIFEST"
```

The Colab adapter additionally requires isolated code/artifact roots and
per-account HOME, OAuth store, session namespace and quota. Candidate
environments strip inherited HOME, production database URLs, tokens, API keys,
OAuth, cookies, credentials, passwords and SSH variables.

## Defaults that intentionally fail closed

- token budget must be positive; `0` is never unlimited;
- GPU budget `0` disables GPU;
- old backlog is excluded until explicit import;
- only CPU is enabled in the example backend registry;
- the compute scheduler requires the PostgreSQL `compute_jobs_v1` store;
- ephemeral in-memory idempotency is disabled outside explicit unit tests;
- an unknown backend submission requires manual reconciliation and is never
  automatically retried on a different backend;
- SSH/LocalGPU/Colab are disabled;
- provider or compute unavailability does not create low-quality fallback
  output;
- provider cooldowns are durable; inability to load/save cooldown authority
  fails the granted route closed;
- operational `supported` does not authorize manuscript or positive learning
  without a canonical scientific decision;
- harness upgrade requires held-in, held-out, canary and reviewer approval.
