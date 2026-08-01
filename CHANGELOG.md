# Changelog

## 0.2.0 — meta-harness-v1 (accepted scoped release)

Release scope: CPU + SSH A100. Colab is implemented but explicitly excluded
from this release. Master merge and deployment are separate operator actions.

### Added

- Agenda-scoped Frontier/Decision/ResourceGrant/Outcome contracts with hard
  token/GPU caps, explicit backlog import, and fail-closed legacy paths.
- Durable PostgreSQL migration, compute claims, idempotency, usage settlement,
  restart recovery, and unknown-outcome quarantine.
- CPU and SSH GPU backends with secret-reference credentials and strict SSH
  known-host pinning.
- Role-separated LLM routing with metering, durable cooldowns, retry policy,
  and failure observations.
- Hash-pinned bubblewrap held-in/held-out/canary evaluation and signed,
  subject-bound reviewer approval.
- Durable scoped ingestion and Colab queue contracts; Colab remains disabled
  in the 0.2.0 release scope.

### Verification

- Physical disposable PostgreSQL restore: migration twice, 48 table counts
  preserved including `claims`, FK/orphan/scope checks, and repository tests.
- Policy/fault/evaluator lanes passed; adapted legacy failures remain explicitly
  classified without weakening grant or scope rules.
- CPU/API, SSH A100, real provider, provider cooldown restart, and reviewer
  approval passed in isolation. No production database, push, deployment or
  remote-ref deletion occurred.

## 0.1.x — previous DeepGraph engine

Literature ingestion, evidence graph construction, opportunity discovery,
experiment orchestration, manuscript generation, and the original dashboard.
See Git history for the detailed pre-0.2.0 development record.
