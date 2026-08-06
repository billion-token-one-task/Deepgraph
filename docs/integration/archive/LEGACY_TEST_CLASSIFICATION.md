# Adapted legacy test classification

Run: `local-20260731-meta-harness-legacy-after` (Python 3.13 isolated venv,
production database variables unset). Result: **39 passed / 30 failed**;
report SHA-256 `648abb2ccea4d53536cb31352d0c8c1417e293b69663cdde7974b352a6f81396`.

The failures below are classified individually. The legacy lane is not a
release gate for restoring the removed contracts: no grant, agenda scope,
database isolation, or fail-closed behavior was relaxed to make these tests
pass. Replacement coverage is provided by the policy lane and the guarded
PostgreSQL integration modules.

| Test | Classification | Decision |
|---|---|---|
| `test_experiment_forge.py::GenerateScaffoldTests::test_checkpoint_run_state_serializes_incremental_fields` | Old unscoped checkpoint API; `agenda_id` is now mandatory. | rejected/obsolete; replacement must pass agenda scope |
| `test_experiment_forge.py::GenerateScaffoldTests::test_fallback_scaffold_produces_real_benchmark_train_py` | Old synthetic/real benchmark fallback expectation; generic code now fails closed without an audited runner plugin. | rejected/obsolete |
| `test_experiment_forge.py::GenerateScaffoldTests::test_generate_scaffold_accepts_evidence_plan` | Old direct `call_llm_json` surface; scaffold generation now requires a granted role route and scope. | rejected/obsolete; adapt only with a scoped route fixture |
| `test_experiment_forge.py::GenerateScaffoldTests::test_generate_scaffold_injects_real_benchmark_runner_for_gpu_route` | Same removed direct-LLM API and unscoped GPU scaffold contract. | rejected/obsolete; adapt only with a scoped route fixture |
| `test_experiment_forge.py::GenerateScaffoldTests::test_generated_runner_refuses_cross_domain_gsm8k_fallback` | Safety behavior remains refusal, but the test asserts an obsolete topic-specific diagnostic string; the generic audited-runner blocker is authoritative. | rejected/obsolete; semantic refusal retained |
| `test_validation_loop.py::ValidationLoopGitFallbackTests::test_determine_final_verdict_marks_reproduction_only_runs` | Old `reproduced` scientific verdict; reproduction without a positive, complete evidence package is now `inconclusive`. | rejected/obsolete |
| `test_validation_loop.py::ValidationLoopGitFallbackTests::test_run_validation_loop_blocks_non_formal_experiment` | Fixture has no agenda/grant. New validation checks the active scoped grant before the non-formal branch. | rejected/obsolete; adapt fixture to the new grant contract |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTimeoutPolicyTests::test_queue_run_preserves_zero_timeout_as_uncapped` | Old zero-timeout/unlimited behavior; hard timeout zero is now rejected. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_claim_worker_ignores_idle_worker_with_running_job` | Legacy unscoped GPU fixture; queue/recovery now requires agenda, idea, grant and durable identity. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_allows_legacy_gsm8k_manifest_for_math_prm_run` | Legacy topic-manifest scheduler path; generic topic aliases are no longer allocation authority. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_allows_legacy_mbpp_manifest_for_formal_code_run` | Legacy unscoped GPU queue fixture and worker path. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_blocks_legacy_gsm8k_manifest_for_agent_workflow_run` | Legacy topic-manifest policy path, excluded from the generic registry. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_blocks_legacy_gsm8k_manifest_for_formal_run` | Legacy topic-manifest policy path, excluded from the generic registry. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_blocks_legacy_gsm8k_manifest_for_physical_spatial_run` | Legacy topic-manifest policy path, excluded from the generic registry. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_blocks_legacy_mbpp_manifest_for_molecular_equivariant_run` | Legacy topic-manifest policy path, excluded from the generic registry. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_next_job_fails_recipe_blocked_run_without_launching` | Legacy worker fixture does not carry the durable scoped compute claim. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_periodic_recovery_runs_after_poll_interval` | Legacy recovery hook; current startup recovery is agenda-scoped and guarded by the scheduler lock. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_queue_run_creates_gpu_job` | Old unscoped queue insertion; current API requires matching active ResourceGrant and (PostgreSQL) ComputeScheduler identity. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_queue_run_downshifts_gpu_large_vram_to_schedulable_worker` | Same removed unscoped queue contract. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_recover_completed_experiment_with_open_manuscript_requeues` | Legacy recovery mutation lacks agenda/grant scope and durable compute identity. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_recover_skips_active_local_job_without_gpu_process` | Legacy recovery fixture lacks the scoped run/grant contract. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_recover_stale_local_running_job_requeues_after_restart` | Legacy stale-job recovery fixture lacks agenda/grant scope; current recovery is fail-closed for unscoped rows. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_register_default_workers_does_not_fabricate_local_gpu_without_inventory_or_visible_devices` | Test setup depends on removed legacy scheduler globals and unscoped worker lifecycle. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_register_ssh_workers` | Expects password material in the scheduler module; SSH now accepts credential references only. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_release_worker_stays_busy_when_another_job_is_running` | Legacy worker release fixture inserts unscoped jobs. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_run_job_blocks_manuscript_until_benchmark_manifest_is_complete` | Legacy GPU worker execution path bypasses durable compute admission and scoped grant validation. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_run_job_bundle_failure_does_not_overwrite_completed_experiment` | Same removed legacy execution path; terminal settlement is now durable and grant-scoped. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_run_job_handles_none_validation_result` | Legacy direct worker execution and unscoped fixture; current failures cannot become success. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_run_job_uses_full_benchmark_completion_stage` | Legacy worker bypasses the durable compute claim and scientific evidence authority. | rejected/obsolete |
| `test_vnext_gpu_scheduler.py::GpuSchedulerTests::test_ssh_recovery_skips_job_active_in_this_process` | Legacy SSH recovery fixture is unscoped and expects removed credential/config behavior. | rejected/obsolete |

The classification is an audit conclusion, not a waiver. Any future adapted
test must construct a disposable agenda, active ResourceGrant, scoped run and
durable compute claim; it must not add default arguments that recreate the old
global/unlimited behavior.
