# Isolated CI evidence template

Copy this file into the approved evidence store for one immutable candidate.
Do not commit database URLs, credentials, OAuth material, raw reviewer
signatures or business-row contents.

```text
candidate_commit: d3650fe0a2270eb265ef9dc40041b3ccab537efd
candidate_tree: c9a2efac23e30abda6c9ab87242d76ecfa66d6679272404fbf27402d86db6114
source_candidate_commit: 6851a991154906f11d8cfc247d22a5d5caa0a834
source_candidate_tree_fixture: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
source_archive_refs:
  production: 7d0b42af8e8f061c3c16800c44224c110f3b94a0
  github_master: 6048a9568c79b011074e0dba2662fd473cfab250
  topic_gate: 9d24d29c6a7d1017301ffa9c36ff9b4b3dfae88d
ci_run_id: local-20260731-meta-harness-final
ci_image_digest: local-venv-python3.13.14-no-pinned-image
operator: Codex isolated workspace
started_at_utc: not-instrumented
completed_at_utc: 2026-07-31T11:05:59Z

isolation:
  disposable_database_name_hash: none-no-postgresql-server
  production_url_unset: true
  production_path_read_only: true
  network_disabled_in_evaluator: true
  candidate_tree_before: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  candidate_tree_after: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  verification_git_tree: 7e3183cd039cb7bace420355a6aad6b0a67f1358

lane_1_static:
  status: passed
  broad_python_ast_count: 260
  release_python_ast_count: 260
  scoped_mutation_count: 154
  unscoped_mutation_count: 0
  sql_literal_count: 839
  sql_countable_count: 836
  sql_dynamic_review_count: 114
  migration_statement_count: 90
  migration_bytes: 27734
  migration_sha256: 3b73e0647c5edfb13f82efbba79081b29f19734a4504a4327e4eabdbf06241f0
  migration_destructive_tokens: []

lane_2_policy:
  status: passed (71 passed)
  test_report_sha256: 76c9e6d391f2e6d62b21e29a49b6fb56125a043adaf660d5fd9d391f55cd2669
  validation_loop_fault_status: passed (22 passed, 0 failed)
  validation_loop_fault_report_sha256: 8c7f1fdad2f6e5c3a60fb237d75de6e9f9d84e96af3022b6b274f4b843168075
  synthetic_fault_status: passed (60 passed, 0 failed)
  synthetic_fault_report_sha256: d41e3e25974b8dd8b9343e3b0b81cfe6a0a75088aac7017823f410a055d30f97

lane_3_postgresql:
  migration_first_status: blocked_not_run_no_disposable_postgresql
  migration_second_status: blocked_not_run_no_disposable_postgresql
  preexisting_counts_preserved: false
  compute_restart_status: guarded_tests_skipped_postgresql_unavailable
  colab_queue_fault_status: guarded_tests_skipped_postgresql_unavailable
  scoped_ingestion_fault_status: guarded_tests_skipped_postgresql_unavailable
  test_report_sha256: not-run-no-postgresql-server-or-docker

lane_4_evaluator:
  held_in_evaluator_hash: 45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d
  held_in_suite_hash: 69c862126118dcc05cfe8fb10ba9fda4feeff8e67db5f3904923040bad389dda
  held_in_manifest_hash: 36f869064a5e928ad14c54be30e5038f2221191d2995a7a33eb43b96e9fb6e21
  held_out_evaluator_hash: 45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d
  held_out_suite_hash: 901e491cb69ec30a6f5d21a1a537f4a73d4f76a393ce907ff570fb072707e463
  held_out_manifest_hash: 1cecaee031b94841516bb8c7951502bc4b0432add8f676f7f5a948298b36007d
  canary_evaluator_hash: 45bddd4cbd5eba5ba6a6377b765debcd4e67fa15fcab879bbb321c752d6c362d
  canary_suite_hash: 21c44b3011bdfbe5f9bec5856f0349ce579d1c06dc4edd9fb75c0717884a2480
  canary_manifest_hash: f17ad8fc961e48869fa619aab4da32488288eb3a311b2a696b4efa44a9e71367
  protected_write_negative_test: true (result 1b65ec3b98c5dec4acfd447c8c8a08bbad9f19d1de6a9c48f2f4f0c539d1d4b0)
  network_negative_test: true (probe blocked)
  unisolated_fallback_negative_test: true (refused missing bwrap)
  post_fix_real_rerun_status: blocked_host_bwrap_netlink_route
  post_fix_real_rerun_report_sha256: 63230055746e4949ab7337f70f94786b77edc77f87883eb2e56a0a487cf1ae0c
  post_fix_candidate_tree_before_after: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  post_fix_mock_contract_status: passed (1 passed)
  post_fix_mock_contract_report_sha256: 99a24796443a1d07f66a424f8116a06a2079063af758e7d3c87801c191ad2ecd
  post_fix_missing_bwrap_fallback_status: passed
  post_fix_missing_bwrap_fallback_report_sha256: 334df8dd61952b1b911e62d74cd7ecd27fd728dabf7980889c19d3fe200aa698

lane_5_fault_canary:
  synthetic_provider_status: passed targeted router/cooldown tests; synthetic aggregate 60/60
  synthetic_backend_status: passed targeted backend/app/fault tests; PostgreSQL backend recovery unverified
  duplicate_submission_count: 0 observed in mocked/durable queue tests; PostgreSQL unverified
  unknown_usage_quarantine_status: implemented and mock-tested; PostgreSQL unverified
  approved_cpu_canary_status: not_run_no_approved_canary
  approved_gpu_canary_status: not_run_no_gpu-or-colab-credentials

review:
  reviewer_approval_record_id: none
  reviewer_signature_hash: none
  rollback_rehearsal_status: passed_isolated_7d0b42a_temp-startup
  all_16_gates_accepted: false
  master_replacement_approved: false
```

The adapted legacy lane is intentionally retained as a separate audit lane:
39 passed / 30 failed. See
[LEGACY_TEST_CLASSIFICATION.md](LEGACY_TEST_CLASSIFICATION.md) for the
test-by-test rejected/obsolete or new-contract classification. No compatibility
shim restores grantless, unscoped, password-bearing or unlimited behavior.

Every `false`, empty field or missing hash remains a blocker. A code-defined
test or runbook is not evidence that its corresponding field passed.

## Recorded final-session result

The block above records the immutable source candidate and the disposable
evaluator evidence collected on 2026-07-31. The real bubblewrap held-in,
held-out and canary evaluator lanes passed, including the protected-write and
missing-isolation-binary negative checks. The post-fix real evaluator rerun was
blocked by the host bwrap network namespace restriction; its candidate tree
hash remained unchanged. The PostgreSQL lane could not run:
this host had `psql` but no server, `initdb`/`pg_ctl`, or usable Docker daemon.
The CPU/GPU/Colab canary lanes and reviewer approval were not authorized or
available. Consequently `all_16_gates_accepted` and
`master_replacement_approved` remain false.
