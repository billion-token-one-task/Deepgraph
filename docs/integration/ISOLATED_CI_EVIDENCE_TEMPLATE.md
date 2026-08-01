# Isolated CI evidence template

Copy this file into the approved evidence store for one immutable candidate.
Do not commit database URLs, credentials, OAuth material, raw reviewer
signatures or business-row contents.

```text
candidate_commit: a18dc4968b38290d40603c8909b17a888b57157c
candidate_tree: d12d6882aafda2780ba93563ced0b88780f346b7312306f382748b5d8128fcd9
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
completed_at_utc: 2026-07-31T19:00:37Z

isolation:
  disposable_database_name_hash: 8acad1a69532ce07220b7b91279e139ffa9b0e39c14d66c7cb181de03814cb40
  production_url_unset: true
  production_path_read_only: true
  network_disabled_in_evaluator: true
  candidate_tree_before: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  candidate_tree_after: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  verification_git_tree: 593663a7e36e769d28cfd14828b8b8ee92bbbd75

lane_1_static:
  status: passed
  static_report_sha256: 91c9dcc5af43b4b439191175a4dc6024fdb32e542de6e3a0d98d88416e8d564c
  scope_report_sha256: 0f8de09da9f5d057f1c3141eb84a47ff1863a7a908623e9358e6123ed3491e7f
  sql_report_sha256: 9cd62926be4148686f91dca409e6c6d2d8a8d2723b92844eec9a0dd2c2dabbd8
  state_authority_report_sha256: ab02a1636e736faa8fba8ec8aa00e3df8d0204857e5b74c4907557839d84acc5
  llm_report_sha256: 7ea25bcbf45728afb0c11edaf092dd32ecb28da53a045c2b6ce0b95ad751e5e4
  broad_python_ast_count: 260
  release_python_ast_count: 260
  scoped_mutation_count: 154
  unscoped_mutation_count: 0
  sql_literal_count: 839
  sql_countable_count: 836
  sql_dynamic_review_count: 114
  migration_statement_count: 91
  migration_bytes: 27823
  migration_sha256: 5ead56c64fc977b01c6ad29abe61f8a6da3c15995e414cdc752b45ef1bdfc912
  migration_destructive_tokens: []

lane_2_policy:
  status: passed (71 passed)
  test_report_sha256: 89b3a752b649ae4ebb1143c6b04f42a2c6849a9ac390bf1b83be13aaa7e260a0
  validation_loop_fault_status: passed (22 passed, 0 failed)
  validation_loop_fault_report_sha256: 34ecfb71ad791822cf59270e77c6244dd92b671d0b3e0098c093d7c512e86cde
  synthetic_fault_status: passed (60 passed, 0 failed)
  synthetic_fault_report_sha256: ac999ae38f1b85b3e4e02a5cd5369e686a2b929df72d000afd6f355b9db46d9f

lane_3_postgresql:
  restore_status: synthetic_schema_restore_passed (vector-neutralized disposable schema; no production dump supplied)
  migration_first_status: applied (clean disposable restore)
  migration_second_status: already_applied (checksum no-op)
  migration_report_sha256: ac03dc55bdc1ce87ab5a711fdf5a7d72d345e62c471c4cb4e9de283e03475f14
  preexisting_counts_preserved: true (synthetic baseline papers/deep_insights/auto_research_jobs)
  compute_restart_status: passed (4 passed)
  compute_report_sha256: 5cd5265d12717754cd621e2d12792e961e4c63e7a7a2656d4537bcd2eeb45b51
  evidence_status: passed (1 passed)
  evidence_report_sha256: 6262dd4ae1e1610248aa7a1d90c86e6b7861110a1a434aeede186690d3652770
  queue_fault_status: passed (multi-agenda reservation, Colab quarantine, scoped ingestion retry)
  queue_report_sha256: 587d554dae9e9f724fec3011be4f16ed3525b70a2c47ff34998d180c243b4e68
  sql_fk_scope_status: passed (zero orphan/cross-scope rows)
  sql_fk_scope_report_sha256: 12a71f92fe192ef98295f5ec941eecfc6dd00ed2f2e8c13fe46ce2f834d32496
  physical_backup_restore_status: passed (PostgreSQL 18, private pgvector 0.8.1, socket-only disposable instances)
  physical_migration_first_status: applied
  physical_migration_second_status: already_applied (checksum no-op)
  physical_preexisting_count_status: passed (48 tables, including claims=10270)
  physical_fk_orphan_scope_status: passed (zero integrity failures)
  physical_repository_integration_status: passed (6 passed)
  physical_first_report_sha256: efca3d6e23e77210a90a293b66a4f4cb24775632042732f4100aa5f52ebdff8c
  physical_second_report_sha256: 7abe42b953f114b8e29c89ffb80ad20e1af4459c6e095e9ba6f70cd27b47a071
  physical_counts_report_sha256: f6f127488f312c9e91eaa59d19c89107b250ee6a51df35afa1c6936a6631144f
  physical_integrity_report_sha256: ef06fef9b8bd896cfbd11f76d2fdafac1f0258cacb8f8a2707d21ab2e1b733e7
  physical_integration_report_sha256: 8b2f2ee0fc169e4c0ce96d5256b846a3ac33569e302f7ddc4ab5336518b050cf

lane_4_evaluator:
  final_candidate_commit: 211124be179f88480477c7eb87f7973c2acf096d
  final_candidate_tree_before_after: 2efd623d20b7301662c4071220368aa2569520ed6e7a1cb79d1b69955c629206
  final_candidate_real_rerun_status: passed (held-in, held-out, canary; read-only inputs; unshared network)
  final_candidate_real_rerun_report_sha256: e011366d03614ef239e889b84b0943d73b20c1734a254ebae5b15d129435074f
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
  post_fix_real_rerun_status: passed (held-in/held-out/canary)
  post_fix_real_rerun_report_sha256: adfc86924e309f063aa36a47afd205f5b8303fb5b928b1999fa301aee5a2cb09
  post_fix_candidate_tree_before_after: b1c1e8ebfbc0607cc39bb617dad9d56fd949d214122a37dd70541bd634d9feab
  post_fix_mock_contract_status: passed (1 passed)
  post_fix_mock_contract_report_sha256: 99a24796443a1d07f66a424f8116a06a2079063af758e7d3c87801c191ad2ecd
  post_fix_missing_bwrap_fallback_status: passed
  post_fix_missing_bwrap_fallback_report_sha256: 334df8dd61952b1b911e62d74cd7ecd27fd728dabf7980889c19d3fe200aa698

lane_5_fault_canary:
  ssh_reference_regression_status: passed (18 passed; no remote target contacted)
  ssh_reference_regression_report_sha256: d015b2f31ef3fe7900d05082b66c5ec8069a2ebe988abc3a1192c3455c86b15d
  ssh_host_key_regression_status: passed (19 passed total)
  ssh_host_key_regression_report_sha256: 6968763c69824dd8cc791358f5c9339e0aa99b6ad38f3a1033f997e2995dda8b
  target_3_read_only_probe_status: timed_out_after_30_seconds
  synthetic_provider_status: passed targeted router/cooldown tests; synthetic aggregate 60/60
  synthetic_backend_status: passed targeted backend/app/fault tests; PostgreSQL backend recovery unverified
  duplicate_submission_count: 0 observed in mocked/durable queue tests; PostgreSQL unverified
  unknown_usage_quarantine_status: implemented and mock-tested; PostgreSQL unverified
  approved_cpu_canary_status: passed_control_plane_cpu_and_api_with_short_lived_grant
  approved_gpu_canary_status: passed_control_plane_a100_ssh_with_secret_reference
  control_plane_canary_status: passed (CPU/A100 submit-settle, terminal idempotency rejection, submission_unknown quarantine)
  control_plane_canary_report_sha256: 526b61cf3c302201a73e8121d8ba159048291571c328c0dac498b14aca0970a9
  outcome_record_status: passed (trusted persistence assembly; outcome_record_id redacted from repository evidence)
  compute_evidence_rerun_report_sha256: f45f868d8aa06b206a57b1222c20fe0f56111900750d0ced0adc003a189e65c2
  scientific_gpu_colab_provider_status: not_claimed

review:
  reviewer_approval_record_id: isolated_report_2
  reviewer_signature_hash: 5350eeaba85c5deb34c45690b3ac07a84c1e970134e21b09adb9da73e950a795
  reviewer_id: service@diwenbao.co
  reviewer_key_id: aws-reviewer
  reviewer_purpose: harness_upgrade
  reviewer_subject: agenda/candidate/patch-bound (recorded in isolated report)
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
evidence collected on 2026-07-31, followed by its final-code physical-backup
revalidation. The real bubblewrap held-in, held-out and canary evaluator lanes
passed, including protected-write and missing-isolation-binary negative checks.
The physical PostgreSQL 18 backup restore now passes count preservation,
migration idempotency, integrity checks and all three repository integration
files. The adapted legacy lane remains 39 passed / 30 failed with each failure
classified as obsolete or requiring a new scoped/granted fixture. Approved
The control-plane CPU/A100 canary, API status check, trusted OutcomeRecord assembly and signed reviewer approval passed using a disposable PostgreSQL agenda,
short-lived ResourceGrant, and secret references only. It is not a scientific
benchmark or Colab/provider canary. Reviewer approval is still unavailable;
consequently `all_16_gates_accepted` and `master_replacement_approved` remain
false. Real LLM provider execution/restart remains unverified and is not in
the CPU + SSH A100 canary evidence.
