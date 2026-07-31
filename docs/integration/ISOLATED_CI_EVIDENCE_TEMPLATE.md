# Isolated CI evidence template

Copy this file into the approved evidence store for one immutable candidate.
Do not commit database URLs, credentials, OAuth material, raw reviewer
signatures or business-row contents.

```text
candidate_commit: d33a9f5fbb1bb912f6edff2f87b749d38ec19d25
candidate_tree: 18a5a677ee13ed81d550710c5c390ae3e3b3c23c0991036af465237f164abe2f
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
  verification_git_tree: 607a1fb701357aad77c7003743093f51ab867ce2

lane_1_static:
  status: passed
  static_report_sha256: e091c3d4484b4d59b5ed0af2355f1dcb4d22c9f26ec23abd4358b89f5408927d
  scope_report_sha256: 4038c4265f3d05d1a83dc1a52295e94f5c98ee89c20eb1f92321ae440ade840d
  sql_report_sha256: 286c3c35345b3c3dbaa2b653adb89cb7a3e8416547904a7d39cdaf2142fd9ca7
  state_authority_report_sha256: fbb44b6ad15756f980fd654efb0acf2dc810565367423fa4f917001cf9865b73
  llm_report_sha256: 90da4875b8e4cf2e24c1016fb87dbbe884a04c61a43ec22555918572ea673ad9
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
  test_report_sha256: f03d5c20230a5e3b048f0e203f77466578a094309110b6001e044cc34f1c068b
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
  post_fix_real_rerun_status: passed (held-in/held-out/canary)
  post_fix_real_rerun_report_sha256: adfc86924e309f063aa36a47afd205f5b8303fb5b928b1999fa301aee5a2cb09
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
evidence collected on 2026-07-31. The real bubblewrap held-in, held-out and
canary evaluator lanes passed, including protected-write and
missing-isolation-binary negative checks, with an unchanged candidate tree.
The PostgreSQL lane passed against a local disposable synthetic schema restore;
no production dump was supplied, so real backup-row preservation remains open.
The adapted legacy lane remains 39 passed / 30 failed with each failure
classified as obsolete or requiring a new scoped/granted fixture. Approved
CPU/GPU/Colab hardware canaries and reviewer approval were not authorized or
available. Consequently `all_16_gates_accepted` and
`master_replacement_approved` remain false.
