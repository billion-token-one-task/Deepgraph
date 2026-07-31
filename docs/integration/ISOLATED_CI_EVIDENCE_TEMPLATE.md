# Isolated CI evidence template

Copy this file into the approved evidence store for one immutable candidate.
Do not commit database URLs, credentials, OAuth material, raw reviewer
signatures or business-row contents.

```text
candidate_commit:
candidate_tree:
source_archive_refs:
  production:
  github_master:
  topic_gate:
ci_run_id:
ci_image_digest:
operator:
started_at_utc:
completed_at_utc:

isolation:
  disposable_database_name_hash:
  production_url_unset: false
  production_path_read_only: false
  network_disabled_in_evaluator: false
  candidate_tree_before:
  candidate_tree_after:

lane_1_static:
  status:
  broad_python_ast_count:
  release_python_ast_count:
  scoped_mutation_count:
  unscoped_mutation_count:
  sql_literal_count:
  sql_countable_count:
  sql_dynamic_review_count:
  migration_statement_count:
  migration_bytes:
  migration_sha256:
  migration_destructive_tokens:

lane_2_policy:
  status:
  test_report_sha256:

lane_3_postgresql:
  migration_first_status:
  migration_second_status:
  preexisting_counts_preserved: false
  compute_restart_status:
  colab_queue_fault_status:
  scoped_ingestion_fault_status:
  test_report_sha256:

lane_4_evaluator:
  held_in_evaluator_hash:
  held_in_suite_hash:
  held_in_manifest_hash:
  held_out_evaluator_hash:
  held_out_suite_hash:
  held_out_manifest_hash:
  canary_evaluator_hash:
  canary_suite_hash:
  canary_manifest_hash:
  protected_write_negative_test: false
  network_negative_test: false
  unisolated_fallback_negative_test: false

lane_5_fault_canary:
  synthetic_provider_status:
  synthetic_backend_status:
  duplicate_submission_count:
  unknown_usage_quarantine_status:
  approved_cpu_canary_status:
  approved_gpu_canary_status:

review:
  reviewer_approval_record_id:
  reviewer_signature_hash:
  rollback_rehearsal_status:
  all_16_gates_accepted: false
  master_replacement_approved: false
```

Every `false`, empty field or missing hash remains a blocker. A code-defined
test or runbook is not evidence that its corresponding field passed.
