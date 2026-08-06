# Docs cleanup task brief (2026-08-06)

给执行本任务的 Claude Code session: 这是 2026-08-06 全量文档体检的执行清单。
体检范围是仓库顶层 8 份 markdown + docs/ 下 33 份, 结论已由 owner 过目。
用中文和 owner 对话。

## 背景

- 仓库: /home/ec2-user/Deepgraph-meta-harness-v1 (共享 worktree, 多个 session 同时工作)
- 2026-08-05: 根 README 已按两账制重写并提交 (627614e), 未 push, push 由 owner 负责
- 627614e 之后已有其他 session 的提交 (c0af044, 566d2e8, eb627f1 ...)
- 开工前必须先 git log -5 和 git status 重新确认现状, 不要假设本文件写下时的状态仍然成立

## 不可协商的真实性规则 (与根 README 口径一致)

- 系统没有产生任何科学发现; scientific_decisions_total = 0 (截至 2026-08-04)
- 运营数字 (论文数/idea 数/run 数) 是投入和过程, 不是成果; 所有统计必须带日期
- 两账制不得混写: operational (RUN) 与 scientific (EVIDENCE/DECIDED) 是两个登记簿
- 不得夸大自治: 全局自动开关当前为 OFF
- 首个 pilot (OutcomeRecord id=1: 2612 tokens, failed, inconclusive, 阶梯停在 planned,
  未用预留已退还) 只能作为 "记账诚实" 的证据, 不能作为成果

## 边界

- 只动仓库文件; 不碰线上服务, 数据库, .env, systemd, /home/billion-token 下任何东西
- 共享 worktree: 每次提交前 git status, 只提交本任务书涉及且确为你改动的文件,
  绝不把其他 session 的文件扫进 commit
- 不 push; owner 负责 push。完成后停下, 汇报每个 commit 动了什么和为什么
- CHANGELOG.md 由合作伙伴维护 (2026-06-10 决定), 不许编辑
- 本机没有 python3.12, 测试套件跑不起来; 引用测试结果一律注明出处文档和日期,
  不要伪造运行结果

## 任务 0 (先问, 不阻塞其他任务): LATENT_COMMUNICATION_RESEARCH.md

仓库根的这份文档 (2026-04-21) 声称 "已验证 512 字节压缩潜向量在 GSM8K 上达到
与全 KV 传递同等准确率 (91%)"。这是全仓库唯一一句既成事实式的科学结论, 与
README 的 "无科学发现" 主张正面冲突, 且仓库内查不到任何证据链。只有 owner
知道 91% 的来历。用 AskUserQuestion 问一次, 选项:

  a) 归档 (推荐): git mv 到 docs/archive/, 顶部加免责头 -- 历史 pitch 文档,
     其中主张未经过本系统证据阶梯, 不代表系统结论
  b) 删除
  c) owner 提供来历, 据实改写

得到答案前不要动这份文件; 先做任务 1-4, 最后回来处理。

## 任务 1: 2026-04-21 交接三件套统一归档

HANDOFF.md, SYSTEM.md, PRD_DEEPGRAPH_HANDOFF_2026-04-21.md -- 三份都停在
2026-04-21 交接 (commit e12fe16) 时代: 论文数字 2,790/2,566 与现口径 6,729
矛盾, 自我介绍 ("automatically discovers genuine research opportunities",
"可运行的研究闭环雏形") 是 README 刚剪掉的宣传口径, PRD 里的代码路径全是
旧仓库布局 (Deepgraph/ 前缀), 现在指不到。

处理: git mv 到 docs/archive/2026-04-21-handoff/, 每份顶部加统一免责头:
"本文件是 2026-04-21 交接快照。其后系统经历 meta-harness v1 与
recovery/frontend merge 两代升级, 架构与全部数字均已过时。现状见根 README。"
不重写正文。

## 任务 2: docs/integration/ 拆分 + docs 索引

docs/integration/ 现在 20 份文档里一次性会话过程文档和长期参考混装。

- 移入 docs/integration/archive/ (移动前逐份扫一眼, 若发现仍被活文档引用的
  内容, 留下并在汇报里说明): SESSION_STATUS.md, COMMIT_PLAN.md, BASELINE.md,
  CI_VALIDATION.md, DURABLE_QUEUE_VALIDATION.md, EVALUATOR_ISOLATION.md,
  ACCEPTANCE_EVIDENCE.md, UNVERIFIED.md, IMPLEMENTATION_CHECKLIST.md,
  PORTING_LEDGER.md, LLM_CALLER_INVENTORY.md, LEGACY_TEST_CLASSIFICATION.md
- 留在原地 (长期参考): CONFIGURATION.md, STATE_DICTIONARY.md,
  MIGRATION_RUNBOOK.md, ROLLBACK_RUNBOOK.md, CANARY_RUNBOOK.md,
  SCHEMA_DIFF.md, ARCHITECTURE.md, ISOLATED_CI_EVIDENCE_TEMPLATE.md
- docs/ 根的三份任务书 (README_OVERHAUL_TASK.md,
  AGENDA7_RESEARCH_DRIVER_TASK.md, FRONTEND_CLAUDE_CODE_DISCOVERY_TASK.md)
  和本文件一起移入 docs/tasks/
- 新建 docs/README.md 索引: 按 "现状参考 / runbooks / 前端 / 任务书 / 归档"
  分组, 每份一行说明。注意 README 底部 Further reading 表里已列的 8 份,
  索引与其保持一致
- 所有移动完成后, 全仓库跑一遍 markdown 链接检查 (2026-08-06 体检时是 0 断链,
  移动后必须恢复到 0), 修掉因移动断掉的链接, 包括根 README 里指向被移动
  文件的链接

## 任务 3: 补 STATE_DICTIONARY.md

docs/integration/STATE_DICTIONARY.md 自称状态字典, 但落后于系统现用词汇。
补齐以下条目, 每条给定义 + 源码出处 (file:line), 逐条从源码核对, 不要
从本任务书照抄而不验证:

- topic gate 的 parked (无预测的候选被停放; 相关: 无 TOPIC_GATE_ENABLED
  开关, gate 是纯函数)
- 证据阶梯 refusal blocker 串, 见 meta_harness/evidence_state.py:
  raw_artifacts_missing, holdout_ref_missing,
  benchmark_contract_hash_missing_or_invalid,
  positive_evidence_decision_failed, pilot_cannot_complete_full_benchmark,
  reviewer_approval_required (refusal 直接 raise, 不落库)
- decide_evidence 的 fail-closed blocker, 见
  contracts/scientific_evidence.py:74-135: metric_missing, baseline_missing,
  baseline_zero, full_benchmark_incomplete, raw_artifacts_incomplete,
  claim_ledger_incomplete, independent_evaluator_missing, p_value_missing,
  not_significant
- self-heal 的 3 个 restart_* 理由与 hold_* 家族, 见
  orchestrator/selfheal_policy.py:37-57
- backend 能力三态 enabled / unknown / disabled (无 fallback)
- ResourceGrant 释放理由 grant_expired / grant_revoked

## 任务 4: P2 零碎修补 (合并为一个 commit)

- docs/paper_grade_benchmark_agent_plan.md: 顶部把两份不存在的文档
  (docs/cggr_run32_paper_artifact_gate.md, docs/cggr_current_completion_audit.md)
  指为权威合同。加注说明这两份不存在于本仓库, 旧矩阵仅供历史参考;
  或整份归档, 二选一并说明理由
- docs/runbooks/SELFHEAL.md: 引用的 deploy/manifest/selfheal_v2.json 不存在
  (deploy/manifest/ 下只有 bounded-pilot_2026-08-04, frontend_2026-08-04,
  recovery_2026-08-03)。改指现存 manifest 或注明
- docs/top_venue_manuscript_chain.md: H:\Deepgraph\workspace\pdfs 是另一台
  Windows 机器的路径, 加注说明
- docs/deepgraph_ppt_content.md: 顶部注明数据为 2026-06-22 快照且采自
  8081 端口 (当前 8080), 已过时
- fix.md (仓库根, 2026-06-02 启动韧性修复报告): 移入 docs/archive/
- docs/cloud_gpu_pain_points.md: 顶部注明代码路径为旧仓库布局 (Deepgraph/
  前缀), 现已迁移

## 任务 5 (可选, 先问 owner 要不要): docs/ROADMAP.md

README 重写时删掉了内嵌 changelog 里的 3 周计划和 18 周路线图 (旧内容可用
git show 627614e~1:README.md 找回)。体检结论: 3 周计划大部分已实现。若 owner
要, 新建 docs/ROADMAP.md: 已建成的质量闸门 (带 file:line), 尚未实现的条目
(从旧计划逐条对照源码核实, 不要凭印象), 以及 18 周路线图为何作废。

## 交付方式

- 每个任务一个 commit, message 用 "docs:" 前缀, 说清动机
- 顺序建议: 1 -> 2 -> 3 -> 4 -> 0 -> (5)
- 完成后停下, 用中文汇报: 每个 commit 的内容, 遇到的意外, 留给 owner 的决定
