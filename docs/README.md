# docs 索引

最后更新: 2026-08-06。按用途分组, 每份一行。根 README 底部 Further reading
表列出的 8 份核心文档在此均有对应条目。

## 现状参考

- [integration/ARCHITECTURE.md](integration/ARCHITECTURE.md) - meta-harness v1 控制平面架构详解
- [integration/STATE_DICTIONARY.md](integration/STATE_DICTIONARY.md) - 全部状态词汇的含义与边界 (两账制口径)
- [integration/CONFIGURATION.md](integration/CONFIGURATION.md) - 策略字段与 secret-reference 配置
- [integration/SCHEMA_DIFF.md](integration/SCHEMA_DIFF.md) - schema 增量与迁移策略 (仅基于源码 SQL 与 Git diff)
- [agent_architecture.md](agent_architecture.md) - agent 目录归属边界 (兼容优先布局)
- [reflection-20260806-autonomy-repair.md](reflection-20260806-autonomy-repair.md) - 自治链条为何没走通: 反思与修复方案 (2026-08-06)
- [paper_grade_benchmark_agent_plan.md](paper_grade_benchmark_agent_plan.md) - 早期通用 benchmark 计划, 旧矩阵仅供历史参考 (见文内状态注)
- [top_venue_manuscript_chain.md](top_venue_manuscript_chain.md) - 论文生产标准与强制它的管线改动
- [deepgraph_ppt_content.md](deepgraph_ppt_content.md) - 项目展示 PPT 内容稿 (2026-06-22 数据快照, 已过时)
- [cloud_gpu_pain_points.md](cloud_gpu_pain_points.md) - 云 GPU (SSH 远端) 与编排痛点交接 (代码路径为旧仓库布局)
- pre_registrations/ - 实验预注册 JSON (当前: agenda7_idea98.json)

## Runbooks

- [runbooks/RECOVERY_2026-08-03.md](runbooks/RECOVERY_2026-08-03.md) - self-heal, topic gate, frontier authority, backend truth 与部署基线
- [runbooks/FRONTIER_BOOTSTRAP.md](runbooks/FRONTIER_BOOTSTRAP.md) - 为没有 packet 的 Agenda 签发并审计 bootstrap authority
- [runbooks/SELFHEAL.md](runbooks/SELFHEAL.md) - watchdog 理由码与操作员响应
- [integration/MIGRATION_RUNBOOK.md](integration/MIGRATION_RUNBOOK.md) - 隔离 PostgreSQL 迁移 runbook (不授权碰生产)
- [integration/ROLLBACK_RUNBOOK.md](integration/ROLLBACK_RUNBOOK.md) - 回滚与生产参照 runbook
- [integration/CANARY_RUNBOOK.md](integration/CANARY_RUNBOOK.md) - 候选与算力 canary runbook
- [integration/ISOLATED_CI_EVIDENCE_TEMPLATE.md](integration/ISOLATED_CI_EVIDENCE_TEMPLATE.md) - 隔离 CI 证据模板 (每个不可变候选填一份)

## 前端

- [frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md](frontend/FRONTEND_DISCOVERY_AND_RECOMMENDATION.md) - UI 为什么长这样; 未决产品决策
- [frontend/FRONTEND_MERGE_IMPLEMENTATION.md](frontend/FRONTEND_MERGE_IMPLEMENTATION.md) - frontend merge v1 实际改了什么

## 任务书 (tasks/)

给各 Claude Code session 的一次性任务简报, 完成后保留作过程记录。

- [tasks/README_OVERHAUL_TASK.md](tasks/README_OVERHAUL_TASK.md) - README 重写任务 (产出 627614e)
- [tasks/AGENDA7_RESEARCH_DRIVER_TASK.md](tasks/AGENDA7_RESEARCH_DRIVER_TASK.md) - agenda 7 监督研究驱动任务
- [tasks/FRONTEND_CLAUDE_CODE_DISCOVERY_TASK.md](tasks/FRONTEND_CLAUDE_CODE_DISCOVERY_TASK.md) - 前端 discovery 任务
- [tasks/DOCS_CLEANUP_TASK.md](tasks/DOCS_CLEANUP_TASK.md) - 本次文档清理任务 (2026-08-06)

## 归档

### archive/2026-04-21-handoff/ - 交接快照, 架构与全部数字均已过时

- [archive/2026-04-21-handoff/HANDOFF.md](archive/2026-04-21-handoff/HANDOFF.md) - 2026-04-21 交接: 架构与待完成任务
- [archive/2026-04-21-handoff/SYSTEM.md](archive/2026-04-21-handoff/SYSTEM.md) - 2026-04-21 系统自述 (含已剪掉的宣传口径)
- [archive/2026-04-21-handoff/PRD_DEEPGRAPH_HANDOFF_2026-04-21.md](archive/2026-04-21-handoff/PRD_DEEPGRAPH_HANDOFF_2026-04-21.md) - 2026-04-21 接手 PRD (代码路径为旧仓库布局)

### integration/archive/ - 2026-07/08 integration 会话过程记录

- [integration/archive/SESSION_STATUS.md](integration/archive/SESSION_STATUS.md) - 2026-07-31 会话状态快照
- [integration/archive/COMMIT_PLAN.md](integration/archive/COMMIT_PLAN.md) - 本地提交记录与未来拆分指引
- [integration/archive/BASELINE.md](integration/archive/BASELINE.md) - 2026-07-30 集成基线与安全边界
- [integration/archive/CI_VALIDATION.md](integration/archive/CI_VALIDATION.md) - 隔离 CI 验证矩阵 (2026-07-31 终版记录)
- [integration/archive/DURABLE_QUEUE_VALIDATION.md](integration/archive/DURABLE_QUEUE_VALIDATION.md) - 持久队列验证步骤 (未在本机执行)
- [integration/archive/EVALUATOR_ISOLATION.md](integration/archive/EVALUATOR_ISOLATION.md) - evaluator bubblewrap 隔离的 CI 程序
- [integration/archive/ACCEPTANCE_EVIDENCE.md](integration/archive/ACCEPTANCE_EVIDENCE.md) - CPU + SSH A100 范围的 master 验收证据
- [integration/archive/UNVERIFIED.md](integration/archive/UNVERIFIED.md) - 该会话刻意不做的主张清单
- [integration/archive/IMPLEMENTATION_CHECKLIST.md](integration/archive/IMPLEMENTATION_CHECKLIST.md) - 受控 B 集成细粒度检查单 (2026-08-02)
- [integration/archive/PORTING_LEDGER.md](integration/archive/PORTING_LEDGER.md) - 语义移植台账
- [integration/archive/LLM_CALLER_INVENTORY.md](integration/archive/LLM_CALLER_INVENTORY.md) - LLM 调用方清单与 grant 边界 (2026-07-31)
- [integration/archive/LEGACY_TEST_CLASSIFICATION.md](integration/archive/LEGACY_TEST_CLASSIFICATION.md) - 30 个 legacy 测试失败的逐条分类
