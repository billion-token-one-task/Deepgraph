# DeepGraph 接手 PRD / Handoff

日期: 2026-04-21

作者: Codex

适用对象:
- 接手 DeepGraph 的后续工程师
- 负责运维 / 远程 GPU / LLM 网关的支援工程师
- 需要判断该项目是否继续投入的负责人

## 1. 背景

DeepGraph 当前已经打通了一个可运行的研究闭环雏形:

`论文摄取 -> 结构化抽取 -> 知识图谱 -> 深度 insight 生成 -> 新颖性验证 -> SciForge 实验 -> 远程 GPU 执行 -> 论文 bundle 生成 -> Web 服务对外展示`

但它还没有达到“稳定、全自动、可长期无人值守”的工程状态。

目前系统更准确的定位是:
- 已经验证了端到端链路是可行的
- 已经产出 pilot 级实验结果和论文草稿
- 但链路里仍有若干 P0/P1 级可靠性和一致性问题
- 新工程师接手后，不应直接继续堆新能力，而应先做链路硬化

## 2. 当前状态快照

截至 2026-04-21，本地环境观察到的真实状态如下:

- 对外服务:
  - `https://deepgraph.sora2.today` 已可访问
  - `nginx` 已接好 HTTPS
  - DeepGraph Web 已由 `systemd` 常驻
- 数据库:
  - 生产运行时使用 PostgreSQL
  - 本地仍保留一份历史 SQLite 文件 `deepgraph.db`
- 当前生产库状态:
  - `deep_insights`: 15 条
  - `auto_research_jobs`: 15 条
  - `experiment_runs`: 14 条
  - `manuscript_runs`: 2 条
  - `submission_bundles`: 3 条
- 典型状态分布:
  - `auto_research_jobs`: `completed=9`, `blocked=4`, `running_gpu=1`, `researching=1`
  - `experiment_runs`: `completed=6`, `failed=6`, `bundle_ready=1`, `scaffolding=1`
- 已有论文产物:
  - 自动 PaperOrchestra bundle: `run_3`
  - 手工补装的 conference draft: `run_8`
- 最新重要现象:
  - `deep_insight_id=13` 已经出现 `confirmed` 实验结果
  - 但最新 GPU run 后续又因下游错误被覆盖为 `failed`
  - 说明“实验成功”与“整条链路成功”目前不是同一个状态

## 3. PRD 目标

本阶段目标不是扩功能，而是把 DeepGraph 修到“可交接、可稳定运行、可持续产出”的状态。

### 3.1 目标

1. 把生产链路修成单一真相、单一状态机。
2. 让 discovery / experiment / manuscript 三条子链路在 PostgreSQL 环境下稳定运行。
3. 保证一次成功实验不会被后续 bundle 或 LLM 故障回写成失败。
4. 让 manuscript bundle 和数据库状态一致，不再出现“磁盘有稿、库里没状态”。
5. 让新工程师能在一周内定位系统入口、运行状态、故障面和验收标准。

### 3.2 非目标

- 本阶段不追求新的研究方向或新 benchmark 扩展
- 不追求更强的论文质量优化
- 不优先做新的前端 UI 功能
- 不优先重写整套架构

## 4. 端到端链路分段

当前链路可拆为 6 段:

1. 摄取与抽取
2. Discovery 与 Deep Insight 生成
3. Novelty Verification / Full Research
4. SciForge 实验编排与远程 GPU 执行
5. Manuscript / PaperOrchestra 出稿
6. Web 服务、状态回写、对外可观测性

问题也主要集中在这 6 段之间的接口层，而不是单个模块的算法逻辑。

## 5. 问题清单

以下优先级按接手工程师的实际修复顺序排列。

### P0-1. LLM 网关不稳定，直接打断 discovery、verification、GPU run

现象:
- 当前生产日志仍在出现 `502 / 504 / connection refused / provider cooldown`
- 影响 discovery、taxonomy expansion、novelty verification、GPU hypothesis run
- 失败后系统会进入 cooldown 或直接把实验 run 标记失败

证据:
- `Deepgraph/logs/web-systemd.log`
- `Deepgraph/agents/llm_client.py`
- `Deepgraph/config.py`

当前风险:
- 任何依赖 LLM 的后台线程都可能随机失败
- 失败会污染状态机，而不是只影响单次调用
- Provider 路由目前偏“可用就上”，缺少明确的健康探针和降级策略

需要修正:
- 为 provider 增加健康检查、熔断、恢复策略
- 区分“上游暂时不可用”和“本任务逻辑失败”
- 引入更清晰的主/备/兜底 provider 策略
- 所有下游状态写入前要判断错误归因，不能把上游 502/504 直接沉成业务失败

验收标准:
- 单个 provider 失效时，任务自动切到可用 provider 或进入 retry queue
- 502/504 不会直接把已完成的 experiment verdict 覆盖成失败
- provider 冷却状态可在 dashboard 或 API 中清晰看到

### P0-2. PostgreSQL 兼容性不完整，导致 pipeline 事务中断

现象:
- 当前日志中出现:
  - `invalid input syntax for type double precision: ""`
  - `current transaction is aborted`
- 根因之一是 SQL 把 PostgreSQL 数值列拿去和空字符串比较

证据:
- `Deepgraph/agents/insight_agent.py`
- `Deepgraph/orchestrator/pipeline.py`
- `Deepgraph/logs/web-systemd.log`

当前风险:
- 一条错误 SQL 会把整个 PG 事务打进 aborted 状态
- 后续 pipeline 查询全部失败
- discovery refresh 失败后，处理线程可能整体崩掉

需要修正:
- 清理所有 SQLite 风格、弱类型风格 SQL
- 重点检查:
  - `metric_value != ''`
  - 数值列与字符串比较
  - `GROUP_CONCAT` / SQLite 方言兼容写法
- 数据访问层增加事务失败后的回滚保护

验收标准:
- 在 PostgreSQL 环境下跑 `api/start`，连续处理 100 篇 paper 不出现 `InFailedSqlTransaction`
- 任何 SQL 异常都会显式 rollback，不污染后续查询

### P0-3. GPU scheduler 把“实验成功”覆盖成“整条链路失败”

现象:
- `experiment_run` 可能已经得到 `hypothesis_verdict='confirmed'`
- 但只要后面的 manuscript bundle 或 LLM 写作失败，整个 `experiment_runs.status` 会被更新成 `failed`
- 这会制造“结论成功，但 run 状态失败”的自相矛盾状态

证据:
- `Deepgraph/orchestrator/gpu_scheduler.py`
- 当前库里 `run_id=14` 曾出现 `hypothesis_verdict='confirmed'` 但最终 `status='failed'`

当前风险:
- 业务结果与系统状态脱钩
- 后续调度器、dashboard、人工判断都会读错
- 会丢失真正成功的实验记录

需要修正:
- 把链路状态拆开:
  - experiment execution status
  - knowledge loop status
  - manuscript generation status
  - bundle publication status
- manuscript 失败只能影响 manuscript 子状态，不能反写 experiment 失败

验收标准:
- 一次成功实验即使 manuscript 失败，`experiment_runs.status` 仍保持 `completed`
- manuscript 失败只记录到 bundle/manuscript 相关字段

### P0-4. 生产环境存在双数据库真相，接手极易误判

现象:
- 服务运行时用 PostgreSQL
- 仓库目录下还保留历史 SQLite `deepgraph.db`
- 该 SQLite 结构明显更旧，字段缺失很多

证据:
- `Deepgraph/.env`
- `Deepgraph/db/database.py`
- `/home/billion-token/Deepgraph/deepgraph.db`

当前风险:
- 新工程师很容易对着错库排障
- 本地脚本和生产服务可能读写不同 backend
- 会导致“代码看起来不对，实际上只是查错库”

需要修正:
- 明确 PostgreSQL 为唯一生产真相
- 为 SQLite 标记 legacy / local-only
- 文档、脚本、README 全部更新为 PG-first
- 增加一个 `doctor` 或 `/api/system/info` 输出当前真实 backend

验收标准:
- 接手工程师在 5 分钟内能确认当前服务连接的是哪一套库
- 所有运行文档默认指向 PostgreSQL

### P1-1. auto_research / pipeline_events 存在状态漂移

现象:
- `pipeline_event_consumers` 进度存在滞后
- `auto_research_jobs` 可能停在 `running_gpu`、`researching`
- 即使已有 `experiment_run_completed` 事件，也未必完成状态刷新

证据:
- `Deepgraph/orchestrator/auto_research.py`
- `Deepgraph/orchestrator/gpu_scheduler.py`
- `Deepgraph/db/database.py`

当前风险:
- dashboard 看到的状态不是最新状态
- 同一 insight 可能重复调度或卡在假运行中

需要修正:
- 明确事件驱动与轮询驱动的优先关系
- 为 consumer 增加 lag 监控
- 为 stale job 增加自动纠偏逻辑
- 区分 queue 中 job 的“实际执行中”和“状态未刷新”

验收标准:
- `gpu_job_completed` / `experiment_run_completed` 发出后，相关 job 状态在 30 秒内收敛
- stale job 自动被纠偏或标记异常

### P1-2. Manuscript 流程有“自动 bundle”和“手工 bundle”两套路径，状态不一致

现象:
- DB 中有自动 PaperOrchestra 产物，也有手工补装的 `stale` bundle
- `deep_insights.submission_status` 与磁盘上的实际 bundle 可能不一致
- 最新 confirmed run 并没有自动得到新的 bundle 绑定

证据:
- `Deepgraph/agents/manuscript_pipeline.py`
- `Deepgraph/agents/paper_orchestra_pipeline.py`
- `deepgraph_manuscripts/run_8/*`

当前风险:
- 不知道“哪个 bundle 才是当前真稿”
- run 与 paper 没有严格一一对应关系
- downstream 负责人无法判断哪篇稿可继续推进

需要修正:
- 明确 manuscript 只允许一条正式生产路径
- 手工 bundle 必须以导入工具回填数据库并标注来源
- `experiment_run -> manuscript_run -> submission_bundle` 必须可追踪

验收标准:
- 每个正式 bundle 都能追溯到唯一 run_id
- `submission_status` 与 `submission_bundles` 完全一致

### P1-3. Research bridge 仍依赖 legacy `insights` 表，和 v2 `deep_insights` 脱节

现象:
- 部分桥接逻辑仍查 `insights`
- 当前系统主链路已经以 `deep_insights` 为核心

证据:
- `Deepgraph/agents/research_bridge.py`

当前风险:
- 老链路和新链路混用
- 后续工程师会误以为所有研究 proposal 都来自统一数据模型

需要修正:
- 明确 legacy `insights` 是否保留
- 若保留，则标注仅兼容旧链路
- 若不保留，则把 bridge 统一迁到 `deep_insights`

验收标准:
- 新链路全部围绕 `deep_insights`
- legacy API 明确标记 deprecated

### P1-4. Tier-1 insight 质量门禁还不够硬，仍有不完整记录进入系统

现象:
- 当前库里仍有 Tier-1 记录因为缺失 `Field A / Field B / formal structure / transformation` 被 verification 阻塞

证据:
- `Deepgraph/contracts/pipeline.py`
- `Deepgraph/tests/test_vnext_discovery.py`
- 当前 `auto_research_jobs` 中 blocked 条目

当前风险:
- 低质量 insight 会污染验证队列
- 新工程师会误判为 verification 有问题，其实是输入合同未满足

需要修正:
- 所有 Tier-1 写入路径统一走 contract 校验
- 给历史坏数据做清洗或标记 `invalid`
- 前端和 API 明确提示“不可验证的原因”

验收标准:
- 新生成的 Tier-1 insight 不再出现缺关键字段后才进入 verification 的情况

### P1-5. `/api/events` 存在 datetime JSON 序列化问题，影响前端实时事件流

现象:
- 当前日志里出现 `TypeError: Object of type datetime is not JSON serializable`
- SSE 端点使用 `json.dumps(e)`，没有 `default=str`

证据:
- `Deepgraph/web/app.py`
- `Deepgraph/logs/web-systemd.log`

当前风险:
- 实时事件流中断
- 前端状态刷新看起来“随机卡住”

需要修正:
- SSE 输出统一可序列化
- 所有 event payload 统一 contract 化或预序列化

验收标准:
- `/api/events` 连续运行 24 小时不出现 datetime 序列化崩溃

### P1-6. 生产服务仍使用 Flask 开发服务器

现象:
- `main.py` 直接 `app.run(...)`
- 当前是 `systemd + nginx -> Flask dev server`

证据:
- `Deepgraph/main.py`
- `Deepgraph/deploy/deepgraph-web.service`

当前风险:
- 并发、稳定性、超时处理都不适合长期生产
- 后台线程和 Web 线程耦合度过高

需要修正:
- 切换到 gunicorn / uvicorn + worker model
- 背景任务与 Web 进程解耦
- 增加 healthcheck 和 restart policy

验收标准:
- 生产 Web 不再依赖 Flask dev server

### P2-1. 环境依赖未被显式声明，导致首轮 run 失败

现象:
- 历史失败里出现 `[Errno 2] No such file or directory: 'git'`
- 说明运行时环境前置检查不完整

证据:
- `experiment_runs.error_message`
- `Deepgraph/orchestrator/gpu_scheduler.py`

需要修正:
- 启动前做环境自检
- 对 git、ssh、EvoScientist、PaperBanana、GROBID、Postgres 等做统一 doctor 报告

### P2-2. 测试覆盖仍缺关键生产场景

现状:
- 现有 `tests.test_vnext_discovery`、`tests.test_vnext_manuscript` 可通过
- 但还缺以下回归保护:
  - PostgreSQL 数值列空字符串比较
  - 事务 abort 后 rollback 恢复
  - SSE datetime 序列化
  - bundle 失败不覆盖成功 experiment
  - event consumer lag 与 stale job 收敛

## 6. 建议的修复顺序

### 阶段 A: 先止血

目标:
- 不再让成功实验被覆盖为失败
- 不再让 pipeline 因 PG 事务错误整体崩掉
- 不再让 provider 抖动把状态机打乱

任务:
- 修 PostgreSQL 不兼容 SQL
- 给数据库异常补 rollback
- 拆分 experiment status 与 manuscript status
- 给 LLM provider 增加稳定性分层与 retry 逻辑

### 阶段 B: 状态机收敛

目标:
- 所有 run / job / bundle 都能在库里形成一致状态

任务:
- 清理 stale manual bundle
- 建立 run -> manuscript -> bundle 一致映射
- 修 auto_research / gpu_scheduler 事件收敛
- 增加 stale job 自动修复

### 阶段 C: 统一数据模型与可观测性

目标:
- 新工程师只需要认一个主链路

任务:
- 统一到 `deep_insights`
- 标记 legacy `insights` / SQLite 为兼容层
- 增加 system doctor / backend info / provider health API

### 阶段 D: 生产化

目标:
- 从“能跑”升级到“能长期跑”

任务:
- 切换生产 WSGI/ASGI
- 背景 worker 解耦
- 加 healthcheck、metrics、structured logging

## 7. 交接建议

建议至少安排 2-3 位工程师协作:

### 角色 A: 后端主程

负责:
- PostgreSQL 兼容
- 状态机与事件流修复
- legacy -> v2 数据模型统一

### 角色 B: 基础设施 / 平台工程师

负责:
- LLM provider 健康检查
- 远程 GPU worker 稳定性
- Web 生产部署、systemd、nginx、HTTPS、健康探针

### 角色 C: 研究工程师

负责:
- Manuscript 自动化链路
- Verification / Research prompt 和输入合同
- 实验成功后的 bundle 生产一致性

## 8. 建议的一周接手计划

### Day 1

- 确认 PostgreSQL 为唯一生产库
- 跑 system doctor
- 检查 provider 可用性
- 拉齐当前 stuck jobs / failed runs / stale bundles

### Day 2-3

- 修 SQL 与事务 rollback
- 修 experiment status 被 downstream 覆盖的问题
- 修 SSE datetime 序列化

### Day 4-5

- 修 manuscript 状态机
- 统一 deep_insights / research bridge
- 补关键回归测试

### Day 6-7

- 切换生产 Web 运行方式
- 完成 handoff 演练
- 产出新的 runbook / oncall 文档

## 9. 验收标准

项目可认为“完成本轮交接修复”的标准:

1. `api/start` 在 PostgreSQL 环境下可稳定处理 100 篇 paper，不出现事务中断。
2. 一个 Tier-2 insight 从验证到实验到 manuscript 能跑完，且状态一致。
3. 成功实验不会因 bundle 或 LLM 下游错误被回写成失败。
4. `/api/events`、`/api/auto_research/jobs`、`/api/manuscripts` 三类状态接口持续稳定。
5. 最新 bundle 会自动绑定到最新成功 run，而不是漂浮在磁盘目录里。
6. 新工程师可只依赖本 PRD 和仓库文档完成本地接手。

## 10. 相关入口文件

优先阅读:

- `Deepgraph/README.md`
- `Deepgraph/SYSTEM.md`
- `Deepgraph/HANDOFF.md`
- `Deepgraph/db/database.py`
- `Deepgraph/orchestrator/pipeline.py`
- `Deepgraph/orchestrator/auto_research.py`
- `Deepgraph/orchestrator/gpu_scheduler.py`
- `Deepgraph/agents/llm_client.py`
- `Deepgraph/agents/manuscript_pipeline.py`
- `Deepgraph/agents/paper_orchestra_pipeline.py`
- `Deepgraph/web/app.py`

## 11. 结论

DeepGraph 不是“失败项目”，也不是“已经完成的项目”。

它当前最真实的状态是:
- 路已经打通
- 价值已经验证
- 但还没有被工程化到可稳定交接

如果后续还有工程师支援，这个项目是可以继续推进的，而且现在最应该做的不是再扩能力，而是把现有链路修成一个稳定的产品化研究系统。
