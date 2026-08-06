# 反思与修复方案: 自治链条为什么没走通 (2026-08-06)

写于动手修复之前, 依据 2026-08-04 ~ 08-06 约 37 小时试运行的实测数据 + 本日上午的独立取证。
所有关键论断带出处 (file:line 或数据库查询), 均为本日重新核实, 不是转抄任务书。

## 0. 最重要的修正: 断链不是一处, 是四处

任务书把 R1 (idea 生成无触发器) 列为"第一因"。核实属实, 但**按验收标准衡量, 它只是四个
同级阻断中的第一个**。对 boot 路径 (main.py 启动的全部线程) 逐一 grep 核实:

| # | 链条环节 | 状态 | 证据 |
|---|---------|------|------|
| 1 | idea 生成 | 无自动触发器 | 唯一生产者 run_full_discovery 仅剩 CLI 脚本 run_bulk_deep_insights.py 一个入口; auto_research 是纯消费者 (agents/agenda_orchestrator.py 全文 40 行, 只读 deep_insights) |
| 2 | portfolio 裁决 | 无自动调用方 | decide_portfolio (meta_harness/portfolio.py:139) 只被 HTTP 操作者端点调用; auto_research.py 中 grep 'awaiting_portfolio_decision\|decide_portfolio' 零命中 |
| 3 | grant 发放 | 无自动调用方 | issue_resource_grant (meta_harness/portfolio.py:256) 同上, 只有操作者 HTTP 端点 |
| 4 | granted 执行 | 无自动调用方 | execute_granted_candidate (orchestrator/bounded_execution.py:332) 唯一调用方是 scripts/run_bounded_pilot.py (操作者 CLI); 该模块 docstring (bounded_execution.py:4) 自述 portfolio_granted "nothing in the codebase ever read that stage back" |

外加一个善后缺口: reconcile_expired_grants (meta_harness/repository.py:961) 无任何周期
调用方。实测后果: grant 7 于 08-04 16:38Z 过期, 至今 (~40 小时) status 仍为 active,
一直占着 agenda 7 的唯一并发名额。

**为什么 37 小时试运行只暴露了第 1 环**: agenda 10/11 连候选都没有, 链条根本没走到 2-4 环。
而 08-04 的 cycle 1 里, 2-4 环全部由操作者手工驱动 (OPERATORS.log 可查), 掩盖了它们
同样没有自动化的事实。auto_research 循环真正自动做的只有一件事: 从已有 deep_insights
挑一个, 写到 stage='awaiting_portfolio_decision', 然后没有下文。

推论: 只修 R1, 验收窗会死在 "candidates >= 1 但 grants = 0, runs = 0"。P1 必须把
2-4 环一起补上, 否则不用开验收窗。

## 1. Q1: 方案 A vs 方案 B, 以及第三条路

**结论: 用 B 的载体实现 A 的目标 -- 新建 ideation job 队列 (镜像 scoped_ingestion 形态),
外加一个小的"饥饿自动入队"策略。**

理由:

1. 方案 A (在 auto_research 循环里直接触发 discovery) 违反系统自己的模块不变量:
   agents/agenda_orchestrator.py:1-5 与 agents/agenda_selector.py:3-4 的 docstring 明文
   规定选择循环不得调用 LLM / 不得发 grant。这个不变量是上一轮"两注册表纪律"重构的
   刻意产物, 不应为省事打破。
2. 方案 B 有现成模板可抄: scoped_ingestion_jobs_v1 已实现全套 fail-closed 纪律 --
   操作者 token、幂等键 (UNIQUE agenda_id+idempotency_key)、enqueue 时与 claim 时双重
   grant 校验、SKIP LOCKED 租约、attempt 上限、单 agenda 单 running 约束、带 reason code
   的失败落库 (meta_harness/ingestion_queue.py:57-260)。ideation job 照抄这个形态,
   审阅面最小。
3. 纯 B (只有端点, 等操作者提交) 不满足"提方向就自动出结果"。所以补一个策略层:
   active 且有预算的 agenda, 当 (a) 无可选候选 (b) 无 running/queued ideation job
   (c) 未触及频率上限 (每 agenda 同时最多 1 个, 每日最多 N 个) 时, 由系统自动 enqueue。
   flag 单独控制 (DEEPGRAPH_AUTO_IDEATION_ENABLED), 关掉即回到纯 B。

grant 纪律怎么接: discovery 的 LLM 调用目前完全游离于 grant 机制之外 (run_full_discovery
路径里没有任何 require_active_scope) -- 这本身就是要修的缺陷。接法: 复用 cycle 1 已验证的
pre-idea 身份模式 (agents/paper_idea_agent._proposal_candidate_and_grant, 不花 LLM 造一个
proposal-pending 候选作为 grant 的挂靠点), ideation grant 限 stage='ideation',
token_cap <= 5000, TTL 2h, 从 agenda 自己的 token_budget 走 agenda_resource_ledger 预留,
LLM 调用经 meta_harness/scoped_llm.require_active_scope 计账。

内存护栏 (7GB 主机, exit 137 已发生一次): ideation job **跳过 harvest_signals 阶段**。
实证依据: harvest_entity_overlap (agents/signal_harvester.py:460-468) 无 LIMIT 拉全量
paper_entity_mentions join, harvest_pattern_matches (:568-573) 无 LIMIT 拉全表 patterns
进 Python -- 这就是播种进程被 OOM 杀的直接嫌疑。而造 idea 真正需要的只是
refresh_research_problems (有界 limit) + run_tier2_discovery (有界), 两者都是 LLM 密集
而非内存密集。harvest 阶段给全局信号表用, 单独作为带 LIMIT 的维护任务放 P4。
另注意 run_full_discovery 会二次调用 refresh_research_problems (discovery_scheduler.py:139
在 tier2 内部又刷一遍), ideation job 直接调 tier2 入口, 避免双倍花费。

**第三条更小的路 (systemd timer 定时跑 run_bulk_deep_insights CLI): 拒绝。**
无 grant 纪律、无 token 台账、OOM 无界、且本质是把"人工代劳"制度化为 cron 代劳 --
系统仍然没有"自己决定去 ideate"的结构。仅保留为验收失败后的对照应急手段。

## 2. Q2: web 挂死根因 (py-spy 实锤) 与看门狗改法

**根因已闭合, 不是猜测。** py-spy dump (证据存
/home/ec2-user/deepgraph-reports/evidence-pyspy-web-hang-20260806T0815Z.txt):
8 个 waitress worker 线程 (waitress-0..7) **全部**停在同一行 -- web/app.py:1588,
即 GET /api/events (SSE 端点) 的 generate 生成器。

机制链条:

1. /api/events 是 `while True` 无限生成器, 每轮 get_events 无新事件就 time.sleep(2) (web/app.py:1576-1588);
2. 管线死寂 -> pipeline_events 自 06-21 无新行 -> 生成器**永远不 yield 任何字节**;
3. 不写 socket 就永远探测不到客户端 (nginx) 已断开 -> 线程永久泄漏;
4. 外部三个 IP 的仪表盘反复连 SSE, 数小时内把固定 8 个线程逐个吃光;
5. 队列积压 (Task queue depth 90) -> waitress 拒绝新连接 -> 一切 HTTP 死亡,
   而后台调度线程 (gpu-scheduler / auto-research) 照常存活。

这解释了全部观测: 为什么每次重启后数小时复发 (03:48Z 重启, 06:29 日志停止);
为什么 systemd 显示 active; 为什么 selfheal 判 unknown; 为什么 GPU 心跳还在刷。

修复 (两行级补丁): 空轮询时 yield SSE 注释行 ": keepalive\n\n" (强制触发 socket 写,
死连接立即抛错回收线程), 外加连接最大寿命 (如 10 分钟后正常结束, EventSource 会自动重连)。
可选缓解: waitress threads 8->16。

**与 08-04 记录的 "connection closed/lost" 是两个不同缺陷**: 那个是 db.get_conn 线程本地
PG 连接无存活探测/无重连 (OPERATORS.log 2026-08-04T12:24Z 有分析), 属独立卫生项, 列 P4。
同类卫生项: auto-research 线程常驻 idle-in-transaction (实测一个事务从开机挂到现在 4h+,
每 30s 在事务里查询但从不 commit), 只读也该及时 rollback。

**看门狗要不要改: 要, 但窄改。** "TCP 能连上 + HTTP 连续 N 次超时" 不是 unknown, 是可判定
的 http_stalled 状态。改法: selfheal 探测加 curl 超时判据, 命中 http_stalled 且满足
(a) 无 running 的 experiment_run/gpu_job (沿用 R8 重启时机纪律) (b) 6 小时内未重启过
(c) 重启前自动落一份 py-spy dump 作证据 -- 三条全满足才 restart。对真正的 unknown
(如 DB 本身连不上) 保持 fail-safe 不动作。SSE 修复后此路径应长期不触发, 属纵深防御。

## 3. Q3: 上一 session 哪些操作是"代劳", 验收窗如何避免

代劳清单 (违背"只在输入端提方向"的程度从高到低):

1. **手工跑 run_full_discovery 为 agenda 10/11 造候选** (08-04 20:40 决定, 结果被 OOM
   杀掉, 未遂) -- 直接替系统做第 1 环。
2. **cycle 1 全链手工驱动**: frontier authority -> packet -> topic gate -> selection ->
   portfolio decide -> grant -> bounded pilot, 全部操作者出手 (OPERATORS.log 12:26-14:43)。
   当时定位就是"监督驱动", 无可厚非, 但它掩盖了第 2-4 环无自动化的事实。
3. 手工 settle 台账 (reservation 9) -- 系统缺 forge 建 run 前失败的合法结算路径, 属替系统
   善后; 该路径缺口列入 P1 第 4 项。
4. 手工 upsert 32036 的 worker 行 -- 基础设施供给, 不算业务代劳, 但心跳维护本应归系统。

验收窗纪律 (比任务书第 1 节收紧一档):

- 允许: 重启服务; 开关 .env; 经 /api/meta-harness/v1 一次性提交方向 / ingestion job /
  (若采用纯 B 形态) 每 agenda 一次 ideation bootstrap job。
- 禁止: 人工触发任何 frontier/gate/portfolio/grant/selection/execution/settle;
  禁止人工写业务表 (原有约束); **链条停在哪一环, 就如实记为该环失败**, 不许人肉推过去。
- 观察进程只读 (scripts/observe_agenda_run.py 本来就是只读采样)。

## 4. Q4: 验收标准的漏洞与补强

1. **"metric_value 非空"可被无意义数值满足** (例如 hello-world 吞吐)。补强: outcome 必须
   过既有 verdict integrity gate (真 metric + 非零 baseline + 真 effect); 阴性/证伪结果
   照算通过, "零移动"与"无 baseline 的孤数"都算失败。
2. **模型代际违约漏洞 (会必然发生)**: forge 的 _default_real_model_targets
   (agents/experiment_forge.py:1023-1054) 硬编码 TinyLlama/Qwen2.5/ViT-B, 加上
   config.py:581 默认 Qwen2.5-7B, **每一个都在 agenda v2 的明文禁用清单上** (两个 v2 json
   的 model_generation_currency 条款逐字核对过, 且点名了 experiment_forge.py)。Qwen3 在
   Python 源码中零出现。不修 forge, 链条就算跑通也是违约运行, 不应计为验收通过。
   因此 forge 模型修正从 P4 提升为 **P1 关键路径**。
3. **新颖性闸漏洞** (partially_exists/verifying 不拦, agents/topic_gate.py:96
   _OBSOLETE_STATUSES 只有 exists/duplicate/obsolete/solved): 操作者已决定保留为观察点,
   尊重该决定不改行为; 但验收报告必须披露每个过闸候选的 gate reason codes 与
   topic_gate_json 预登记, 让"平庸 idea 走完链条"至少可见、可审计。
4. **语料 84 天陈旧**: idea 可能建立在过时文献上。不把补语料设为本次验收前置 (那是
   ingestion 链自己的事), 但报告必须如实披露语料时间上限。
5. **预算检查单薄**: 全循环唯一预算判据是 research_agendas.token_budget > 0
   (agents/agenda_repository.py:116-127)。自动化 2-4 环后, 每环都要带显式 cap
   (见 P1 设计), 否则自动链条会把 500k 预算烧在无价值重试上。

## 5. 修复方案 (P0-P4)

### P0-a: 救活 web (立即, 用户执行重启)

取证已完成 (py-spy dump 已存档)。当前无 running run/gpu_job, 符合 R8 重启时机。
请用户执行: `! sudo systemctl restart deepgraph-web.service`。
SSE keepalive 补丁并入 P1 同一个部署窗口, 避免两次窗口。

### P0-b: 32036 半机队 (已定性, 不阻塞验收)

32036 端口 TCP 可达, 远端环境 08-04 已部署并实测 rc=0。心跳停摆根因是**结构性的**:
register_default_workers (orchestrator/gpu_scheduler.py:605-624) 只注册 .env 单端点
(现为 32035) 的 4 张卡, 调度器没有多端点概念; 32036 的行是 08-04 手工 upsert 的,
之后无人续心跳。验收窗用 32035 的 4x A100 足够 (模型 <= 8B)。多端点支持
(DEEPGRAPH_GPU_REMOTE_SSH_ENDPOINTS=host:port,host:port 列表, register 循环即可,
执行路径本就按 worker 行 metadata 里的 ssh_port 连) 是小补丁, 作 P1 可选项, 不强求。

### P1: 补全四环 (核心交付, 全部 flag 独立控制, 全部 fail-closed)

1. **ideation job**: 新端点 POST /api/meta-harness/v1/ideation/jobs + 新表
   ideation_jobs_v1 (照抄 scoped_ingestion_jobs_v1 全套纪律) + worker 线程
   (flag: DEEPGRAPH_IDEATION_WORKER_ENABLED)。job 执行 = pre-idea 身份 + ideation grant
   计账 + 有界 refresh_research_problems + 有界 tier2 discovery, 跳过 harvest。
   饥饿自动入队策略 (flag: DEEPGRAPH_AUTO_IDEATION_ENABLED): 条件与频率上限见第 1 节。
2. **auto portfolio**: awaiting_portfolio_decision 的消费者, 按 cycle 1 已走通并留档的
   序列自动执行: frontier authority -> packet -> gate -> decide_portfolio ->
   issue_resource_grant, 每步沿用既有 reviewed 模块与既有 caps (pilot 5000)。
   flag: DEEPGRAPH_AUTO_PORTFOLIO_ENABLED。拒绝必须带 reason code 落库。
3. **auto execution**: portfolio_granted 的消费者 -> execute_granted_candidate,
   并发受 agenda max_concurrency 与 grant backend_allowlist 约束。
   flag: DEEPGRAPH_AUTO_EXECUTION_ENABLED。两个随行修复:
   (a) forge 模型代际: 目标模型选择读 agenda 约束 + env 钉 Qwen3
   (DEEPGRAPH_REAL_LLM_MODEL=Qwen/Qwen3-1.7B 等), 命中 agenda reject 清单的直接拒绝并
   落 reason code; (b) grant 分角色子预算: job 98 的饿死 (benchmark_design 吃 3063/5000,
   code_scout 饿死) 是确定性复发点, 给 bounded_execution 的计量按角色留底额。
4. **善后循环**: reconcile_expired_grants 并入调度循环周期执行 (幂等, 只把过期 grant
   翻状态); 补 forge 建 run 前失败的合法结算路径 (cycle 1 手工 settle 的那条路)。

改动全部做在 /home/ec2-user/Deepgraph-meta-harness-v1, 部署遵循 deepgraph-safe-update-plan
+ ops-rules (diff-gate, tar/pg_dump 锚点, OPERATORS.log 前后各一行, 单次重启窗口)。
schema 只增不改 (additive-only)。

### P2: 观察面耐重启化

observe_agenda_run.py (agenda 10/11) + render_observation_report.py 装 systemd timer
(或 cron @reboot), 轨迹 JSONL 与 HTML 全落 /home/ec2-user/deepgraph-reports/, 更新
latest.html 软链。selfheal 的 http_stalled 窄改 (第 2 节) 一并进这个窗口。

### P3: 验收窗 (6-12 小时)

按第 3 节纪律执行, 第 4 节补强后的标准判定。产出报告 + 诚实结论: 链条每环到没到,
数值是不是真的, 模型是不是合规代际, gate 放过了什么。

### P4 (有余力再做, 按影响排序)

fair 轮转 (list_active 按序首中即返, agenda 10 会饿死 11; MAX_ACTIVE=1 加剧);
db.get_conn 重连 + 只读事务及时 rollback; harvest_signals 加 LIMIT/分批;
main.py 失败策略统一 (paper worker 可降级但 scoped ingestion/gpu scheduler 直接 raise
拖死启动) + AUTO_PIPELINE_ENABLED 死旗清理 (paper_worker.start 无条件返回 disabled,
该 flag 现在是空操作); 32036 多端点 (若 P1 未带); LLM json-literal 崩溃 (null/true
写进 Python, 占 crash 24%) 的 forge 侧防护。

## 6. 花费与风险

- P1 范围比任务书预估大 (1 环 -> 4 环 + 2 个随行修复), 但每环都是"薄 worker 调用已有
  reviewed 模块", 不新造科学逻辑; 风险集中在 auto portfolio 的序列编排, 以 cycle 1 的
  OPERATORS.log 记录为唯一蓝本。
- 验收窗 token 预估 (系统自花, 出自 agenda 各自 500k 预算): 每 agenda ideation <= 5000
  + pilot <= 5000, 两 agenda 合计 <= 20000, 在"累计 25000 无新批准"纪律内; 超出即停等批。
- 每个变更步骤前后写 OPERATORS.log; .env 改动先 tar 备份; 不 push (owner 负责)。
