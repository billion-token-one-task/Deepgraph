# Deepgraph 升级两版拆分: V1 先跑起来, V2 升级 + 精简 (2026-08-06)

前提共识 (用户 2026-08-06 决定): GPU 机队在持续烧钱, 第一优先级是**让链条立刻出成果**,
哪怕保留大量硬编码; 自治纯度验收与架构精简放到 V2。Vercel 暂不考虑。
背景诊断见 docs/reflection-20260806-autonomy-repair.md (断链四处 + SSE 挂死实锤)。

## V1: 小修小补, 目标 72 小时内 GPU 出第一批真实验结果

原则: 不新建表, 不新建 API, 不动核心模块逻辑; 只加 "补丁 + 脚本 + timer + env"。
胶水全部调用既有 reviewed 模块入口, 不写裸 SQL 改业务表。

按依赖排序:

1. **SSE keepalive 补丁** (web/app.py /api/events, ~3 行): 空轮询 yield ": keepalive"
   注释行 + 连接最大寿命 (10 分钟)。不修这个, web 活不过几小时, 后面全免谈。
   这是 V1 里唯一必须动核心文件的改动。
2. **env 两处, 零代码**:
   - DEEPGRAPH_REAL_LLM_MODEL=Qwen/Qwen3-1.7B (config.py:581 现成的 env 钩子,
     GPU 主实验模型立即换到 agenda v2 允许的代际; 换前先对 hf-mirror 做一次存在性
     确认)。CPU 分支的 TinyLlama/Qwen2.5-0.5B 硬编码 V1 容忍不动。
   - .env 改动前照例 tar 备份。
3. **有界播种脚本** scripts/seed_ideation.py: 直接调 refresh_research_problems(小 limit)
   + tier2 discovery(有界), **跳过 harvest_signals** (OOM 元凶, 上次 exit 137 的直接嫌疑),
   以 systemd-run --scope -p MemoryMax=3G 运行, agenda 10/11 各跑一次。
   这是有意识的"操作者代劳", V1 接受, V2 的 ideation worker 取代它。
4. **链条推进器** scripts/auto_advance.py + systemd timer (每 10 分钟):
   每轮顺序执行, 全走既有模块:
   a. MetaHarnessRepository.reconcile_expired_grants() -- 顺手清掉 grant 7 僵尸
      (过期 40h 仍 active, 占 agenda 7 并发名额);
   b. 对 stage=awaiting_portfolio_decision 的 job, 按 cycle-1 在 OPERATORS.log 留档的
      序列推进: frontier authority -> packet -> topic gate -> decide_portfolio ->
      issue_resource_grant (pilot cap 5000, backend 含 ssh_gpu, TTL 2h);
   c. 对 portfolio_granted 的 job, 调 execute_granted_candidate (bounded_execution);
   d. 拒绝/失败原样落 reason code, 不重试超过模块自身的 attempt 纪律;
   e. 每轮 token 支出汇总打到日志, 累计超 25000 自动停摆等新批准。
5. **多端点 GPU 注册** (~20 行, 强烈建议进 V1): DEEPGRAPH_GPU_REMOTE_SSH_ENDPOINTS=
   host:port,host:port 列表, register_default_workers 循环注册。8 张 A100 全部可用,
   直接回应 "GPU 在烧钱"。执行路径本就按 worker 行 metadata 的 ssh_port 连, 只缺注册环。
   (若想再省, 也可以先只用 32035 的 4 张, 此项挪 V2 -- 默认建议做。)
6. **观察面落持久盘** (原 P2 原样搬入): observe_agenda_run.py + render 报告挂 timer,
   JSONL/HTML 落 <REPORTS_DIR>/, 更新 latest.html。

V1 明确容忍的债 (V2 偿还): CPU 分支硬编码模型; 新颖性闸漏 (partially_exists 不拦);
语料 84 天陈旧; timer 轮询而非事件驱动; 无学习回路; auto_advance.py 本身是脚手架。

V1 的隐藏价值: 运行期间收集 coverage/调用观测, 让 V2 的删码决策基于
"哪些代码真的被执行过" 的实测, 而不是靠猜。

预期管理: 历史执行良率 9.75% (89.5% crash, 首因 LLM 把 JSON null/true 写进 Python);
git 身份修复后 run102 声称 40x 改进, 待本轮复证。所以 V1 的正确期望是:
**一批 run 上 GPU, 其中一部分出真 metric, 相当一部分 crash -- crash 与阴性结果
也按 verdict integrity gate 落 outcome, 同样是产出**。"零移动"才算 V1 失败。

花费纪律: ideation 每 agenda <= 5000, pilot 每 grant <= 5000, 观察窗累计 25000 上限,
触顶自动停。GPU 卡时走 agenda 各自 100 卡时预算。

## V2: 升级 + 精简同做, 六步走

### 冗余的实测证据 (2026-08-06, 不含 tests 与 venv)

主源码共 97993 行。确认或高度疑似的死重:

- plugins/examples/cggr 三个文件 6242 行示例代码随主仓库分发;
- paper_orchestra_pipeline.py 4715 行 (活性待 V1 coverage 判定);
- auto_research.py 4562 行, 其中真正在跑的自动逻辑 (run_scoped_cycle 链) 约 40 行;
- validation_loop.py 3819 行 (活性待判定);
- signal_harvester.py 2186 行, 现状无人能安全调用 (无 LIMIT 全表进内存, 7GB 必 OOM);
- web/app.py 2203 行里 24 个 410 墓碑路由 + before_request 全局拦截双保险 (留一处即可);
- pipeline.run_continuous ~300 行, 唯一调用方是一个路径全错的坏脚本
  (run_pipeline_forever.sh, ROOT 指向 /root/hf_models/...);
- paper_worker.py 整体死码 + DEEPGRAPH_AUTO_PIPELINE_ENABLED 死旗 (start 无条件返回
  disabled, 从不起线程); discovery_scheduler 5 处 blocked 存根; config 死旗
  (BACKFILL_GRAPH_ON_START 等 import 后从未引用);
- gpu_workers 表 9 行 retired 幻影; insights/paper_insights/deep_insights 等 legacy
  表并存 (schema additive-only, 表保留, 代码引用清理)。

粗估: 纯减法可砍 25-35% 源码, 不改任何行为。

### 步骤 (每步独立交付、独立回滚, 遵循 safe-update + ops-rules)

- **S0 安全网**: 给要保留的链路模块 (frontier/gate/portfolio/grants/ledger/
  ingestion queue/bounded_execution/gpu_scheduler) 补特征化测试; 从 V1 运行期收集
  coverage; pg_dump + tar 锚点; 死码清单与 LOC 基线入库。
- **S1 纯减法, 先删后改**: 按上表逐项删除, 行为不变, 测试全绿为闸。
  这一步与 V1 运行可并行 (删的都是 V1 不依赖的东西)。
- **S2 四环产品化**: ideation_jobs_v1 (表+端点+worker, 照抄 scoped_ingestion_jobs_v1
  全套纪律: 幂等键/租约/SKIP LOCKED/attempt/单 agenda 单 running/reason code) +
  饥饿自动入队; auto portfolio worker; auto execution worker; 维护循环
  (reconcile_expired_grants + forge 建 run 前失败的合法结算路径)。四环各配独立
  fail-closed flag。**此步完成后 auto_advance.py/seed_ideation.py 退役**, 并按原任务书
  标准 + 反思文档第 4 节的补强重跑一次"纯自治验收窗" (人工只许提方向)。
- **S3 bitter-lesson 改造**: forge 删除代码内模型知识 (模型来自 agenda json/env 配置,
  运行时对 hf-mirror 做存在性验证, 命中 agenda reject 即拒绝); grant 分角色子预算
  (job 98 饿死类问题的根治); 执行层用通用 run-观察报错-修-重跑 循环替代按错误类型
  打补丁; portfolio 决策用薄 LLM judge (proposer/evaluator 角色分离已有先例),
  不再加权重公式。设计验收项: 每环的效果应随预算/并行候选数单调提升, 调参不改码。
- **S4 学习回路** (长期杠杆最大): outcome_records 与 failure_clusters 回灌选题排序与
  forge 提示; llm_route_observations 真正写与读; discovery_track_record 启用。
  现状是 112 个 run 只留 1 条 outcome、路由观测全表 2 行 -- 系统用得越多并不会变好,
  这正是 bitter lesson 警告的人工知识平台期。
- **S5 运维硬化**: db.get_conn 存活探测+重连, 只读事务及时 rollback (auto-research
  线程常驻 idle-in-transaction 实测 4h+); selfheal 增加 http_stalled 判定
  (TCP 通 + HTTP 连续超时 -> 限速自愈重启, 重启前自动落 py-spy 栈); agenda 公平轮转
  (现为 list_active 首中即返, agenda 10 会饿死 11); nginx 静态资源直出、只反代 /api。

### S6 缩小手写面 (2026-08-25 补, 与 S3/S4 同源)

2026-08-25 一天内在生产上抓到七个缺陷, 没有一个是测量不诚实, 全部是接线:
worker 注册表选择只看 status 不看心跳, 九条历史对账记录把 scoped_ingestion
压成 halted 八天, 缓存版本号写死六个提交, --gpu-class 默认指向已停机数月的
A100 集群, data-verdict 属性写进 DOM 从无读者, agenda_signal_outcomes 在 150
个 outcome 之后仍为 0 行, _direction_evidence 返回空仍照常入库并锁死三个新
agenda。静态普查: 1,176 个字段中 18 个从未在生产代码里出现 (1.5%); 251 个模块
中最多 92 个从五个真实入口不可达 (上界, 含动态加载误判)。

结论: "系统会悄悄停在某处" 与 "流程是写死的" 是同一事实的两面。手写的耦合有
多少, 需要手写的检查就有多少, 而检查永远滞后于耦合。因此 S6 的目标不是补检查,
是让需要检查的东西变少。

- **S6.1 排序权重从后验学出来, 不再手写。** 2026-08-25 为了让图能给出线索, 手工
  加了 OPENING_RESEARCH_WEIGHT (open_question 100 / contradiction 90 /
  limitation 70 / benchmark_diversification 10)。这是 bitter lesson 明令禁止的
  那种人工知识, 当时是权宜: agenda_signal_outcomes 为空, 没有可学的东西。同日已
  接通写入路径 (outcome_finalizer 结算时回灌信号后验)。待该表积累到足以区分信号
  类型后, 权重必须改为从后验读取, 手写表降级为冷启动先验。验收: 删掉手写表后排序
  结果不劣化。
- **S6.2 接受规则由测量决定, 不由代码规定。** agenda 18
  (acceptance-rule-benchmark) 的产出——贪心 / held-out 闸 / 序贯规则在未被任何
  规则看过的分割上的泛化间隙——应当直接改写系统自己的接受阈值。当前的接受逻辑写在
  代码里且从未被自己的实验检验过; 一个搜索自身脚手架却不检验自身接受规则的系统,
  它的 "自进化" 只到候选层为止。
- **S6.3 阶段图变成数据。** 现在新增一个流水线阶段要改 auto_advance.py, 阶段之间
  的耦合因此是手写且不可枚举的, "这个阶段的产物有没有消费者" 只能靠人想起来去查。
  阶段与其输入输出契约落成数据后, 该问题可由测试自动回答, 与
  tests/test_job_state_consumers.py 已经守住 86 个 (status, stage) 组合同理——那
  86 个组合在 2026-08-25 一个都没出事, 这条路已被证明有效, 只是当初做完一个类别
  就停了。

即刻可做的一次性清理 (半天, 不属于 S6 主线): 那 18 个从未被读过的字段, 逐个判定
删除或接上消费者, 使该类别归零。

### 节奏

V1: 1-2 天工作量, 立即开工。S0-S1 与 V1 运行期并行。S2 完成即恢复"纯自治"验收。
S3-S5 按余力与实测痛点排期, 不设硬日期。
