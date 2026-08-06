# Roadmap 现状对账 (2026-08-06)

2026-08-04 的 README 重写 (627614e) 删掉了内嵌 changelog 里的 "3 周计划" 和
"18 周质量路线图" (均写于 2026-06-26)。本文记录它们的下落: 哪些已经建成 (带
源码出处), 哪些没有, 以及 18 周路线图为什么作废。旧文全文可用
`git show 627614e~1:README.md` 找回。

全部 file:line 均于 2026-08-06 对照源码核实。"已实现" 指机制存在于代码树并被
管线强制, 不代表它产出过任何通过科学闸门的论文 -- 截至 2026-08-04,
`scientific_decisions_total = 0`, 没有任何 run 到达 `manuscript_allowed`。

---

## 1. 现行质量闸门 (2026-08-06 核对)

这些闸门是旧计划 "cycle 1: 把合同/证据/审稿/PDF 检查变成强制" 目标的最终形态,
但以硬闸门而非迭代周期的方式落地:

| 闸门 | 作用 | 源码出处 |
|---|---|---|
| Topic gate | 纯函数筛选候选 (无 LLM, 无 DB); 无预测的候选直接给 `topic_gate_prediction_missing` blocker | `agents/topic_gate.py:381` (screen_candidate), `:72` (理由码), `:396-411` (缺预测路径) |
| Portfolio 决策 | promote/kill/park 三分, 带显式理由与 revisit 条件 | `meta_harness/portfolio.py:139-186` |
| 证据阶梯 | 每次只允许前进一步; 任何 blocker 直接 raise, 拒绝不落库 | `meta_harness/evidence_state.py:58` (advance), `:67-68` (单步), `:147-148` (raise) |
| decide_evidence | fail-closed 证据裁决: metric/baseline/全量 benchmark/raw artifacts/claim ledger/独立评估/p 值缺一即 blocker | `contracts/scientific_evidence.py:74-146` |
| Manuscript 闸门 | `manuscript_allowed` 需要 supported verdict + 审稿人批准 (签名, 会过期) + verdict hash | `meta_harness/evidence_state.py:137-146`, `meta_harness/reviewer_approval.py:147-171` |
| Benchmark contract hash | 全量 benchmark 及之后的每次转移都要求 64-hex sha256, 回溯审计时重算比对 | `meta_harness/evidence_state.py:48-56,90-92`, `meta_harness/retrospective_review.py:373-378` |
| Self-heal 策略 | 每 tick 只允许一个动作, 全部 restart 路径过同一冷却闸 | `orchestrator/selfheal_policy.py:206-208` (decide), `:176-204` (冷却) |
| Bounded pilot | 不开全局自治, 一次只执行一个拿到 grant 的候选; 该路径从不读自治开关, 有测试防其变成后门 | `orchestrator/bounded_execution.py:332`, `tests/test_bounded_execution.py:275-276` |

自治开关的精确表述: 部署实例以 `DEEPGRAPH_AUTO_RESEARCH_ENABLED=false`,
`DEEPGRAPH_AUTO_PIPELINE_ENABLED=0` 运行 (2026-08-03 核实,
`docs/runbooks/RECOVERY_2026-08-03.md`)。注意这是部署环境变量的事实, 不是代码
默认值: `config.py:684` 里 `AUTO_RESEARCH_ENABLED` 的代码默认是 True
(`AUTO_PIPELINE_ENABLED` 默认 False, `config.py:680`)。

## 2. 旧 3 周计划逐条状态

18 个条目: 10 条已实现, 7 条部分实现, 1 条未找到。

### Week 1: 问题意识与贡献框架

| # | 条目 | 状态 | 证据 |
|---|---|---|---|
| 1 | 稿件前 problem-first 闸门 | 部分 | `agents/paper_completeness.py:582-596` 要求 central_question/motivation/method_answer/result_claim 四要素, `agents/paper_orchestra_pipeline.py:3809-3819` 缺失即 raise; 但只查 "有没有", 不判 "空不空" (无 evidence-free gap / raw-result-as-contribution 检查) |
| 2 | 强制 paper_contract.json | 已实现 | `agents/paperorchestra/writing_standard.py:155-207` 生成 (含 banned_expressions), `agents/paper_orchestra_pipeline.py:4396` 落盘, `orchestrator/manuscript_watchdog.py:29` 列为必需文件; 但 "required experiments/figures" 不在合同字段里 |
| 3 | 标题/摘要 critic (打分制) | 部分 | `agents/paper_title_policy.py:159` 重写标题而非打分; 摘要有禁语与 oracle-metric 检查 (`agents/reference_corpus_audit.py:17-24`, `agents/paper_orchestra_pipeline.py:1751-1765`); 无任何数值评分 rubric |
| 4 | 强论文参照集 | 部分 | `agents/reference_corpus_audit.py:132-221` 对参照 PDF 语料做画像并比对; 但本仓库 checkout 里 `workspace/pdfs/` 为空, 语料在另一台 Windows 机 (见 `docs/top_venue_manuscript_chain.md`), 缺语料时降级为 medium issue |
| 5 | 空短语回归测试 | 未找到 | "significantly improves" 等三个短语在源码与测试中均无; 最接近的是 `orchestrator/manuscript_watchdog.py:85-109` 的 SOTA/first 正则和 cggr 插件的 `COMPLETED_EVIDENCE_FORBIDDEN_PATTERNS` (`plugins/examples/cggr/full_pipeline.py:309,353`), 都不含 "附近须有证据" 规则, 也无对应测试 |

### Week 2: 证据与实验

| # | 条目 | 状态 | 证据 |
|---|---|---|---|
| 6 | 标准实验矩阵 | 部分 | `agents/evidence_planner.py:127-181` (main_table/ablation/visualization), `agents/benchmark_protocol.py:459-487` (必需 artifact 清单), `agents/experiment_forge.py:1688-1693` (frontier/sweep/breakdown 要求); 唯独没有 negative/boundary-condition 实验槽位 |
| 7 | 机器可读实验 artifact | 已实现 | `agents/paper_completeness.py:387-414` (evidence_manifest_v1: 数据集/模型/baseline/metric/seed/硬件/日志/统计), failure log 实际产出在 cggr 插件 (`plugins/examples/cggr/experiment_runner.py:1839`) |
| 8 | 证据完备性审计员 (拦稿) | 已实现 | `agents/paper_completeness.py:640-684` (`paper_generation_allowed`), `:522-580` (blocker 清单), 两个稿件入口都强制 (`agents/paper_orchestra_pipeline.py:3809-3819`, `agents/manuscript_pipeline.py:613`); smoke run 拦截在 `agents/paper_orchestra_pipeline.py:3498-3499` |
| 9 | Baseline 公平性检查 | 部分 | `agents/benchmark_protocol.py:510-518` 声明公平性要求, `agents/paper_completeness.py:539-545` 查存在性 (>=2 baseline, prompt/decoding 披露); 但没有验证 baseline 真的在同预算下跑过, 也不产出 "stated differences" 记录 |
| 10 | 统计报告 (多 seed/CI/检验) | 已实现 | 要求侧: `agents/benchmark_protocol.py:505-509` (>=3 seed), `contracts/scientific_evidence.py:100-110` (p_value_missing/not_significant); 计算侧在 cggr 插件: `plugins/examples/cggr/experiment_runner.py:1315` (paired permutation), `:1330` (bootstrap CI, 2000 轮) |
| 11 | 三类必需图 | 已实现 | `agents/paper_orchestra_pipeline.py:1558-1588` (角色分类), `:1771-1775` (<3 图或角色覆盖不足即 high issue), `agents/paperorchestra/figure_standard.py:57-58,92-96` (概念图 block_if_missing), `agents/visual_layout_auditor.py:672-681` |
| 12 | 三份 JSON artifact | 已实现 | `agents/paper_orchestra_pipeline.py:4400-4412` 落盘 evidence_manifest/claim_evidence_matrix/reviewer_report, `contracts/pipeline.py:325-345` (ExperimentJudgement), `agents/experiment_forge.py:3338`, watchdog 强制 (`orchestrator/manuscript_watchdog.py:21-30`) |

### Week 3: 写作, 审稿与修订

| # | 条目 | 状态 | 证据 |
|---|---|---|---|
| 13 | 分节写作 critic | 已实现 | `agents/paper_orchestra_pipeline.py:1618-1755` (必需节, 节序, 引言/方法公式规则, Related Work 子节, Discussion 长度, 摘要); 无独立 Conclusion 专项, limitations 并入 Discussion |
| 14 | >=3 人格审稿模拟 | 部分 | 存在三个独立审稿阶段: `agents/paper_completeness.py:598-637` (12 题确定性 checklist), `agents/paperorchestra/refinement_loop.py:14-58` (area-chair 7 轴迭代), `agents/plain_manuscript_reviewer.py:128`; 但没有一个循环同时实例化 >=3 个命名人格, 也没有 clarity/positioning 人格 (`agents/tier2_review_refine.py` 的 Reviewer A/B 审的是 idea 不是稿件) |
| 15 | 结构化修订历史 | 已实现 | `agents/paper_orchestra_pipeline.py:3147-3159` (manuscript_revision_history.json), `:3054-3129` (每轮 before/after), `:3131-3141` (变差即回滚); diff 轴是严重度计数, 不是旧计划要的 claim/evidence/objection 分类 |
| 16 | 图表版式检查 | 已实现 | `agents/visual_layout_auditor.py:408-446` (表格溢出/字号), `:612-646` (caption 规范), `:126-156,557-575` (图的尺寸与位置), `agents/paper_orchestra_pipeline.py:1776-1778` (booktabs); 缺 "caption 必须说结论" 与坐标轴可读性检查 |
| 17 | PDF sanity 检查 | 已实现 | `agents/paper_orchestra_pipeline.py:1146,1685-1686` (编译必须过), `:1433-1500` (页数预算, 排除参考文献), `:2038` (未定义引用), `:1897-1898,3581-3593` (占位图), `agents/manuscript_length_auditor.py:167-190`; 唯独没有 Overfull box 扫描 |
| 18 | 投稿检查单 | 部分 | `agents/paper_orchestra_pipeline.py:4421-4440` 生成 submission_checklist.md, 但所有项硬编码 `[x]`, 不从审计结果计算; 无 ethics / artifact 可用性 / 复现声明项; 匿名化只在 prompt 层 (`agents/paperorchestra/venue_policy.py:106,321`) |

## 3. 残余清单 (旧计划中尚未实现的部分)

按上表提炼, 如果有人要继续这条线, 缺口是:

- 空短语 + 附近证据规则及其回归测试 (条目 5, 完全缺失)
- problem-first 闸门从存在性检查升级为内容质量判定 (条目 1)
- 打分制标题/摘要 critic (条目 3)
- 参照语料入库本仓库或指向可达路径 (条目 4)
- negative/boundary-condition 实验槽位 (条目 6)
- baseline 同预算实际核验 + stated-differences 记录 (条目 9)
- 单一稿件审稿循环内 >=3 命名人格, 含 clarity/positioning (条目 14)
- 修订 diff 按 claim/evidence/objection 分类 (条目 15)
- caption 结论性与坐标轴检查 (条目 16); Overfull box 扫描 (条目 17)
- submission checklist 由审计结果计算, 补 ethics/artifact/复现项 (条目 18)

这里只登记缺口, 不承诺日程。是否值得做, 取决于第 4 节的判断。

## 4. 18 周路线图为何作废

原路线图 (2026-06-26) 是 6 个三周周期: 质量闸门 -> 选题 -> 实验深度 -> 稿件
打磨 -> 审稿修订训练 -> 投稿彩排, 目标 "稳定产出更高质量论文"。作废理由:

1. **前提被两账制审计推翻。** 路线图建立在 "当前论文质量已达 workshop 水平"
   的自评上。按现行两账制, 那批论文没有一篇通过任何科学闸门: 截至 2026-08-04
   `scientific_decisions_total = 0`, 无 run 到达 `manuscript_allowed`。路线图
   优化的是链条的最后一环, 而链条的第一个受审计科学决定从未发生。
2. **Cycle 1 的目标已经以更硬的形式建成。** "把合同/证据清单/审稿模拟/PDF 检查
   变成强制" 这部分活了下来 (见第 1, 2 节), 但落地为 fail-closed 闸门, 不是
   按周排期的质量迭代。
3. **Cycle 2-6 预设的 "论文流" 不存在。** 后五个周期都假设系统持续生成论文供
   迭代打磨。现实是部署实例自治开关关闭 (2026-08-03 核实), 工作以 bounded
   pilot 一次一个候选推进; 唯一闭合的链条 (OutcomeRecord id=1, 2026-08-04) 以
   failed/inconclusive 收场, 阶梯未动。没有流, 就没有可打磨的对象。
4. **按日历排到 "投稿彩排" 与诚实口径冲突。** 18 周从 2026-06-26 排到约
   2026-10-30 收束于投稿彩排; 而现行 manuscript 闸门要求 supported verdict +
   审稿人批准, 目前没有任何工作拿到过。保留一个里程碑在制度上不可能按期发生的
   日程表, 违反 README 的两账制纪律。

取代它的不是新日程表, 而是现行推进方式: 一次一个 bounded pilot, 证据阶梯说了
算。当第一个 `scientifically_decided` 出现时, 再谈论文级别的打磨优先级才有
意义。
