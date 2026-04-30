# DeepGraph → SciForge: 架构与待完成任务

## 2026-04-29 本地接手状态

当前工作区：`E:\Download\Softwares\Deepgraph`

当前重点已经从外部 EvoScientist 执行，收敛为复用本仓库现有 `deep_insights` / `experiment_runs` 的 SciForge 闭环：

```text
deep_insight
  -> research_spec.json
  -> capability selection
  -> forge
  -> single-script validation or benchmark-suite validation
  -> statistical_report.json
  -> evidence_gate.json
  -> manuscript report / paper_candidate.md
  -> AI review
  -> followup_experiment_plan.json
```

本轮新增/补齐的主要模块：

- `agents/research_spec_compiler.py`
- `agents/capability_registry.py`
- `agents/benchmark_suite.py`
- `agents/statistical_reporter.py`
- `agents/evidence_gate.py`
- `agents/review_planner.py`
- `benchmarks/fairness_classification/*`
- `agents/manuscript_writer.py`
- `agents/ai_reviewer.py`

关键原则：

- 不新增平行 research-project CRUD；继续以 `experiment_runs` 为唯一实验执行对象。
- 不写假接口，不写没有 workflow 调用或测试覆盖的冗余接口。
- 失败后先分类再修改：`bottom_logic_bug`、`dependency_or_environment`、`llm_prompt_or_scaffold`、`data_or_evidence_insufficient`、`scientific_hypothesis_unsupported`。
- 不通过弱化阈值、减少种子数、改提示词硬跑来掩盖失败。

当前验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

最近一次结果：`Ran 100 tests ... OK`。

已知限制：

- 当前可离线跑通的真实 benchmark capability 是 grouped fairness classification。
- `safe_rl_cmdp` 只在 capability registry 中声明为 planned，尚未实现 harness。
- `paper_candidate.md` 只表示 evidence gate 通过后的候选稿，不等价于“可直接投稿”。
- 真实投稿质量仍需要更多数据集、消融、引用核验和人类审稿。

## 当前系统状态

| 指标 | 数值 |
|------|------|
| 论文 | 2,790 篇（已处理 2,566） |
| Result tuples | 114,551 |
| Taxonomy 节点 | 470 |
| Research Insights | 123（已全部 paradigm 评分） |
| 矛盾 | 34 |
| 知识图谱实体 | 47,837 |
| 知识图谱关系 | 276,999 |
| Token 消耗 | 38.6M |

## 已完成的链路

```
[1] arXiv 拉论文 ──► [2] 提取知识 ──► [3] 发现 Insight ──► [4] 评分排名 ──► [5] 生成研究方案
     ✅ 自动          ✅ 自动          ✅ 自动             ✅ 自动          ✅ 按钮触发
```

### [1] 论文摄取（自动）
- `ingestion/arxiv_client.py` 按 arXiv 分类拉论文
- `ingestion/pdf_parser.py` 下载 PDF 提取全文
- 3 个 LLM provider 并发（tabcode/kimi/minimax）

### [2] 知识提取（自动）
- `agents/extraction_agent.py` → claims, results, methods
- `db/taxonomy.py` + `agents/taxonomy_expander.py` → 自动扩展分类树
- `db/evidence_graph.py` → 实体-关系知识图谱
- `agents/reasoning_agent.py` → 矛盾检测

### [3] Insight 发现（自动）
- `agents/insight_agent.py` → 跨论文推理，产出 5 类 insight：
  - 矛盾分析、方法迁移、假设挑战、被忽略限制、范式枯竭
- 每个 insight 有：假设 + 证据 + 实验设计 + 影响

### [4] Paradigm 评分（自动）
- `agents/insight_ranker.py` → 1-10 范式颠覆性评分 + 理由
- 存入 DB 的 `paradigm_score` 和 `rank_rationale` 字段

### [5] 生成研究方案（按需）
- `agents/research_bridge.py` → 聚合 30 篇论文 + 40 条 claims + 矛盾 + limitations → 18KB 研究提案
- EvoScientist（6 个 sub-agent）→ 33KB 完整研究方案：
  - 新颖性验证（搜索 2024-2026 是否已有人做）
  - 4 阶段实验设计（具体模型/数据集/指标）
  - 量化成功标准
  - 算力估算
  - 风险 + fallback
  - 论文大纲 + 标题 + 摘要草稿

## 当前补齐的最后一步

```
[5] 研究方案 ──► [6] 执行实验 ──► [7] 分析结果 ──► [8] 写论文/审稿
                  ✅ SciForge      ✅ 结构化产物      ✅ manuscript + review
```

### 已完成

- `agents/artifact_manager.py`：每个 experiment workdir 维护 `artifact_manifest.json`，记录 metrics、logs、execution plan、manuscript、review 等产物。
- `agents/experiment_plan_compiler.py`：把 `deep_insight.experimental_plan`、scaffold 和 codebase 信息编译为 `execution_plan.json`，不新增平行 project 系统。
- `agents/validation_loop.py`：实验执行后写出 `artifacts/results/metrics.json`、`artifacts/results/iterations.json` 和 `artifacts/logs/run.log`。
- `agents/result_interpreter.py` + `agents/knowledge_loop.py`：解释 DB 与 artifact 中的实验结果；如果两者不一致，停止 cascade，避免污染知识图谱。
- `agents/manuscript_writer.py`：从已完成或已反驳的 `experiment_run` 生成 grounded manuscript package。反驳结果生成 `negative_result_report.md`，不会硬写正面论文。
- `agents/ai_reviewer.py`：对生成的 manuscript package 运行结构化 AI review，输出 `review.json` 和 `review.md`。
- `web/app.py`：复用现有 `/api/experiments/*`，只新增两个动作端点：
  - `POST /api/experiments/<run_id>/manuscript`
  - `POST /api/experiments/<run_id>/review`
- `web/static/js/app.js`：Experiments 详情页显示 artifacts，并提供 Generate Manuscript / AI Review 按钮。

### 仍待加强

- SLURM/GPU 后端：当前重点是本地/CPU/proxy task 闭环，尚未提交远程集群。
- 更丰富的论文模板：目前输出 Markdown manuscript、BibTeX、reproducibility statement；LaTeX 模板仍可增强。
- 多轮 rebuttal 自动化：当前有 AI review，尚未实现根据审稿意见自动排队补实验和 rebuttal 草稿。
- 真实投稿质量仍需要人类把关：系统保证 grounding 和 artifact 可追溯，不保证论文一定可发表。

## 文件位置

### 代码
```
/home/ec2-user/deepgraph/                  # DeepGraph 主代码
├── agents/research_bridge.py              # DeepGraph → EvoScientist 桥接（改这个）
├── agents/insight_agent.py                # Insight 发现
├── agents/insight_ranker.py               # Paradigm 评分
├── web/app.py                             # API 端点（加新端点）
├── web/static/js/app.js                   # 前端（加按钮）
└── config.py                              # LLM 配置

/home/ec2-user/EvoScientist/               # EvoScientist 源码
├── EvoScientist/EvoScientist.py           # Agent 构建（create_cli_agent）
├── EvoScientist/subagent.yaml             # 6 个 sub-agent 定义
├── EvoScientist/prompts.py                # System prompts
└── .venv/bin/EvoSci                       # CLI 入口
```

### 配置
```
~/.config/evoscientist/config.yaml         # EvoScientist LLM 配置
/home/ec2-user/deepgraph/config.py         # DeepGraph LLM 配置
```

### 产出示例
```
/home/ec2-user/research/
├── insight_6_Argument-graph_.../
│   ├── research_proposal.md               # DeepGraph 输入（18KB）
│   ├── final_report.md                    # EvoScientist 输出（16KB）
│   └── todos.md                           # 执行清单
├── insight_103_Leaderboard_.../
│   ├── final_report.md                    # 33KB 完整方案
│   └── todos.md
├── insight_scan/
│   └── final_report.md                    # 123 insights 排名报告
└── paradigm_break/
    └── final_report.md                    # 5 个范式断裂分析
```

## LLM 配置

### DeepGraph（3 provider 并发）
| Provider | URL | Model | 用途 |
|----------|-----|-------|------|
| tabcode | `api2.tabcode.cc/openai` (Responses API) | gpt-5.4 | 主力提取 |
| kimi | `api.kimi.com/coding/v1` | kimi-latest | 备用 |
| minimax | `api.minimaxi.com/v1` | MiniMax-M1 | 备用 |

### EvoScientist
```yaml
# ~/.config/evoscientist/config.yaml
provider: "custom-openai"
custom_openai_base_url: "https://api2.tabcode.cc/openai"
custom_openai_api_key: "sk-user-..."
model: "gpt-5.4"
```
注意：tabcode 只支持 Responses API，启动时需要 `CUSTOM_OPENAI_USE_RESPONSES_API=true`。

## 启动方式

```bash
# 启动 dashboard
cd /home/ec2-user/deepgraph && python3.12 main.py

# 暴露到公网
cloudflared tunnel --url http://localhost:8080

# 手动跑 pipeline
curl -X POST http://localhost:8080/api/start -H 'Content-Type: application/json' -d '{"max_papers": 100}'

# 手动生成研究方案
curl -X POST http://localhost:8080/api/research/launch -H 'Content-Type: application/json' -d '{"insight_id": 103}'
```
