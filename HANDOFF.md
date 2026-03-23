# DeepGraph → EvoScientist: 架构与待完成任务

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

## 未完成的最后一步

```
[5] 研究方案 ──► [6] 执行实验 ──► [7] 分析结果 ──► [8] 写论文
                  ❌ 待做          ❌ 待做          ❌ 待做
```

### 目标
让 EvoScientist 接收 `final_report.md` 中的研究方案，自动：
1. 生成可运行的实验代码
2. 执行实验（或提交到集群）
3. 收集结果、画图、做统计
4. 写出论文初稿

### EvoScientist 已有的 agent 能力
EvoScientist 内置 6 个 sub-agent，其中后 4 个就是做这件事的：

| Agent | 作用 | 当前状态 |
|-------|------|----------|
| Planner | 设计实验阶段 | ✅ 已在用 |
| Research | 搜索验证新颖性 | ✅ 已在用 |
| **Code** | **写实验代码** | ⚠️ 能力在但未接入链路 |
| **Debug** | **调试代码** | ⚠️ 同上 |
| **Analysis** | **分析结果、画图** | ⚠️ 同上 |
| **Writing** | **写论文** | ⚠️ 同上 |

### 需要做的事

#### 任务 1：接续执行（核心任务）
在 `research_bridge.py` 的 `launch_evoscientist()` 之后，加一个 `execute_plan()` 函数：
- 读取 `final_report.md` 中的实验计划
- 让 EvoScientist 进入执行模式（不是规划模式）
- Prompt 类似：`"Read final_report.md. Execute Stage 1. Write code, run it, collect results."`
- EvoScientist 的 Code Agent 会写代码，Debug Agent 会修 bug，Analysis Agent 会分析结果

实现方式：
```python
def execute_plan(insight_id: int, stage: int = 1):
    """让 EvoScientist 执行研究方案中的某个阶段"""
    workdir = find_workdir(insight_id)
    prompt = f"Read final_report.md. Execute Stage {stage}. Write runnable code, execute it, collect results into artifacts/."
    # Launch EvoScientist with --workdir pointing to existing workspace
    # It will see final_report.md + todos.md and continue from there
```

关键点：EvoScientist 的 workspace 是持久的，可以用 `/resume` 在同一个 thread 上继续对话。

#### 任务 2：计算资源适配
当前机器没有 GPU。两种路径：
- **路径 A**：挑选不需要 GPU 的 insight（meta-analysis、统计分析、benchmark 审计），在 CPU 上直接跑完
- **路径 B**：Code Agent 生成代码 + SLURM 脚本，提交到有 GPU 的集群，等结果回来后继续分析

#### 任务 3：结果 → 论文
EvoScientist 的 Writing Agent 已有论文写作能力。需要：
- 实验产出的 figures/tables 放在 `artifacts/` 下
- Prompt Writing Agent：`"Read final_report.md and artifacts/. Write a complete paper draft to paper.md."`
- 它会产出 Markdown 格式的论文，包括 Introduction、Related Work、Method、Experiments、Results、Discussion、Conclusion

#### 任务 4：Dashboard 集成（可选）
在 Insights tab 的每个卡片上加：
- `Execute Stage N` 按钮（依次执行实验阶段）
- 实时显示执行状态（running/completed/failed）
- 查看产出的 artifacts（图表、代码、论文草稿）

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

## 本次分支记录

- 分支：`issue-1-opportunity-closed-loop`
- 已实现：机会 triage 队列、评分排序、Dashboard 展示、pipeline 末尾重建 triage
- 验证：`python -m pytest -q tests/test_opportunity_engine.py tests/test_web_app.py`
- 提醒：这版先补“发现 → 排序”的中段闭环，自动实验执行仍在后续计划中

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
