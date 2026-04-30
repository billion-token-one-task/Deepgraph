# DeepGraph / SciForge 当前进展与下一步路线

日期：2026-04-29

## 一句话状态

当前项目已经从“能发现课题但实验链路不可靠”推进到“可以把 deep insight 转成标准 research spec，选择真实 capability，运行单脚本或 benchmark-suite 实验，记录 artifact，生成 evidence-gated manuscript，调用 AI reviewer，并把审稿意见转成 follow-up experiment plan”。

但是，它目前还不能自动承诺产出可发表论文。系统现在可以产出 `paper_candidate.md`，但这个状态只表示 artifact/evidence gate 通过了当前离线 benchmark 标准，不等于真实投稿已经充分。主要限制仍是数据集覆盖、消融、引用核验、领域审稿和更广的实验能力。

## 本轮通用框架进展

开发计划文档：

```text
docs/superpowers/plans/2026-04-29-generalized-research-execution-framework.md
```

已经按计划落地：

- `research_spec.json`：把 insight/title/hypothesis 编译成通用研究规格，不把逻辑写死到某个标题。
- `capability_registry.py`：按 domain/task 选择真实 capability；当前实现 `fairness_classification`，保留 `safe_rl_cmdp` 为 planned。
- `benchmarks/fairness_classification/*`：离线 grouped fairness benchmark，支持多 seed、多方法、真实指标。
- `benchmark_suite.py`：读取 `benchmark_config.json`，运行 capability harness，写 `benchmark_results.json`。
- `statistical_reporter.py`：写 `statistical_report.json` 和 `tables/main_results.md`。
- `evidence_gate.py`：区分 `preliminary`、`needs_more_experiments`、`not_publishable`、`paper_ready_candidate`。
- `experiment_forge.py` / `validation_loop.py`：已接入 research spec、capability selection、benchmark suite 和 evidence gate，保留旧 single-script 路径。
- `manuscript_writer.py`：无 evidence gate 时默认 `preliminary_report.md`；证据不足写 `additional_experiments_required.md`；只有 gate 允许时写 `paper_candidate.md`。
- `ai_reviewer.py`：支持 review 新的 manuscript artifact 类型。
- `review_planner.py`：把 `review.json` 中的 required experiments 写成 `followup_experiment_plan.json`。
- `web/app.py` / `web/static/js/app.js`：实验详情页复用 artifact manifest 显示 evidence gate/follow-up 状态，并提供 follow-up planning action。

最近验证：

```text
.\.venv\Scripts\python.exe -m unittest discover -s tests
Ran 100 tests ... OK
```

## 项目现在主要能做什么

当前链路是：

```text
deep_insight
  -> experiment_forge
  -> validation_loop
  -> result_interpreter / knowledge_loop
  -> manuscript_writer
  -> ai_reviewer
```

也就是说，它现在可以：

- 从已有知识图谱和 deep insight 中选出一个待验证假设。
- 为该假设创建实验目录和执行计划。
- 运行 baseline reproduction。
- 让 LLM coding agent 修改实验代码。
- 对每轮实验结果做 keep / discard / crash 判断。
- 把 metrics、iterations、logs、manuscript、review 都写成 artifact。
- 把实验结论回写到数据库和知识循环。
- 生成正向 manuscript 或 negative-result report。
- 调用 AI reviewer 给出 structured review，并指出是否不够发表。

## 接手时已有的基础能力

下面这些是项目在我们本轮修改前已经具备的基础，不是本轮从零新增：

- 已有 Flask Web dashboard，入口在 `web/app.py` 和 `web/static/js/app.js`。
- 已有 SQLite 数据库和 schema，包含 papers、claims、taxonomy、deep insights、experiment runs 等核心表。
- 已有知识图谱/文献处理相关 agent，例如 taxonomy、abstraction、signal harvesting、novelty verification、knowledge feedback 等。
- 已有 `deep_insights` 作为自主课题发现结果的主要对象。
- 已有 `experiment_runs` 和 `experiment_iterations` 作为实验执行记录的主要对象。
- 已有 `agents/experiment_forge.py` 雏形，用于把 deep insight 转成实验 scaffold。
- 已有 `agents/validation_loop.py` 雏形，用于 baseline reproduction 和 hypothesis-testing loop。
- 已有 `agents/result_interpreter.py`、`agents/knowledge_loop.py`、`agents/meta_learner.py` 等后处理/反馈组件。
- 已有远程 LLM client 机制，项目本身不是本地模型推理系统。
- 已有 dashboard 中的 deep insight、experiment 相关入口，只是当时实验链路还不可靠，artifact/manuscript/review 部分也没有完整闭环。

因此，项目原本已经有“自主科学发现引擎”的骨架：文献/知识图谱 -> insight -> 实验 run -> 知识反馈。我们这轮工作主要是把实验执行、artifact、manuscript、review 和失败判定补成可验证的闭环。

## 本轮新增/修复的工程改动

### 1. Artifact 管理

本轮新增了 artifact manifest 机制，实验输出不再散落在目录里。

现在每个实验 workdir 下会有：

```text
artifact_manifest.json
artifacts/logs/
artifacts/results/
artifacts/figures/
artifacts/tables/
artifacts/manuscript/
artifacts/reviews/
```

已经记录的 artifact 类型包括：

- `execution_plan`
- `metrics`
- `iterations`
- `log`
- `manuscript`
- `references`
- `reproducibility`
- `review`

### 2. Structured execution plan

本轮新增 `agents/experiment_plan_compiler.py`。

它把 deep insight、实验计划、scaffold、codebase 信息整理成：

```text
execution_plan.json
```

这个文件记录：

- hypothesis
- primary metric
- metric direction
- stages
- success criteria
- datasets
- baselines
- codebase

### 3. Validation loop 修复

修复了多个底层问题：

- 不再把旧 evaluator 输出的 `metric_value: 0.0` 当成真实 metric。
- 如果 `evaluate.py` 失败，但 run log 里已经有真实 metric，会保留 run log metric。
- validation loop 会读取 `program.md` 里的 run command，不再盲目只跑 `train.py`。
- 没有 baseline metric 时会写出失败 artifact。
- confirmed 必须有真实 hypothesis-testing improvement，不允许 baseline-only 成功。
- 对已有 git repo 会先提交 scaffold baseline，避免 `git reset` 把生成的 `train.py` 删除。
- Tier 1 insight 现在会把 `formal_structure`、`transformation`、`experimental_plan` 传给 coding agent，而不是只传空的 `proposed_method`。

### 4. Experiment forge 修复

修复了 scaffold 生成的问题：

- LLM 没给 `train.py` 时会生成透明 bootstrap harness。
- fallback evaluator 不再伪造 `0.0` metric。
- 非 scratch 代码库如果只是库源码，没有 runnable experiment entrypoint，也会生成 benchmark harness。
- Fairlearn 代码库会生成一个真实的小型 Fairlearn/sklearn proxy benchmark。

Fairlearn fallback benchmark 会输出：

```text
accuracy
demographic_parity_gap
fairness_score
```

其中：

```text
fairness_score = accuracy - 0.45 * demographic_parity_gap
```

### 5. Manuscript writer

本轮新增 `agents/manuscript_writer.py`。

它现在可以：

- 对 confirmed / inconclusive run 生成 `paper.md`。
- 对 failed / refuted run 生成 `negative_result_report.md`。
- 生成 `references.bib`。
- 生成 `reproducibility.md`。
- 把 artifact 路径写入 manifest。
- 在 manuscript 中写入 metric definition、audit artifact 路径和 per-iteration audit trail。
- 避免把 proxy result 描述成投稿级科学确认。

### 6. AI reviewer

本轮新增 `agents/ai_reviewer.py`。

它会生成：

```text
artifacts/reviews/review.json
artifacts/reviews/review.md
```

review schema 包括：

- `overall_score`
- `recommendation`
- `major_concerns`
- `minor_concerns`
- `required_experiments`
- `citation_risks`
- `reproducibility_risks`

### 7. Web/API 集成

复用了现有实验 API，没有新建并行 research-project 系统。

新增或增强：

- `GET /api/experiments/<run_id>` 返回 artifact 列表。
- `POST /api/experiments/<run_id>/manuscript`
- `POST /api/experiments/<run_id>/review`
- `POST /api/experiments/run_full` 可选 manuscript/review。

## 当前本地实验记录

### Run #1：原始 Fairlearn 失败

状态：

```text
status = failed
hypothesis_verdict = null
```

失败原因不是数据太少，而是底层逻辑问题：

- scaffold 选择了 Fairlearn 库内部模块作为入口，不是真正 benchmark。
- 旧 validation loop 忽略了 `program.md` 的 run command。
- 当时虚拟环境缺少 Fairlearn 运行依赖。

这个 run 保留为诊断证据。

### Run #2：synthetic baseline healthcheck

状态：

```text
status = completed
hypothesis_verdict = inconclusive
baseline accuracy = 0.8
best accuracy = 0.8
```

用途：验证 baseline-only 不会被误判为 confirmed。

### Run #3：synthetic improvement healthcheck

状态：

```text
status = completed
hypothesis_verdict = confirmed
baseline accuracy = 0.8
best accuracy = 0.9
```

用途：验证整条 plumbing 可以成功跑通。

注意：这是合成健康检查，不是科学结果。

### Run #4：Fairlearn proxy 初次 rerun

状态：

```text
status = completed
hypothesis_verdict = refuted
baseline fairness_score = 0.5007863219992292
best fairness_score = 0.5007863219992292
```

失败原因不是科学假设被真正反驳，而是当时发现了两个环境/底层问题：

- coding agent 的远程 LLM 调用被 shell 沙箱拦截。
- cloned repo 的 git reset 会删除未纳入 baseline commit 的 `train.py`。

这两个问题已经修复或通过授权解决。

### Run #5：Fairlearn proxy 成功 rerun

状态：

```text
status = completed
hypothesis_verdict = confirmed
```

结果：

```text
primary metric = fairness_score
baseline = 0.5007863219992292
best = 0.6157093943616044
effect = +0.1149230723623752
effect_pct = +22.95%
iterations_total = 5
iterations_kept = 3
```

workdir：

```text
sciforge_runs/exp_7_Fairlearn_preference_cone_proxy
```

artifact：

```text
execution_plan.json
artifacts/results/metrics.json
artifacts/results/iterations.json
artifacts/logs/run.log
artifacts/manuscript/paper.md
artifacts/manuscript/references.bib
artifacts/manuscript/reproducibility.md
artifacts/reviews/review.json
artifacts/reviews/review.md
```

AI reviewer 结果：

```text
overall_score = 1
recommendation = reject
```

解释：工程链路跑通了，proxy benchmark 也取得了提升。但 reviewer 正确指出，这不足以证明原始大命题。

## 当前为什么还不能输出可发表论文

当前 insight 标题是：

```text
Social-choice CMDPs: safe RL and group fairness as the same constrained preference-cone optimization
```

这个主张很大，涉及：

- social choice
- constrained MDP
- safe RL
- group fairness
- preference cone optimization

而 run #5 只是一个 Fairlearn/sklearn grouped classification proxy benchmark。

它能支持的弱结论大概是：

```text
在一个小型 grouped classification proxy benchmark 上，
preference-cone-style threshold selection 可以改善本地定义的 fairness_score。
```

它不能支持：

```text
safe RL 和 group fairness 在 CMDP preference-cone 框架下被统一证明，
并且该统一方法在真实 constrained RL / fairness benchmark 上有效。
```

主要缺口：

- 没有真实 constrained RL / safe RL 环境。
- 没有多数据集。
- 没有多 seed。
- 没有 confidence interval 或强统计检验。
- 没有强 baseline suite。
- 没有 ablation。
- 没有 sensitivity analysis。
- 没有正式数学推导。
- citation 只是 evidence list，还不是 related work synthesis。

## 接下来要升级什么

推荐把下一阶段目标定义为：

```text
把系统从“能跑单个 proxy experiment”
升级成“能自动构建 benchmark suite、统计报告、AI 审稿反馈、下一轮补实验计划”的科研闭环。
```

### 阶段 1：Benchmark suite

新增：

```text
agents/benchmark_suite.py
```

职责：

- 管理 dataset × seed × baseline × method 的实验矩阵。
- 输出统一格式的 result table。
- 支持 timeout 和失败记录。
- 不替代 `validation_loop`，而是作为 validation loop 的可执行实验后端。

第一版建议只做 Fairlearn/group fairness，不直接上 safe RL。

建议实验矩阵：

Datasets：

- synthetic grouped classification
- Adult 或本地可生成 Adult-like proxy
- COMPAS 或本地可生成 COMPAS-like proxy

Baselines：

- LogisticRegression
- Fairlearn ExponentiatedGradient
- Fairlearn ThresholdOptimizer
- unconstrained threshold search
- proposed preference-cone threshold method

Metrics：

- accuracy
- demographic parity gap
- equalized odds gap
- fairness_score
- runtime_seconds

Seeds：

- 最少 10 个 seed

### 阶段 2：Statistical reporter

新增：

```text
agents/statistical_reporter.py
```

职责：

- 汇总 benchmark suite 输出。
- 计算 mean/std。
- bootstrap confidence interval。
- paired test。
- effect size。
- 生成 machine-readable `statistical_report.json`。
- 生成 manuscript 可引用的 markdown/table。

输出：

```text
artifacts/results/benchmark_results.json
artifacts/results/statistical_report.json
artifacts/tables/main_results.md
```

### 阶段 3：Review-to-next-experiments

新增：

```text
agents/review_planner.py
```

职责：

- 读取 `review.json`。
- 抽取 `required_experiments`。
- 转成下一轮 `execution_plan.json` 的候选补实验。
- 标记当前 manuscript 是否 publishable。

建议规则：

```text
review recommendation = reject
  -> manuscript_status = not_publishable
  -> generate follow-up experiment plan

review recommendation = major_revision
  -> manuscript_status = needs_more_experiments

review recommendation = accept / weak_accept
  -> manuscript_status = paper_ready_candidate
```

注意：即使 reviewer 给 accept，也不代表自动投稿，只代表可以进入人工检查。

### 阶段 4：Manuscript quality gate

升级 `manuscript_writer`：

- 如果没有 benchmark suite，就只能写 preliminary report。
- 如果没有 statistical report，就不能写 positive paper。
- 如果 AI reviewer reject，就不能标记 paper-ready。
- 如果 citation grounding 不足，就生成 related-work-needed report。

建议增加：

```text
manuscript_status
```

取值：

```text
preliminary
not_publishable
needs_more_experiments
paper_ready_candidate
```

### 阶段 5：Safe RL / CMDP 扩展

只有 Fairlearn/group fairness benchmark 稳定后，再扩展到 safe RL。

需要新增：

- constrained bandit 环境
- gridworld CMDP 环境
- occupancy-measure LP baseline
- safety violation cost
- preference cone constraints
- fairness classification 和 CMDP 的统一接口

这一步才开始真正接近原始大命题。

## 推荐实施顺序

### Step 1：先做 Fairlearn/group fairness benchmark suite

原因：

- 依赖已经装好。
- 当前 run #5 已证明小 proxy 可运行。
- 范围可控。
- 可以最快把系统从单实验升级到多 seed、多 baseline、多指标。

### Step 2：接 statistical reporter

没有统计报告就不应该写正向论文。

### Step 3：让 reviewer 输出变成下一轮实验计划

这样系统才是闭环，不只是“跑完被拒”。

### Step 4：把 manuscript writer 改成 evidence-gated

让它在证据不足时自动写 preliminary / negative / additional-experiments-needed，而不是强行写 paper。

### Step 5：再上 safe RL / CMDP

等 group fairness benchmark 体系可靠后，再扩展到原始大命题。

## 通用标题适配升级

后续标题会变化，因此下一阶段不能继续把逻辑写死在 Fairlearn 或当前这个 social-choice/CMDP 标题上。推荐升级为通用科研执行框架：

```text
deep_insight/title
  -> research_spec_compiler
  -> capability_registry
  -> experiment_forge
  -> benchmark_suite or single_script_validation
  -> statistical_reporter
  -> evidence_gate
  -> manuscript_writer
  -> ai_reviewer
  -> review_planner
  -> next experiment
```

核心原则：

- 标题和 insight 先编译成标准 `research_spec.json`。
- 系统根据 spec 匹配 capability，而不是在代码里写死某个标题或某个仓库。
- benchmark harness 从 `experiment_forge` 中拆出来，形成可复用的 capability 后端。
- `validation_loop` 继续作为执行入口，但可以运行单脚本或 benchmark suite。
- manuscript 是否能写成论文候选由 `evidence_gate` 判断，不由 `confirmed` 字段单独决定。
- AI reviewer 的 reject 进入 `review_planner`，生成下一轮补实验计划。

完整开发计划已写入：

```text
docs/superpowers/plans/2026-04-29-generalized-research-execution-framework.md
```

## 下一阶段验收标准

下一阶段完成后，至少应该达到：

- 能自动跑 dataset × seed × method 的 benchmark suite。
- 至少 10 seed。
- 至少 3 baseline。
- 每个结果有 artifact。
- 有 `statistical_report.json`。
- manuscript 中包含主结果表和统计摘要。
- AI reviewer 不再主要批评“没有 benchmark / 没有统计 / baseline 不清楚”。
- 如果 reviewer 仍 reject，系统能生成下一轮补实验计划。

## 当前测试状态

### 2026-04-30 继续执行记录

本轮按“先定位失败原因，再定点修改”的原则继续推进，重点不是把 reviewer 调松，而是修掉真实底层问题并让证据范围更诚实。

已完成：

- 修复 benchmark claim 过度继承原始 `Social-choice CMDPs` 标题的问题：benchmark 模式下 `experimental_claims` 使用 `benchmark_config.scoped_claim`。
- 修复 benchmark interpretation 沿用原始 `metric_direction=lower` 的 bug：benchmark suite 以 `statistical_report.metric_direction` 为准。
- 修复 workdir 污染后继续验证 fresh run 行为。
- 新增 OpenML / Fairlearn 公共 fairness 数据集：
  - `openml_adult_sex`
  - `openml_german_credit_sex`
  - `fairlearn_credit_card_sex`
  - `fairlearn_bank_marketing_age`
- 将 synthetic / sklearn 人工分组数据移出默认 paper-ready benchmark，只保留为开发测试数据。
- 修复 preference-cone 阈值搜索在 Adult 上近似二次复杂度导致完整流程超时的问题：大数据集阈值候选使用 quantile cap。
- 修复普通预测模型可能吃到 protected attribute 的公平性 bug：普通 Logistic/Fairlearn estimator fit 前移除最后一列 protected attribute；protected attribute 仅作为 `sensitive_features` 或 threshold routing 使用。
- 新增 `validation_selected_fairlearn_baseline`，让 Fairlearn baseline 也能用训练内 validation 按同一 scalar objective 选择。
- 将默认 headline method 从直接训练阈值搜索切换为 `validation_selected_preference_cone`，直接训练阈值搜索降为 ablation。
- `statistical_reporter` 增加：
  - secondary metric summaries；
  - best-vs-all pairwise comparisons；
  - group-conditional metrics进入 benchmark rows。
- `manuscript_writer` 增加：
  - algorithmic specification；
  - train/validation/test protocol；
  - related work context；
  - statistical procedure；
  - dataset/environment details；
  - trade-off interpretation；
  - protected-attribute leakage/comparability说明。
- `evidence_gate` 修复：`weak_reject` / `borderline` 会阻断 `paper_ready_candidate`，进入 `needs_more_experiments`。

真实 pipeline 验证：

- Run #9：完整流程超时。归因：Adult 上 preference-cone 阈值搜索复杂度过高，不是 LLM prompt 或数据太少。
- Run #10：跑通但 AI review = `reject`。归因：metric direction 反向解释、稿件内部不一致、人工分组数据和过宽 claim。
- Run #11/#12：收窄到真实 protected-attribute 数据集后，AI review 从 `reject` 进步到 `weak_reject`。
- Run #13/#14：修复 protected attribute 泄漏、新增 fairlearn baseline 和 age 数据集后仍为 `weak_reject`。
- Run #15：headline 改为 `validation_selected_preference_cone` 后完整跑通，AI review 仍为 `weak_reject`，evidence gate 正确降为 `needs_more_experiments`。

当前结论：

- 工程闭环已经能真实执行：forge -> benchmark suite -> statistical report -> evidence gate -> manuscript -> AI review -> follow-up plan。
- 但还不能诚实地产出“通过 AI review 的可发表论文”。当前阻塞不是依赖或底层执行失败，而是科学贡献/证据问题：
  - 方法新颖性不足，容易被视为已有 group-specific threshold post-processing 的变体；
  - 主指标是方法直接优化的 scalar objective，construct validity 风险仍在；
  - 相对 fairness-aware baselines 的优势不够稳定；
  - equalized-odds 等非优化 fairness metric 有退化；
  - 缺少 Pareto/lambda sweep、更多 protected attributes、更完整 baseline hyperparameter tuning 和多重比较处理。

下一步如果继续追求 review pass，不能再靠稿件措辞硬推，需要真正补实验和/或提出更有新颖性的算法：

- 让所有 baselines 也在同一 validation scalar objective 下做 lambda/grid sweep。
- 输出 full Pareto frontier，而不只固定 `fairness_score`。
- 增加 aggregate-level CI/significance，并明确多重比较处理。
- 把 raw seed-level results、group metrics、package versions、dataset IDs 写入 manuscript/reproducibility artifact。
- 如果仍要对齐原始 `Social-choice CMDPs` 大命题，需要实现 `safe_rl_cmdp` capability，而不是继续用 fairness classification proxy 代替。

最近一次验证：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

结果：

```text
Ran 127 tests in 11.620s
OK
```

仍存在的测试警告：

- sqlite connection ResourceWarning。
- `datetime.utcnow()` DeprecationWarning。

这些警告不是当前科研链路阻塞项，但后续可以清理。

## 重要原则

后续继续开发时保持这几条：

- 不写假接口。
- 不新建冗余工作流对象。
- 尽量复用 `experiment_runs`、artifact manifest、validation loop、manuscript/review API。
- 不靠调参数硬凑 success。
- 失败时必须分类：底层逻辑问题、依赖问题、LLM prompt/scaffold 问题、数据/证据不足、科学假设本身不成立。
- AI reviewer 的 reject 应该触发下一轮实验设计，而不是被忽略。
