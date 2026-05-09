# Paper-Grade Benchmark Agent Plan

本文档定义 DeepGraph 中用于 CGGR / selective deliberation QA 的论文级真实 benchmark 实验组织方式。目标不是跑通脚本，而是让每个 agent 只对自己的合同负责，并产出可审计、可复现、可写入论文主文的实验 artifact。

## Non-Negotiable Rules

- 真实 benchmark 才能支持论文 claim。Smoke / sanity run 只能验证 infrastructure，例如数据加载、prompt 渲染、远程 GPU 调度、结果落盘、metric parser 是否可用。
- Smoke / sanity 结果不得用于 abstract、introduction、results、conclusion、claim register、主表格或显著性结论。它们只能出现在 engineering log 或 appendix 的 infrastructure validation 描述中。
- 所有 claim 必须能追溯到 frozen benchmark contract、locked run matrix、raw generation、metric table、statistical audit 和 manuscript claim register。
- Contract 冻结后，任何 dataset、model、method、prompt、threshold、seed、metric、budget 的变化都必须进入 amendment log，并标记哪些旧结果作废。
- Test split 不可用于调参、threshold search、prompt selection、failure-driven patching 或 cherry-picking。所有 selection 只能基于 validation / calibration split。
- Oracle 只能作为 upper bound 或 diagnostic，不得被描述为 deployable baseline。
- 每个 method 必须在相同 dataset split、answer normalizer、base model revision、decode budget 和统计流程下比较；若预算不同，必须报告 token、latency、GPU hour 和 deliberation rate。

## Agent Roles And Contracts

### 1. Contract Architect

**Mission:** 把用户的研究问题转成不可变 benchmark contract 和 claim contract。

**Inputs**

- `research_question`: CGGR / selective deliberation 要证明的问题。
- `target_claims`: 允许论文尝试支持的 claim 列表。
- `resource_budget`: GPU 型号、GPU hours、API/model 权限、截止时间。
- `candidate_datasets`: GSM8K、StrategyQA、HotpotQA、MuSiQue、2WikiMultiHopQA。
- `candidate_models`: Qwen 与 Llama 系列的可用 checkpoint。
- `candidate_methods`: Direct、CoT、SelfConsistency、confidence gate、disagreement gate、random gate、oracle gate、CGGR variants。

**Forbidden**

- 不得写 results、改 method、调 threshold 或看 test label 后修改 contract。
- 不得把 smoke/sanity 结果写入 claim。
- 不得允许未 pin revision 的模型或未固定 split 的数据进入主实验。

**Output Artifacts**

- `benchmark_contract.yaml`: dataset/model/method/seed/metric/budget 的 frozen contract。
- `claim_register.md`: 每条 claim 对应的必要 evidence cells。
- `run_matrix.csv`: 全部 required、optional、ablation、diagnostic run cell。
- `amendment_log.md`: contract 变更、原因、影响范围、作废结果列表。

### 2. Dataset / Baseline Specialist

**Mission:** 锁定数据、标准化 answer extraction，并定义公平 baseline。

**Inputs**

- Frozen `benchmark_contract.yaml`。
- 原始 dataset 下载方式、license、split 说明。
- Baseline prompt family: Direct、CoT、SelfConsistency。
- Evaluation metric definition: exact match、numeric EM、F1、yes/no accuracy、supporting fact F1。

**Forbidden**

- 不得为了提高方法效果删除困难样本、重写 test split 或按结果筛样本。
- 不得让 CGGR 方法独享更强 answer normalizer、retrieval context、few-shot examples 或更高 token budget。
- 不得用 test set 选择 prompt wording、few-shot exemplars、threshold 或 parser fallback。

**Output Artifacts**

- `dataset_manifest.json`: dataset name、version/source、split、sample ids、checksum、license note。
- `normalizer_spec.md`: 每个 dataset 的 answer extraction、normalization、invalid output policy。
- `baseline_prompt_pack.md`: Direct/CoT/SelfConsistency prompts 和 few-shot source。
- `baseline_card.md`: 每个 baseline 的 budget、decode settings、expected output schema。
- `leakage_audit.md`: contamination / overlap / accidental label exposure 检查。

### 3. Harness Engineer

**Mission:** 建立可重复运行的实验 harness contract，使所有 method 在同一执行和记录标准下运行。

**Inputs**

- Frozen `benchmark_contract.yaml` 与 `run_matrix.csv`。
- Dataset manifest、normalizer spec、prompt pack。
- Model serving interface: HF local、vLLM、TGI、OpenAI-compatible endpoint 或 remote GPU command。
- Result schema requirements。

**Forbidden**

- 不得改变 method logic、prompt 内容、seed list 或 dataset cell。
- 不得吞掉失败样本；失败必须进入 result JSONL 并计入 audit。
- 不得只保存 aggregated metric 而丢弃 raw generations。

**Output Artifacts**

- `harness_contract.md`: CLI/API inputs、environment variables、expected result paths。
- `result_schema.json`: per-example JSONL schema。
- `run_launcher_spec.md`: 如何从 run matrix 精确启动每个 cell。
- `failure_taxonomy.md`: OOM、timeout、parse_error、invalid_answer、remote_disconnect 等分类。
- `infrastructure_smoke_report.md`: 只证明 harness 可跑通，不支持论文 claim。

### 4. Remote GPU Runner

**Mission:** 在远程 GPU 上执行 locked matrix，保存完整可审计日志。

**Inputs**

- `run_matrix.csv` 中分配给自己的 run cells。
- Harness command、container/conda environment、model path、GPU allocation。
- Resume/retry policy 与 artifact upload path。

**Forbidden**

- 不得临时改 prompt、method、dataset、seed、threshold、max tokens 或 checkpoint。
- 不得在失败后只重跑成功样本；必须按 run cell 完整 resume，并保留失败历史。
- 不得把本地 smoke run 伪装成 full benchmark。

**Output Artifacts**

- `remote_job_manifest.json`: host、GPU、driver、CUDA、container、commit/archive id、env vars。
- `run_stdout.log` / `run_stderr.log`: 含 stage markers 和启动命令。
- `raw_generations/*.jsonl`: 每个 example 的 prompt hash、response、parsed answer、token usage、latency、status。
- `retry_log.md`: 每次 retry 的原因、范围、旧 artifact 保留路径。
- `artifact_index.md`: 每个 run cell 对应的完整输出文件。

### 5. Method Worker

**Mission:** 在不触碰 benchmark contract 的前提下实现或配置 CGGR / selective deliberation 方法变体。

**Inputs**

- Frozen method slots: `direct`, `cot`, `self_consistency`, `cggr_confidence`, `cggr_disagreement`, `cggr_random`, `cggr_oracle`, `cggr_full`。
- Harness interface 与 result schema。
- Validation/calibration split。
- Budget constraints: max tokens、deliberation rate、number of samples、retrieval/context budget。

**Forbidden**

- 不得读取 test labels 做 controller、threshold、prompt 或 verifier 选择。
- 不得修改 baseline prompts 来削弱 baseline。
- 不得让 oracle 输出进入 deployable method comparison。
- 不得隐藏 deliberation cost；每个 decision 必须记录是否触发、触发原因、额外 token 和额外 latency。

**Output Artifacts**

- `method_card_<method>.md`: algorithm、inputs、decision rule、budget、known limitations。
- `method_config_<method>.yaml`: threshold、sample count、temperature、max tokens、seed policy。
- `controller_decisions_<run>.jsonl`: per-example gate score、trigger decision、deliberation path、final answer。
- `validation_selection_report.md`: threshold/budget selection 只基于 validation 的证据。
- `ablation_mapping.md`: 每个 ablation 与主方法哪个组件对应。

### 6. Stats / Audit

**Mission:** 从 raw outputs 生成统计有效、claim 可追溯的 benchmark evidence。

**Inputs**

- `raw_generations/*.jsonl`、dataset manifest、normalizer spec。
- Claim register、run matrix、failure taxonomy。
- Cost logs: token usage、latency、GPU hour。

**Forbidden**

- 不得添加/删除 run cells 来制造显著性。
- 不得把 parse failure 从 denominator 中静默移除。
- 不得只报告平均值而不报告 confidence interval、seed variance、missing cell 和 failure rate。
- 不得把 oracle upper bound 当成实际方法。

**Output Artifacts**

- `metrics_by_cell.csv`: dataset/model/method/seed 级别 metrics。
- `aggregate_results.csv`: mean、std、95% CI、n、failure rate、cost。
- `significance_report.md`: paired bootstrap / permutation / McNemar 等测试的选择理由和结果。
- `budget_audit.csv`: tokens、latency、deliberation rate、GPU hours。
- `claim_evidence_map.md`: 每条 manuscript claim 对应的 table/figure/run ids。
- `reproducibility_audit.md`: missing cells、non-determinism、retry、contract deviations。

### 7. Manuscript Writer

**Mission:** 只根据 audit-approved evidence 写作论文叙述、表格说明和 limitations。

**Inputs**

- Claim register、claim evidence map、aggregate results、significance report、budget audit。
- Dataset/model/method cards。
- Figure/table artifacts。

**Forbidden**

- 不得写任何没有 evidence id 支撑的性能 claim。
- 不得把 smoke/sanity 描述成实验结果。
- 不得将 validation-tuned observation 写成 held-out test conclusion。
- 不得隐藏负结果、失败率、预算成本或 oracle 的非部署性质。

**Output Artifacts**

- `results_section.md`: 主结果、ablation、cost/quality trade-off、failure analysis。
- `table_plan.md`: 主表、ablation 表、budget 表、dataset/model coverage 表。
- `limitations.md`: dataset/model/generalization 限制、oracle 限制、cost 限制。
- `reproducibility_statement.md`: contract、seeds、hardware、artifact availability。
- `claim_sanitization_report.md`: 被删除或降级的 unsupported claims。

## Benchmark Matrix For CGGR / Selective Deliberation QA

### Main Axes

| Axis | Required settings | Notes |
| --- | --- | --- |
| Datasets | GSM8K, StrategyQA, HotpotQA, plus MuSiQue or 2WikiMultiHopQA | GSM8K covers numeric reasoning; StrategyQA covers binary commonsense reasoning; HotpotQA/MuSiQue/2Wiki cover multi-hop QA. Use official validation/test protocol and freeze sample ids. |
| Models | At least one Qwen instruct checkpoint and one Llama instruct checkpoint | Pin exact model id, revision/hash, tokenizer, context length, quantization, serving engine. Preferred grid includes small and larger sizes when budget allows. |
| Methods | Direct, CoT, SelfConsistency, CGGR-confidence, CGGR-disagreement, CGGR-random, CGGR-oracle, CGGR-full | Random must be budget-matched; oracle is upper bound only; CGGR-full must declare the combination rule. |
| Seeds | Minimum 5 stochastic seeds: 11, 23, 37, 53, 71 | Direct deterministic runs still record seed and decode config. For cost-limited pilot, 3 seeds may be pre-registered, but paper claims must say limited-seed evidence. |
| Metrics | Accuracy/EM/F1 as dataset-appropriate, plus token cost, latency, deliberation rate, parse failure rate | Multi-hop datasets should report answer EM/F1; supporting fact metrics only if the method receives/evaluates support evidence consistently. |
| Budget | Equal max context, max output tokens, temperature policy, and deliberation budget across comparable methods | If SelfConsistency or CGGR uses extra calls, report quality-per-token and quality-per-latency. |

### Required Method Definitions

- `Direct`: answer with minimal instruction, no rationale requirement, one generation.
- `CoT`: one reasoning path before final answer, same base model and answer parser.
- `SelfConsistency`: k sampled CoT paths, majority or normalized answer vote; pre-register k, temperature, tie-break.
- `CGGR-confidence`: trigger deliberation when calibrated confidence falls below threshold selected on validation.
- `CGGR-disagreement`: trigger deliberation when sampled paths, graph candidates, or verifier signals disagree above validation-selected threshold.
- `CGGR-random`: trigger deliberation on a random subset matched to the same deliberation rate or token budget as CGGR-full.
- `CGGR-oracle`: trigger deliberation using label-aware hindsight to estimate an upper bound; never deployable and never part of main practical comparison.
- `CGGR-full`: the proposed selective deliberation controller combining allowed CGGR signals; every trigger must log score, reason, cost, and final override.

### Minimum Full Benchmark Grid

The minimum claim-supporting grid is:

```text
datasets = [GSM8K, StrategyQA, HotpotQA, MuSiQue_or_2Wiki]
models = [Qwen_instruct_pinned, Llama_instruct_pinned]
methods = [Direct, CoT, SelfConsistency, CGGR-confidence, CGGR-disagreement, CGGR-random, CGGR-oracle, CGGR-full]
seeds = [11, 23, 37, 53, 71]
```

Total required cells: `4 datasets x 2 models x 8 methods x 5 seeds = 320 cells`, before ablations. If resources cannot cover this, Contract Architect must shrink the claim, not silently shrink the evidence.

### Required Ablations

- `no_confidence`: remove confidence signal from CGGR-full.
- `no_disagreement`: remove disagreement signal from CGGR-full.
- `no_graph_or_cggr_signal`: replace graph/CGGR signal with plain text or no graph signal, depending on method design.
- `no_verifier`: remove final verification/reranking stage if present.
- `no_budget_model`: use fixed deliberation threshold without expected utility/cost model.
- `budget_sweep`: deliberation target rates at 10%, 25%, 50%, 100%.
- `sample_count_sweep`: SelfConsistency and disagreement k values, e.g. 3, 5, 10.
- `threshold_transfer`: tune threshold on one validation set and evaluate transfer to another dataset/model pair if claiming robustness.
- `prompt_sensitivity`: at least one alternate wording for Direct/CoT on validation only; main test prompt must be locked before test execution.

## Paper-Ready Artifact Checklist

- Frozen `benchmark_contract.yaml` with dataset/model/method/seed/metric/budget.
- `run_matrix.csv` with every required, optional, ablation, smoke, and diagnostic cell marked.
- `claim_register.md` listing claims and required evidence cells before results are known.
- Dataset manifests with sample ids, split names, checksums, license notes, and preprocessing commands.
- Baseline and method prompt packs with prompt hashes.
- Model cards: exact checkpoint id/revision, tokenizer, quantization, context window, serving backend.
- Environment snapshot: OS, Python, CUDA, driver, GPU, container/conda lock, package lock or archive hash.
- Raw per-example generation JSONL for every run cell.
- Per-example controller decisions for every selective deliberation method.
- Parser/normalizer logs including invalid and unparseable outputs.
- Retry and failure logs with failure taxonomy.
- Metrics by cell and aggregate tables with n, mean, std, 95% CI, failure rate.
- Paired significance report and correction policy for multiple comparisons.
- Budget audit: tokens, latency, GPU hours, deliberation rate, cost per correct answer.
- Ablation tables and figures tied to specific run ids.
- Oracle analysis clearly labeled as upper bound.
- Negative results and missing cells report.
- Smoke/sanity report clearly labeled infrastructure-only.
- Claim evidence map linking every manuscript sentence-level claim to table/figure/run ids.
- Reproducibility statement with exact command template and artifact index.
- Manuscript-ready results, limitations, and ethics/reproducibility sections.

## Acceptance Gate

A benchmark can support a paper claim only when all of the following are true:

1. The relevant run cells were declared in the frozen contract before execution.
2. The dataset split, model revision, prompt hash, method config, seed, and metric are recoverable from artifacts.
3. Raw generations and per-example scores exist for every included aggregate.
4. Parse failures, timeouts, OOMs, and remote failures are counted or explicitly audited.
5. Statistical uncertainty and cost are reported next to quality metrics.
6. The manuscript claim is present in `claim_evidence_map.md`.
7. The evidence is not from smoke/sanity runs.

If any condition fails, the Manuscript Writer must downgrade the claim to an observation, engineering note, or future-work statement.
