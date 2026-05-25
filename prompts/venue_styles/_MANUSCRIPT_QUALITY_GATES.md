# DeepGraph 稿件质量硬约束（Manuscript Quality Gates）

> 所有 PaperOrchestra / manuscript 写作 agent 必须遵守。违反任一条须在 Limitations 中如实披露，且不得出现在 Abstract 主结论句。

---

## 1. 证据—叙事一致性（最高优先级）

- **任务对齐**：标题、摘要、Introduction 的问题定义必须与 `experimental_log.md` 中的 **实际数据集、模态、指标** 一致。
  - 若实验跑在 GSM8K / StrategyQA 等文本 QA，不得写成「视频时间定位 / caption matching / moment retrieval」主任务，除非日志中确有对应 benchmark。
  - 若 `codebase_url` 与任务模态不符（如视频仓库 + 文本 LLM 指标），正文必须说明 proxy/harness 边界，不得假装端到端在该仓库上完成主实验。
- **贡献对齐**：每条 Contribution bullet 必须对应 Method 一节或 Experiments 一张表/图；**禁止**贡献写 Allen 区间/几何约束，而消融表显示「去掉该组件反而更好」。
- **消融叙事**：若 `TIGN-base`（完整方法）劣于 `ablated`（去掉核心组件），正文必须：
  - 将主结论改为「哪一组件真正驱动结果」；
  - 不得把被 ablation 否定的机制写进摘要第一句或贡献第一条。

## 2. 统计与 claim 校准

- `p_value >= 0.05` 或 `hypothesis_verdict` 为 `inconclusive` / `refuted` 时：
  - **禁止**在 Abstract 写「显著提升 / outperform / SOTA / 验证假设」；
  - 只允许：preliminary、directional、trend、under review、requires further validation。
- 摘要中若出现相对提升百分比，必须同句附带：seed 数、数据集名、以及「统计检验是否显著」。
- 不得用极小 std（如 0.001）暗示严谨性，却不报告检验方法、多重比较校正或效应量置信区间。
- `claim_evidence_matrix.json` 中 `can_appear_in_abstract: false` 的 claim **不得**进入摘要。
- `current_evidence: missing` 的 claim 只能出现在 Limitations / Future Work。

## 3. 结构与写作卫生

- **禁止重复段落**：同一 section 内不得出现两段以上相同或高度相似的 `\\paragraph{...}` 开头块；写完后自检 Introduction 无 copy-paste。
- **禁止无关 Related Work 节**：不得为凑引用加入与本文任务无关的「Routing and Gating」综述，除非实验日志含 routing 主结果且 claim 矩阵允许。
- **禁止审计体口吻**：正文不得出现 "available logs", "supplied artifacts", "provided materials", "artifact manifest"。
- **图注诚实**：若某图/指标在 evidence 中缺失，图注必须写 "not computed in this run"，不得编造 calibration / significance。

## 4. 实验节最低标准

- Main Results 表中的数据集名必须与 `experimental_log.md` / `result_packet` 完全一致。
- 至少一张 **booktabs** 主结果表；数字只来自日志，禁止手填「好看」的数字。
- 若只有 sanity / proxy / `data_fraction < 1` 运行，全文必须标明 **preliminary**，不得写 full benchmark validation。
- 负结果与 refutation 必须写入 Results 或 Limitations，不得只写正面叙事。

## 5. 摘要四句话检查（写完后自检）

1. 问题是什么？（与日志数据集一致）
2. 方法一句话？（与消融后仍有效的组件一致）
3. 证据一句？（含绝对值 + 是否 significant）
4. 局限一句？（含 inconclusive / 缺失证据）

## 6. Content Refinement 终检

修订稿必须通过：

- [ ] 标题任务 = 实验任务
- [ ] 贡献列表 ⊆ 实验支持的机制
- [ ] 摘要无 unsupported superlatives
- [ ] 无重复 Introduction 段落
- [ ] 消融结论与 Method 主张不矛盾
- [ ] 所有表格数字可追溯到 experimental_log.md
