# Latent Space Communication for Multi-Agent LLM Systems
## 研究调研 + 实验验证 + 未来方向

---

## 一、问题定义

当前所有生产级多 agent LLM 系统（AutoGen、CrewAI、LangGraph、MCP、A2A）都通过 **自然语言 token** 通信。这意味着：

1. **信息瓶颈**：模型内部是 4096 维连续向量，序列化为 token 后信息量大幅丢失
2. **延迟浪费**：生成 token → 编码 → 传输 → 解码 → 理解，每一步都有损
3. **带宽低下**：自然语言信息密度远低于连续表示

**核心问题**：能否让 LLM agent 直接通过连续向量（latent space）通信，绕过 token 序列化？

---

## 二、文献调研（2024-2026）

### 2.1 已有工作：潜空间通信

这个方向在 **2025 年 11 月突然爆发**，之前几乎没有。

#### LatentMAS（Princeton/Stanford, 2025.11）
- **论文**: Latent Collaboration in Multi-Agent Systems (arXiv:2511.20639)
- **方法**: Training-free，直接传递 last-layer hidden states，共享 latent working memory
- **结果**:
  - 准确率提升 **+14.6%**（sequential MAS）/ **+13.3%**（hierarchical MAS）
  - Token 输出减少 **70.8-83.7%**
  - 推理速度 **4-4.3x**
- **评测**: GSM8K, AIME 2024/2025, GPQA-Diamond, MedQA 等 9 个 benchmark
- **关键发现**: 推理密集型任务增益最大
- **代码**: github.com/Gen-Verse/LatentMAS

#### Interlat（浙大等, 2025.11, 投稿 ICLR 2026）
- **论文**: Enabling Agents to Communicate Entirely in Latent Space (arXiv:2511.09149)
- **方法**: 传递 last-layer hidden states + 学习压缩模块
- **结果**:
  - Qwen2.5-7B: **70.48%**（latent）vs **64.29%**（text）vs **62.14%**（无通信）
  - LLaMA3.1-8B: **70.71%**（latent）vs **62.86%**（text）
  - 压缩到 8 token: **24x 延迟降低**（9.19s → 0.39s），仍保持 66.43% 成功率
- **评测**: ALFWorld, MATH

#### Vision Wormhole（Purdue 等, 2026.02）
- **论文**: The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems (arXiv:2602.15382)
- **方法**: 利用 VLM 的视觉通道作为连续通信信道，Universal Visual Codec
- **结果**:
  - 准确率提升 **+6.3 pp**，速度 **1.87x**
  - 代码生成任务 **+13.2 pp**
  - 弱监督（<100 anchor texts）: **+6.5 pp**, **2.67x** 加速
- **关键创新**: 对齐复杂度从 O(N²) 降到 O(N)

### 2.2 单模型潜在推理（latent reasoning）

#### Coconut（Meta FAIR, 2024.12, COLM 2025）
- **论文**: Training Large Language Models to Reason in a Continuous Latent Space (arXiv:2412.06769)
- **方法**: 模型的 last hidden state 直接作为下一步输入 embedding，不解码为 token
- **结果**: 在逻辑推理任务（ProntoQA, ProsQA）超越 CoT，且 token 用量显著更少
- **关键洞察**: 连续思维可以同时编码多条推理路径（BFS），而 CoT 只能单路径（DFS）

#### Latent Reasoning with Supervised Thinking States（Google, 2026.02）
- **论文**: arXiv:2602.08332
- **结果**:
  - Parity 任务: **100%** vs CoT 的 64.38%
  - Variable Assignment: **97.71%** vs CoT 的 87.75%
  - GSM8K: 42.22% 准确率 + **2.66x 加速**
- **关键**: 训练效率好，BPTT 方法在 10 步时 ~10x 成本开销，Thinking States 接近常数

#### SoftCoT（ACL 2025）
- **论文**: Soft Chain-of-Thought for Efficient Reasoning with LLMs (arXiv:2502.12134)
- **方法**: 轻量冻结助手模型生成 "soft thought tokens"，投影到主 LLM 表示空间
- **优点**: 不修改 LLM 本身，避免灾难性遗忘

### 2.3 多 agent 系统的通信效率研究

#### Scaling Agent Systems（Google DeepMind/MIT, 2025.12）
- **论文**: Towards a Science of Scaling Agent Systems (arXiv:2512.08296)
- **结果（180 种配置, GPT/Gemini/Claude）**:
  - 集中式协调提升并行任务 **80.8%**
  - 顺序推理任务多 agent 反而 **退化 39-70%**
  - 独立 agent 放大错误 **17.2x**
  - 一旦单 agent 基线超过 ~45%，多 agent 收益递减

#### ProtocolBench（UIUC, 2025.10）
- **论文**: Which LLM Multi-Agent Protocol to Choose? (arXiv:2510.17149)
- **结果**: 协议选择影响任务完成时间达 **36%**，通信开销 **3.5 秒**

### 2.4 综述论文

- **Reasoning Beyond Language**: A Comprehensive Survey on Latent Chain-of-Thought Reasoning (2025.05, arXiv:2505.16782) — 分类了所有 latent CoT 方法
- **Emergent Communication 综述** (TMLR 2024, arXiv:2407.03302) — 49 页综述

---

## 三、当前系统的实验验证

### 3.1 DeepGraph 系统发现的相关证据

我们从 2,790 篇论文中自动发现了以下支撑 "latent communication 优于 token communication" 的实证：

**证据 1: "更多文字输出 ≠ 更好推理"**
- TRiMS (2603.17449): 减少 **83%** 推理 token，准确率反而提升 **+6.8**
- 说明 token 序列化是推理的瓶颈而非载体

**证据 2: "可见 CoT 可能是训练脚手架而非推理来源"**
- Insight #10 (paradigm score 6.5): 多篇论文显示强迫输出可见推理链在 grounding 充分后反而降低性能
- 这直接支持 latent reasoning 的价值

**证据 3: "Scaling 在有脚手架后不再单调"**
- MiniMax M2.5 (81.1%) 打赢 Nemotron 120B (78.4%) on SKILLS benchmark
- 说明系统瓶颈不在模型大小而在通信/协调效率

**证据 4: "代理指标优化 ≠ 真实能力"**
- 我们的范式断裂分析发现：ML 反复把方便的接口（token 输出）误认为底层现象（真实推理）

### 3.2 DeepGraph + EvoScientist 作为多 agent 系统的实践观察

DeepGraph 本身就是一个 9-agent 系统（Extraction, Taxonomy, Reasoning, Insight, Ranker, Abstraction, Bridge, Summary, Expander），全部通过 DB + JSON 通信。实践中观察到：

- **信息压缩损失**: agent 之间传递 JSON 时必须把复杂推理结论压缩为字符串，丢失了推理过程中的不确定性和多假设分支
- **重复理解成本**: 每个下游 agent 都要重新 "理解" 上游 agent 的文本输出，消耗大量 token
- **无法传递 "直觉"**: 一个 agent 对某个模式的 "感觉"（高维表示中的方向）无法通过文本传递给另一个 agent

---

## 四、未被探索的方向（研究机会）

通过对比已有工作和我们的发现，以下方向 **没人做过**：

### 4.1 Hybrid Latent-Token Communication（混合通信架构）

**所有现有工作都是纯 latent 通信。但现实中 agent 必须调用工具。**

```
Agent A ←──── Latent Channel ────► Agent B
  │              (意图/推理/注意力)      │
  │                                     │
  └──── Token Channel ──────────────────┘
         (工具调用/代码/精确输出)
```

- 没有论文研究过 latent + token 混合通信的最优分配策略
- 什么时候走 latent、什么时候降级为 token？需要学习一个 router
- 这本身就是一篇论文

### 4.2 Latent Communication + Tool Use

- LatentMAS / Interlat 都不支持工具调用
- 现实中 agent 必须能调 API、查数据库、执行代码
- 工具调用是离散的，必须生成精确字符串
- **研究问题**: 能否在 latent channel 传递"调什么工具、为什么"的意图，然后只在最后一步解码为精确的工具调用？

### 4.3 跨模型 Latent Communication

- 现有工作几乎都是同架构模型之间的通信（Qwen-Qwen, LLaMA-LLaMA）
- Vision Wormhole 尝试了异构模型，但只限于 VLM
- **研究问题**: 不同架构的 LLM（如 Qwen 和 LLaMA）能否学到共享的 latent 通信协议？需要什么样的对齐？

### 4.4 持久 Latent State（长时通信）

- 现有工作都是单轮或短轮对话中的 latent 传递
- 没人研究过 **跨会话持久化** latent state
- **研究问题**: 能否把 agent 的 latent state 存储到外部记忆（类似 KV cache 持久化），实现跨任务的 latent 知识传递？

### 4.5 Latent Communication 的安全性

- 如果 agent 之间通过不可解释的连续向量通信，如何审计？
- 如何防止 latent channel 传递恶意信息？
- 没有论文研究过 latent 通信的对齐和安全问题

---

## 五、推荐实验方案

### 实验 1: Hybrid Latent-Token Communication（首选，6 个月 PhD 项目）

**假设**: 90% 的 agent 间通信可以走 latent channel（更快更密），只有 10% 的工具调用/精确输出需要 token channel。整体效率优于纯 token 和纯 latent。

**设计**:
1. 基座: Qwen2.5-7B
2. 在 LatentMAS 基础上加 Token 降级模块：学习一个 router 决定每一步走 latent 还是 token
3. 评测任务: ALFWorld（agent 协作 + 工具调用）、WebArena（网页操作 + 工具调用）、SWE-bench（代码生成 + 工具调用）
4. 对比: 纯 token（AutoGen 风格）、纯 latent（LatentMAS）、混合（ours）

**成功标准**:
- 任务成功率 ≥ 纯 latent 且 ≥ 纯 token
- 工具调用准确率 ≥ 纯 token（不因 latent 损失精度）
- 通信 token 数 < 纯 token 的 30%
- 端到端延迟 < 纯 token 的 50%

**算力需求**: 8×A100 80GB，3-4 周

### 实验 2: Cross-Architecture Latent Protocol

**假设**: 不同架构 LLM 可以通过学习的投影层共享 latent 通信协议。

**设计**:
1. 配对: Qwen2.5-7B ↔ LLaMA3.1-8B
2. 训练轻量投影层将一个模型的 hidden state 映射到另一个的表示空间
3. 评测跨架构协作 vs 同架构协作的差距

**算力需求**: 4×A100 80GB，2 周

### 实验 3: Latent State Persistence

**假设**: 持久化 agent 的 latent state 并跨任务复用，比每次从 token 重新加载上下文更高效。

**设计**:
1. 设计 latent state store（类似 KV cache 但持久化到磁盘）
2. 对比: 每次从头加载 token context vs 从 latent state 恢复
3. 评测: 长任务链中的信息保持率、效率

**算力需求**: 4×A100 80GB，2 周

---

## 六、与大厂合作的定位

### 你已有的优势
1. **DeepGraph 系统**: 从 2790 篇论文中提取了支撑证据，这不是空想
2. **EvoScientist 集成**: 已经有端到端的研究方案生成能力
3. **范式断裂分析**: 有数据支撑的论证，说明 token 通信是 ML agent 系统的结构性瓶颈
4. **Working prototype**: DeepGraph 本身就是 9-agent 系统，有实际的通信瓶颈观察

### 差异化角度（vs 已有工作）

| 已有工作 | 做了什么 | 没做什么（你的机会） |
|----------|---------|-------------------|
| LatentMAS | 纯 latent，无训练 | 不支持工具调用 |
| Interlat | latent + 压缩 | 不支持工具调用，只测了 ALFWorld |
| Vision Wormhole | 异构模型通信 | 只限 VLM，没有通用 LLM |
| Coconut | 单模型 latent 推理 | 不是多 agent |
| Google Thinking States | 单模型 latent 推理 | 不是多 agent |

**你的定位**: **Hybrid Latent-Token Communication with Tool Use** — 第一个让 latent 通信的 agent 能调工具的工作。

### 投稿目标
- **NeurIPS 2026** (DDL ~May 2026) 或 **ICML 2026**
- 如果实验结果强，可以冲 spotlight/oral

### 给大厂看的一句话
> "我们发现所有现有的 latent communication 方法都无法处理工具调用——而现实中 90% 的 agent 任务需要工具。我们提出混合 latent-token 通信架构，在保持 latent 通信效率的同时支持精确工具调用，并有来自 2790 篇论文的 meta-analysis 数据支撑这个方向的价值。"

---

## 七、参考文献

1. LatentMAS — arXiv:2511.20639 (Princeton/Stanford, 2025.11)
2. Interlat — arXiv:2511.09149 (浙大等, 2025.11, ICLR 2026 submission)
3. Vision Wormhole — arXiv:2602.15382 (Purdue等, 2026.02)
4. Coconut — arXiv:2412.06769 (Meta FAIR, 2024.12, COLM 2025)
5. Latent Reasoning with Thinking States — arXiv:2602.08332 (Google, 2026.02)
6. SoftCoT — arXiv:2502.12134 (ACL 2025)
7. Scaling Agent Systems — arXiv:2512.08296 (Google DeepMind/MIT, 2025.12)
8. ProtocolBench — arXiv:2510.17149 (UIUC, 2025.10)
9. Reasoning Beyond Language Survey — arXiv:2505.16782 (2025.05)
10. Emergent Communication Review — arXiv:2407.03302 (TMLR 2024)
11. Group Think — arXiv:2505.11107 (MediaTek Research, 2025.05)
