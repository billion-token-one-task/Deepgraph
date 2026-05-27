# 顶会论文各节写作框架（基于参考库 + PaperOrchestra 范例）

> 数字目标见 `_EMPIRICAL_STATS.md`（`workspace/pdfs` 抽样实测）。  
> 结构范例见 `third_party/paperorchestra_arxiv2604/content/{intro,related_work,method,experiments}.tex`。  
> Outline 分工见 `prompts/paper_orchestra/outline_agent.tex`（Intro 宏观 / RW 微观，禁止重叠检索）。

---

## 0. 全文逻辑链（审稿人第一遍阅读路径）

**问题 → 为何重要 → 现有不足 → 本文做法 → 证据 → 贡献清单**

每一节只完成一个任务；禁止在 Abstract 写 Related Work，禁止在 Introduction 复刻 Method 公式，禁止在 Related Work 贴实验数字。

---

## 1. 标题（Title）

**实测**：PDF 首页标题 median **6 词**（p25–p75：**5–8**）。

| 推荐 | 避免 |
|------|------|
| `方法名: 任务或收益` | 超过 10 词、多个冒号 |
| 可含 1 个标准 benchmark 名 | Towards / Rethinking（非 position 文） |
| 标题动词与摘要主结果一致 | 未在正文定义的缩写作主标题 |

---

## 2. Abstract

### 篇幅（硬目标）
| 指标 | 目标 |
|------|------|
| 英文词数 | **140–250**（median≈**183**；勿写 <130） |
| 单栏排版行数 | **16–31**（median≈**22**） |
| LaTeX 形式 | 通常 **一段** `abstract` 环境，不是多段 `\section` |

### 四拍叙事（顺序固定，可合成一段）
1. **Context（1–2 句）**：任务/应用 + 为何社区关心  
2. **Gap（1 句）**：现有做法的具体失败模式（可量化）  
3. **Method（2–3 句）**：我们提出 X；核心机制；与 prior 对比点（一句话）  
4. **Evidence（2–3 句）**：数据集 + 指标 + **绝对值或相对提升** + 设置关键词（few-shot / zero-shot / N seeds）

### 禁止
- 引用 `\cite`（多数 venue 摘要无引用）  
- Related Work 综述、方法公式、证明  
- “Extensive experiments show” 而无数字  
- 声称 SOTA 但实验节无表格支撑  

---

## 3. Introduction

### 篇幅（硬目标）
| 指标 | 目标 |
|------|------|
| 词数 | **400–900**（median≈**659**；低于 350 视为不合格） |
| 单栏行数 | **57–148**（median≈**101**，约 1.5–2.5 页） |
| 引用 | **宏观** 10–20 篇：综述、奠基工作、问题重要性（**不要**堆 SOTA 对比细节） |

### 段落骨架（按顺序，不可打乱）

**P1 — Hook（3–5 句）**  
领域趋势 + 任务定义（输入/输出/评价指标）。避免空泛 “X is important”。

**P2 — Pain point（4–6 句）**  
具体失败案例：规模、成本、错误类型、域外场景。可用 1 个引用支撑 “问题真实存在”。

**P3 — Why prior work is insufficient（4–6 句）**  
归纳 **2–3 类** 路线及其局限（高层，不展开公式）。与 Related Work 区别：此处不逐篇点评。

**P4 — Our approach（4–6 句）**  
方法直觉 + 与 P3 各类的对应关系；指向 **Figure 1 / motivation figure**（若有）。仍不写定理/完整算法。

**P5 — Empirical preview（3–5 句）**  
最强结果一句 + 数据集 + 主要对比对象；语气克制，数字与实验表一致。

**P6 — Contributions（强制，不可省略）**

```latex
\paragraph{Contributions.}
This paper makes the following contributions:
\begin{itemize}
  \item \textbf{...}  % 动词开头：We propose / We show / We release
  \item ...
\end{itemize}
```

- **3–5 条**，每条 **可独立验证**（对应 Method 一节或 Experiments 一张表/图）  
- 第 1 条：方法/框架；第 2 条：理论或算法性质（若有）；第 3 条：实验发现；第 4 条：资源/基准（仅当真开源）  
- 参考 PaperOrchestra intro：末段 itemize 明确列出框架、基准、人工评测结论  

### Introduction 禁止
- 无 `\begin{itemize}` 贡献列表（参考库 **~75%** 文稿在 intro 有 contribution 信号）  
- 把 Related Work 写成第二篇综述  
- 在贡献条中写 “we conduct comprehensive experiments” 而无具体 claim  

---

## 4. Related Work

### 篇幅（硬目标）
| 指标 | 目标 |
|------|------|
| 词数 | **280–1300**（median≈**540**；低于 250 不合格） |
| 单栏行数 | **40–230**（median≈**93**，约 0.75–1.5 页） |
| 结构 | **2–4 个** `\subsection` 主题簇（禁止 1 段到底） |

### 与 Introduction 的分工（硬性）
| | Introduction | Related Work |
|---|--------------|--------------|
| 检索层级 | 宏观：综述、奠基、问题影响 | 微观：SOTA、直接竞品、同 benchmark |
| 引用数量 | ~10–20 | ~30–50（全文 bib 的一部分） |
| 内容 | 为何做 | 谁做过、差在哪、本文站位 |

### 每个 `\subsection` 模板（5 步，每簇 1–2 段）
1. **定义范式**（2–3 句）：这类方法解决什么子问题  
2. **代表工作**（4–8 句）：3–6 篇 `\cite`，每篇 **一句** 具体贡献（读 abstract 再写）  
3. **局限**（2–4 句）：基于证据的 gap（与本文实验可对照）  
4. **Bridge**（1–2 句）：本文如何填补（**不写** 本文方法公式）  
5. （可选）与上一簇的 **对比句** 防重复  

### 主题簇划分原则（按「范式」不按「时间」）
- ✅ autoregressive vs diffusion vs hybrid；efficient attention vs full attention  
- ✅ PLM adaptation vs task-specific heads vs prompting  
- ❌ “2020–2023 papers” / 按作者字母  
- ❌ 与 Method 小节 **同名** 且内容重复  

### PaperOrchestra 范例
- `\subsection{AI Researcher Frameworks}` + `\subsection{Automated Writing and Literature Synthesis}`  
- 末段用对比表 + 一段 “In contrast, our method …” 收束  

### 禁止
- 无 subsection 的流水账  
- 逐篇摘要粘贴（每篇 >2 句）  
- 在 RW 放本文实验数字或主结果表  

---

## 5. Method

### 篇幅（硬目标）
| 指标 | 目标 |
|------|------|
| 词数 | **330–2000**（median≈**862**；低于 300 不合格） |
| 单栏行数 | **58–500**（median≈**183**，约 1.5–3 页） |

### 推荐 `\subsection` 顺序（实证 ML 通用）

1. **Problem Formulation / Preliminaries**  
   - 符号表：输入、输出、损失、假设  
   - 任务形式化 1 段 + 必要定义 1 段  

2. **Overview**（必须有）  
   - **Figure 1**：pipeline / 系统图；正文 **逐模块** 对应图中编号  
   - 1 段直觉 + 1 段步骤列表（可用 `\paragraph{Step k.}`，见 PaperOrchestra）  

3. **Core Component(s)**（1–3 个子节）  
   - 每个组件：**直觉 → 公式/算法 → 复杂度**  
   - 只写 **相对 baseline 的改动**；标准 Transformer 等用 1 句 + 引用  

4. **Training / Inference**（若与架构分离）  
   - 目标函数、优化、伪代码或算法环境  

5. **Implementation Details**（短，或并入 Experiments Setup）  

### 两种合法组织（二选一，勿混用）

| 类型 | 结构 | 范例 |
|------|------|------|
| **系统/框架文** | Overview + `\paragraph{Step 1..K}` | PaperOrchestra Method |
| **模型/理论文** | Overview + `\subsection{Component A/B}` + Theorem | 经典 ICLR 模型文 |

### 禁止（DeepGraph 常犯）
- 无 Overview 图，直接 `Training Data Construction` 等孤儿子节  
- Method 子节标题与 RW 子节 **一一对应** 且复述  
- 编造未在 `idea.md` / 实验日志出现的符号  

---

## 6. Experiments

### 篇幅（硬目标）
| 指标 | 目标 |
|------|------|
| 词数 | **250–1000**（median≈**473**；低于 200 不合格） |
| 单栏行数 | **45–190**（median≈**89**，约 1.5–2.5 页） |

### 推荐 `\subsection` 顺序

1. **Experimental Setup / Benchmarks**  
   - 表格：Dataset | Split | Metric | Baselines | Implementation  
   - 写清 seeds、算力、超参来源（可附录）  

2. **Main Results**（**必须有 Table 1**）  
   - 多数据集 × 多方法；±std / 95% CI / $\Delta$ 若有多 seed  
   - **排版**：见 `prompts/experiment_table_requirements.md`（三线表、短表头、$\checkmark$/--、rowcolor 高亮主方法行、禁止竖线与 Interpretation 列）  
   - 正文 **先引表再解释**；图仅补充趋势/案例  

3. **Analysis / Ablations**  
   - 每个 contribution 至少 **一行消融或一张图**  
   - 失败案例 / 误差分析（short subsection 或 paragraph）  

4. **（可选）Human Evaluation / Efficiency**  
   - 协议、评判人数、指标  

### PaperOrchestra 范例结构
`Baselines` → `Autoraters`（若适用）→ `Results` → `Human Evaluation` → `Ablation Studies`

### 禁止
- 只有 `figure` 柱状图、无 `table`  
- 实验节 <150 词  
- 数字与 `experimental_log.md` 不一致  
- 把 bootstrap/smoke test 写成 full benchmark  

---

## 7. Conclusion

| 指标 | 目标 |
|------|------|
| 词数 | **110–220**（median≈**154**） |
| 内容 | 1 段总结问题+方法+结果；1 段局限+未来工作（可并） |

勿引入新实验、新引用、新贡献。

---

## 8. 全局反模式（对照 DeepGraph idea_2 / idea_7 草稿）

| 问题 | 表现 | 修复 |
|------|------|------|
| 全文过短 | 各节词数远低于 median | 按上表扩写至 p25 以上 |
| 无 Contributions | Intro 只有 narrative | 强制 P6 itemize |
| RW 乱切 | 单段、185 词 | 2–4 个主题 subsection |
| Method 乱切 | 无 Overview，子节像代码模块 | Overview 图 + 组件子节 |
| 实验像报告 | 120 词 + 单图 | Setup 表 + Main Results 表 + 消融 |
| **任务错位** | 视频/时间叙事 + GSM8K 文本实验 | 按日志改标题/摘要/问题定义 |
| **统计吹胀** | 摘要写 +65% 但 p=1.0 | 改 preliminary；Limitations 写明 inconclusive |
| **消融打脸** | Allen/几何是贡献，ablated 更好 | 贡献改写成 ablation 支持的组件 |
| **段落重复** | Introduction 同段出现两次 | 删重复，保留更强一版 |
| **无关 RW** | 硬塞 routing/gating 综述 | 仅当实验含 routing 主结果 |

详见 `_MANUSCRIPT_QUALITY_GATES.md`（写作 agent 硬性加载）。

---

## 9. Outline / Section Agent 对齐检查清单

写作完成后自检：

- [ ] Abstract 140–250 词，含具体数字  
- [ ] Intro 400+ 词，末段 Contributions itemize  
- [ ] Related Work 250+ 词，2–4 个主题 subsection，末段 bridge  
- [ ] Method 有 Overview（图或段），子节 ≥2  
- [ ] Experiments 有主结果 **表**，Setup 可复现  
- [ ] 全文 cite 密度：Intro 宏观 + RW 微观，不重复堆砌同一批仅 RW 才需要的文  
