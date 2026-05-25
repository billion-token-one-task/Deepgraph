# ACL Rolling Review 专项要求（ACL / NAACL / EACL）

**模版**：`acl.sty` · **双栏** · long **8 页 + unlimited references** · `acl_natbib`

---

## 页预算

| 章节 | 词数 | 说明 |
|------|------|------|
| Abstract | 140–240 | ≤200 词意识（ARR 审稿敏感） |
| Introduction | 450–850 | 含 Contributions |
| Related Work | 280–900 | 2–4 主题子节 |
| Method | 400–1200 | 只写改动部分 |
| Experiments | 300–900 | 主表 + 协议 |
| Conclusion + Limitations | 150–280 | Limitations **建议 1 段** |

---

## 标题

- 任务 + 方法；多语写 **语言范围**  
- 6–9 词常见  

---

## Abstract（NLP 四拍）

Background → Gap → Approach（数据/模型/训练）→ Results（**BLEU/F1/Acc** 等 + 数据集名）

---

## Introduction

- **P6 Contributions 必填**（可含 *We release* 数据/代码）  
- P1 明确 **输入输出**（序列、标签、评价）  
- 宏观引用：LM、指令微调、该任务经典数据集论文  

---

## Related Work（NLP 主题簇示例）

按任务选 **2–4** 个：
1. *Pre-trained Language Models and Adaptation*  
2. *Task-Specific Models and Prompting*  
3. *Datasets and Evaluation Protocols*  
4. *Efficiency and Compression*（若相关）

每节：范式 → 代表工作 → 局限 → bridge。**禁止** 与 Method 小节同名。

---

## Method

1. Task formalization（符号、序列长度、标签空间）  
2. Architecture changes（**仅改动**）  
3. Training objective / data  
4. Inference & decoding  

标准 Transformer/BERT 类：**1 句 + 引用**，不复述全文。

---

## Experiments

- **Table 1**：多数据集 × 强基线（同规模 LM、leaderboard 系统）  
- 人工评测：评判人数、指标、协议  
- 显著性：bootstrap / paired test（若主会对比）  
- Error analysis：**至少 1 段或 1 表**  

---

## 行文

- 方法 **现在时**；实验 **过去时**  
- 术语与 Anthology 一致（token, lemma, F1, BLEU）  

---

## 反模式

- 摘要 ~100 词  
- Intro 无 itemize  
- RW 185 词单段（DeepGraph 典型）  
