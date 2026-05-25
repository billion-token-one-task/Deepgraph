# NeurIPS 2024 专项要求

**模版**：`neurips_2024.sty` · **单栏** · 主文 **9 页** · **双盲**

篇幅目标与 ICLR 单栏相同（见 `_EMPIRICAL_STATS.md`）。

---

## 与 ICLR 的差异

| 维度 | NeurIPS 强调 |
|------|----------------|
| 统计 | **多 seed** mean±std、显著性检验写清 |
| 理论 | Theorem / Proposition 更常见；主文 proof sketch |
| 负结果 | 简短 failure analysis 可加分，勿隐瞒 |
| 伦理/局限 | Checklist 内容压缩进 Conclusion 末段（1 短段） |

---

## Abstract

- 词数 **140–250**；单栏 **~20 行**  
- 理论文：写清 **假设 + bound 量级**（Big-O 或常数范围）  
- 实证文：与 ICLR 相同四拍，**数字优先**  

---

## Introduction

- Contributions **3–5 条**；可含 “We prove …” / “We establish …”  
- P3 三类 prior 可含 **理论线 + 算法线 + 系统线**  
- 宏观引用含经典理论/优化文献（若相关）  

---

## Related Work

- 理论类增加 `\subsection{Prior Theoretical Results}`  
- 实证类按 **learning paradigm** 或 **benchmark family** 分  
- 标注 **concurrent work**（同年 arXiv）1 句即可，勿占半页  

---

## Method

- 算法用 `algorithm` 环境或编号步骤  
- 假设写进 Problem Formulation（可列 Assumption 1..K）  

---

## Experiments

- 主表 + 消融 +（推荐）compute-sample tradeoff 图  
- 写清 **train/val/test** 与 **hyperparameter selection** 协议  
- 避免仅报 single-seed 最佳值  

---

## 反模式

- 忽略方差/seed  
- 摘要过短、Intro 无 contributions  
