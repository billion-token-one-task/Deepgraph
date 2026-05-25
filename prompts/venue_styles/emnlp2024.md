# EMNLP 2024 专项要求

**模版**：`acl.sty`（`template_id=emnlp2024`）· **双栏** · **8 页 + unlimited refs** · 双盲 review

通用 NLP 框架见 `_SECTION_WRITING_FRAMEWORK.md`；本节为 EMNLP 主会差异。

---

## 页预算

与 ACL 相同（词数目标见上）；**以词数为准**。

---

## 标题

- *X for Y* / *Improving Z with W*  
- 多语/低资源：**语言数量** 进标题或首段  

---

## Abstract

- **140–240 词**（勿 <130）  
- 含 **具体 benchmark 名 + 主指标**  
- 可点明语言学动机（歧义、形态、语序等）1 句  

---

## Introduction

- **Contributions 3–5 条**（EMNLP 审稿必扫）  
- P2 强调：**数据偏见、标注噪声、域外、成本** 至少其一  
- P5 若有 **human eval**，在此预告协议关键词  

---

## Related Work

按 **研究问题** 分，非模型品牌：
- *Neural Methods for [Task]*  
- *Multilingual / Low-Resource Extensions*  
- *Human Evaluation and Annotation*（若有人评）  
- *Linguistic Analysis of …*（若做分析实验）

每节末 gap 句 + bridge。

---

## Method

- Overview figure（数据流/系统）  
- 多语：必须写 **language list** + 数据许可  
- 效率：latency / throughput / memory（系统向）  

---

## Experiments

- **Table-first**  
- 定性样例：**选例标准** 写清（勿只贴图）  
- Error analysis subsection 强烈推荐  
- 与 ACL 相比更常出现 **linguistic analysis** 表/图  

---

## Limitations

主文 **1 段 5–8 句**：失败语言、偏见、标注局限、外推边界。

---

## 反模式

- 技术报告口吻（罗列 artifact 路径）  
- 各节词数仅达 idea_2 水平（见 stats 对比表）  
