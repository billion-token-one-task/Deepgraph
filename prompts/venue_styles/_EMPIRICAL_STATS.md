# 参考论文库实测篇幅（自动生成，勿手填）

- **语料**：`/root/Deepgraph-main/workspace/pdfs`，共 **565** 篇 PDF
- **抽样**：n=200（seed=42）
- **方法**：PyMuPDF 前 12 页；按 Abstract / Introduction / Related Work / Method / Experiments / Conclusion 分段
- **行数**：按排版换行估算（**不是** `.tex` 源码行数）
- **复现**：`python scripts/profile_reference_corpus_sections.py --write-stats`

## 标题（PDF 首页首行）

- 词数 median **6**，p25–p75 **5–8**（n=200）

## 主文各节词数与单栏行数（median, p25–p75）

| 章节 | 词数 | 单栏行数 @85 字符/行 | 样本数 |
|------|------|----------------------|--------|
| Abstract | **183** (139–245) | 22 (16–31) | 143 |
| Introduction | **659** (367–877) | 101 (57–148) | 131 |
| Related Work | **540** (282–1327) | 93 (40–233) | 99 |
| Method | **862** (333–1998) | 183 (58–511) | 169 |
| Experiments | **494** (255–1018) | 90 (45–192) | 137 |
| Conclusion | **154** (113–214) | 23 (17–31) | 94 |

## 结构信号（同批样本）

- Introduction 含 contribution / we propose 类表述比例：**75%**
- Related Work 呈多簇/多子主题结构比例：**12%**

## 与 DeepGraph 当前草稿差距（典型问题，非语料）

| 章节 | 参考 median | idea_2 草稿约 |
|------|-------------|---------------|
| Abstract | ~183 词 / ~22 行 | ~105 词 |
| Introduction | ~659 词 / ~101 行 | ~198 词，**无 Contributions itemize** |
| Related Work | ~540 词 / ~93 行 | ~185 词，无主题子节 |
| Method | ~862 词 | ~282 词，子节堆砌无 Overview |
| Experiments | ~473 词 | ~120 词，**无主结果表** |

## LaTeX 写作注意

- `\begin{abstract}` 在源码中常为 **1 段**；目标 **140–250 词**，编译后单栏约 **16–31 行**。
- 禁止按源码 1 行判断摘要长度。
