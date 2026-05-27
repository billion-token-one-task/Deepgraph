# 顶会实验表格排版规范（Experiment Table Requirements）

> 所有生成或修订 **Experiments 节 LaTeX 表格** 的 agent（section writing、publication tables、stats audit、benchmark orchestra）必须遵守。  
> 目标：高级、干净、专业——**数据为王**，版面克制，定性解释留在正文。

---

## 1. 数据为王，极致压缩文字（Data-Driven & Concise）

- **禁止长句列**：表格内不得出现 Interpretation、Main risk、Notes 等长段落列。定性解释全部移至 Results 正文段落。
- **优先衍生指标列**，用数值替代文字描述：
  - 置信区间：`95\% CI` 或 `$\pm$std`（多 seed 时）
  - 绝对差值：`$\Delta$`（相对 full model 或 best baseline）
  - 相对比例：`Rel.`、`$\Delta\%$`
  - 离散程度：`Range`、IQR
- **每格只放数字或极短 token**（方法名、数据集缩写、$\checkmark$/--）。
- 数字必须来自 `experimental_log.md` / `result_packet`；禁止手填或美化。

### 禁止 vs 推荐

| 禁止 | 推荐 |
|------|------|
| `Interpretation: model fails on long contexts` | 正文写一句；表内只留 `Acc.` 与 `$\Delta$` |
| `Main risk: distribution shift` | 移至 Limitations |
| 单列 Yes/No 长描述 | $\checkmark$ / `--` 或 1/0 |

---

## 2. 短表头与符号化（Compact Headers & Symbolization）

- **缩写表头**：`Accuracy` → `Acc.`；`Method by Backend` → `M$\times$B`；`F1 score` → `F1`。
- **布尔值符号化**：禁止 `Yes`/`No`、`True`/`False`；使用 `\checkmark` 与 `--`（需 `\usepackage{amssymb}` 或 `\usepackage{pifont}` + `\ding{51}`）。
- **学术符号代称**：如 Backend Variance Share → `$\eta_{\mathrm{B}}$`；Latency → `$T$ (ms)`。
- **方向箭头**：越大越好用 `$\uparrow$`，越小越好用 `$\downarrow$`，写在表头而非每格重复。

示例表头行：

```latex
Method & Acc.$\uparrow$ & F1$\uparrow$ & $\Delta$ & 95\% CI \\
```

---

## 3. 极简线条结构（Minimalist Structure）

- **标准三线表**（`booktabs`）：
  - `\toprule` — 顶粗线
  - `\midrule` — 表头下细线
  - `\bottomrule` — 底粗线
- **禁止竖线**：不得使用 `|` 列分隔符或 `\hline` 网格；靠列间距与对齐区分。
- **禁止 `\hline` 滥用**：除 booktabs 三线外不加横线。

最小模板：

```latex
\begin{table}[t]
\centering
\caption{Main results on benchmark suites.}
\label{tab:main_results}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lccccc}
\toprule
Method & Dataset A & Dataset B & Avg. & $\Delta$ & 95\% CI \\
\midrule
Baseline & 0.712 & 0.688 & 0.700 & -- & [0.68, 0.72] \\
\rowcolor{gray!10}
\textbf{Ours (full)} & 0.741 & 0.719 & 0.730 & +0.030 & [0.71, 0.75] \\
\bottomrule
\end{tabular}
\end{table}
```

跨栏大表用 `table*` + `tabularx`：

```latex
\begin{table*}[t]
\centering
\renewcommand{\arraystretch}{1.05}
\begin{tabularx}{\textwidth}{l@{\extracolsep{\fill}}cccc}
\toprule
...
\bottomrule
\end{tabularx}
\end{table*}
```

需 `\usepackage{booktabs}`、`\usepackage{tabularx}`、`\usepackage[table]{xcolor}`（若用 `\rowcolor`）。

---

## 4. 克制且高级的视觉引导（Subtle Visual Cues）

- **不要全表加粗最佳/最差**：避免每个数据集列都 `\textbf{}`，造成视觉疲劳。
- **整行浅底色**（推荐）：
  - 汇总行 / 主方法行：`\rowcolor{gray!10}` 或 `\rowcolor{red!8}`
  - 仅对 **1–2 行** 使用，不要多行彩虹底色
- **次选**：仅在 **汇总列**（Avg. / Overall）对最佳值加粗，数据列保持常规字重。
- **禁止**：彩色字体、多重边框、阴影、emoji。

---

## 5. 空间与版面的极致利用（Space Efficiency）

- **宽度自适应**：单栏 `\begin{tabularx}{\linewidth}`；双栏跨栏 `\begin{tabularx}{\textwidth}` + `@{\extracolsep{\fill}}`。
- **行高微调**：`\renewcommand{\arraystretch}{1.05}`（密集数字表 1.05–1.10）。
- **小数位对齐**：同列统一小数位数；用 `siunitx` 的 `S` 列（可选）对齐数量级。
- **内容降维（断舍离）**：
  - 纯环境配置、超参列表、硬件清单 → **不要表格**；压缩为 Experimental Setup 一段文字或附录 bullet list。
  - 表格版面留给 **有对比价值** 的主结果、消融、效率、显著性。

---

## 6. 与 DeepGraph pipeline 的对接

| 表格 | label | 来源 |
|------|-------|------|
| Main Results | `tab:main_results` | `build_publication_main_results_table_tex` / `experimental_log.md` |
| Ablations | `tab:ablations` | `build_publication_ablation_table_tex` / contract ablations |
| 显著性 / CI | （可选 `tab:significance`） | stats audit / seed 聚合 |

- Setup 节：Dataset | Split | Metric | #Seeds 可 **小表或段落**，不必占满页。
- 正文：**先 `\cref{tab:main_results}` 再解释**；表内不放 narrative。
- 若仅 smoke / proxy / `data_fraction < 1`：表标题或表注标明 **preliminary**，不得伪装 full benchmark。

---

## 7. Agent 自检清单

写作或替换表格后确认：

- [ ] 无竖线、无 `\hline` 网格，仅 booktabs 三线
- [ ] 无 Interpretation / risk / 长 Notes 列
- [ ] 表头 abbreviated；布尔用 $\checkmark$ / --
- [ ] 含 $\Delta$ / CI / Rel. 等衍生列（若日志提供）
- [ ] 主方法或 Aggregate 行最多 1–2 行 `\rowcolor{...}`
- [ ] 未对全部 best 值无脑 `\textbf{}`
- [ ] 跨栏表撑满 `\textwidth`；`\arraystretch` 已微调
- [ ] 配置/超参表已降级为正文，核心对比表保留
- [ ] 所有数字可追溯到 `experimental_log.md`

---

## 8. 反模式（对照历史草稿）

| 反模式 | 修复 |
|--------|------|
| 5 列表格 3 列是文字解释 | 删列，解释移正文 |
| `\hline` 包网格 | 改 booktabs |
| Yes/No 列 | 改 $\checkmark$ / -- |
| 每个 best 都加粗 | 改 rowcolor 或仅汇总列加粗 |
| 超参大表占半页 | 改 Setup 段落 |
| 表与日志数字不一致 | 以日志为准，禁止手改 |
