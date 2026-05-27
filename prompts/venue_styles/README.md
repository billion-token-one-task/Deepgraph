# Venue 写作需求文档

Pipeline 通过 `agents.manuscript_templates.style_guides.load_venue_style_guide(template_id)` 注入 **三份合并文档**：

1. **`_EMPIRICAL_STATS.md`** — `workspace/pdfs` 自动统计（勿手改数字；用脚本更新）  
2. **`_SECTION_WRITING_FRAMEWORK.md`** — 各节段落骨架、Intro/RW 分工、反模式  
3. **`{template_id}.md`** — 会议版式与页预算差异  
4. **`_MANUSCRIPT_QUALITY_GATES.md`** — 证据对齐、统计校准、消融一致性、反模式（`build_conference_guidelines` + `DEEPGRAPH_WRITING_GUARD` 加载）
5. **`../experiment_table_requirements.md`** — 顶会实验表格排版（三线表、短表头、rowcolor、数据列为主）

```bash
python scripts/profile_reference_corpus_sections.py --sample 200 --write-stats
```

| `template_id` | 版式 | 主文页 |
|---------------|------|--------|
| `iclr2026` | 单栏 | 9 |
| `neurips2024` | 单栏 | 9 |
| `icml2024` | 双栏 | 8 |
| `acl_arr` | 双栏 | 8 |
| `emnlp2024` | 双栏 | 8 |
| `cvpr2024` | 双栏 | 8 |
| `arxiv_plain` | 单栏 | 建议 12–20 |

**关键**：摘要按 **词数 ~200 / PDF ~20 行（单栏）** 写，不是 `.tex` 源码行数。
