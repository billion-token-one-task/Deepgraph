# Fix Report: Debug-Startup-Resilience

## 背景

在提交 PR 前对项目进行检查时，发现两个会影响启动稳定性和测试可用性的核心问题：

1. 可选依赖问题：`tests/test_helpers.py` 仅调用 fallback 逻辑，但会因为 `agents/domain_summary_agent.py` 顶层导入 `httpx` 相关链路而直接报错。
2. 数据库初始化问题：当 `entity_merge_candidates` 表尚未创建时，候选查询函数会抛出 `sqlite3.OperationalError`，导致流程中断。

## 根因分析

- `agents/domain_summary_agent.py` 在模块加载阶段执行 `from agents.llm_client import call_llm_json`，这会强制要求完整 LLM 依赖存在，即使当前只需要 fallback 功能。
- `db/evidence_graph.py` 的候选查询函数默认假设 `entity_merge_candidates` 表一定存在，缺少 schema 缺失时的降级分支。

## 修复内容

### 1) 延迟导入 LLM 客户端

- 文件：`agents/domain_summary_agent.py`
- 改动：将 `call_llm_json` 的导入移动到 `generate_domain_summary()` 函数内部。
- 效果：fallback 路径不再被可选依赖阻塞，提升弱依赖环境下的可用性。

### 2) 增加 merge candidate 表存在性保护

- 文件：`db/evidence_graph.py`
- 改动：在以下函数中先检查 `sqlite_master` 是否存在 `entity_merge_candidates`：
  - `list_merge_candidates(...)`
  - `get_merge_candidate_context(...)`
- 降级行为：
  - 表不存在时，`list_merge_candidates` 返回 `[]`
  - 表不存在时，`get_merge_candidate_context` 返回 `None`

## 验证结果

- 执行命令：

```bash
python3 -m unittest discover -s tests
```

- 结果：
  - `Ran 17 tests`
  - `OK`

## 分支与提交

- 分支：`Debug-Startup-Resilience`
- 关键修复提交：`f04e216`  
  `Fix startup resilience for optional deps and schema state`
- 额外提交（本地开发配置）：`78a5b0e`  
  `Add VS Code Power Query symbol settings`

## PR 信息

- PR 链接：`https://github.com/koen666/Deepgraph/pull/1`

## 备注

- 若希望 PR 仅包含运行时修复（不含 `.vscode` 配置），建议后续拆分为独立 PR 或在提交历史中分离配置改动。
