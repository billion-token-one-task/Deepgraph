# LLM Agent 潜空间通信：技术全景

## 一、当前三种方案对比

### 方案 A：Training-Free（LatentMAS）
- **传什么**：last-layer hidden state（4096维向量）
- **怎么传**：直接把 Agent A 的输出 hidden state 拼接到 Agent B 的输入 embedding 序列
- **不需要训练**，零成本即插即用
- **结果**：准确率 +14.6%，token 减少 70-84%，速度 4x
- **问题**：只能同架构模型；没有压缩，带宽高

### 方案 B：Learned Compression（Interlat）
- **传什么**：last-layer hidden state → 学习压缩模块 → 8 个 token 的 latent code
- **怎么传**：压缩后的 latent code 作为 prefix 注入 Agent B 的输入
- **需要训练压缩模块**（轻量，几百 M 参数）
- **结果**：24x 延迟降低（9.19s→0.39s），成功率仍 66.43%
- **问题**：压缩有损；仍是同架构

### 方案 C：Visual Channel Hijack（Vision Wormhole）
- **传什么**：推理 trace → Universal Visual Codec → 视觉 token
- **怎么传**：编码成视觉 token 注入 VLM 的图像输入通道
- **需要训练 Codec**
- **结果**：+6.3pp 准确率，1.87x 速度，支持异构模型
- **对齐**：O(N) 而非 O(N²)
- **问题**：只限 VLM

## 二、原生训练方法

| 方法 | 训练方式 | 核心 loss | 开销 |
|------|---------|----------|------|
| Coconut (Meta) | 把 last hidden state 直接回馈为下一步 input embedding | 标准 next-token loss，但 "token" 是连续向量 | 全参数微调 |
| Thinking States (Google) | 每隔 c=8 个 token 插入 thinking token，teacher-forcing 监督 | MSE on thinking state + CE on output | 接近常数训练时间 |
| SoftCoT (ACL 2025) | 冻结主 LLM，只训小助手模型的投影层 | 对齐 soft thought token 到主模型表示空间 | 极轻量 |
| Interlat 压缩模块 | 自监督：压缩→解压→重建原始 hidden state | 重建 loss + 任务 loss | 几百 M 参数 |

**结论**：不需要从头预训练。在冻结基座上训一个轻量头/压缩器就够了。

## 三、压缩方式

| 方法 | 输入维度 | 输出 | 压缩比 | 信息保留 |
|------|---------|------|--------|---------|
| 无压缩（LatentMAS） | 4096 × seq_len | 原样传递 | 1:1 | 100% |
| 线性投影 | 4096 | 256-1024 维 | 4-16x | 高 |
| Learned Bottleneck（Interlat） | 4096 × seq_len | 8 个 token embedding | ~100x | 66% 任务成功率 |
| VQ-VAE codebook | 4096 | 离散码 + codebook index | 极高 | 取决于 codebook 大小 |
| LoRA-style adapter | 全 hidden state | 低秩近似 | rank 决定 | rank 16-64 通常够 |

**结论**：压缩到 8-16 个 latent token 是当前甜点。再压信息损失陡增。

## 四、传递/注入方式

```
Agent A output hidden states
        │
        ├─► [方式1] Prefix Injection：拼到 Agent B 的 input embedding 前面
        │   简单直接，LatentMAS/Interlat 都用这个
        │
        ├─► [方式2] Cross-Attention：Agent B 加 cross-attention 层 attend to Agent A 的 states
        │   需要改架构，但信息利用更灵活
        │
        ├─► [方式3] 视觉通道注入：编码成视觉 token 走 VLM 的图像输入
        │   Vision Wormhole 用这个，巧妙但只限 VLM
        │
        └─► [方式4] Shared KV Cache：两个 agent 共享部分 KV cache
            Group Think 用这个（同模型内多线程），跨模型未验证
```

**结论**：Prefix Injection 最简单最实用，改动最小。

## 五、Tool Use（关键空白）

**现状：没有任何论文解决 latent communication + tool use。**

这是因为工具调用本质是离散的（函数名 + 参数字符串），连续向量无法直接表达。

**可行的混合方案**：

```
                Latent Channel
Agent A ◄───────────────────────► Agent B
   │                                  │
   │  连续向量：意图/推理状态/注意力    │
   │                                  │
   ▼                                  ▼
Router                            Router
   │                                  │
   ├─ latent够 → 继续latent          ├─ latent够 → 继续latent
   │                                  │
   └─ 需要精确输出 → 解码为token      └─ 需要精确输出 → 解码为token
          │                                  │
          ▼                                  ▼
    tool_call("search", {"q": "..."})   code: print("hello")
```

Router 的训练：
- 输入：当前 hidden state
- 输出：binary decision（latent / token）
- 训练信号：任务成功率 + 通信效率的联合奖励
- 类似 early-exit 的思路，但决定的是通信通道而非推理深度

## 六、评测现状

| Benchmark | 测了什么 | 谁用了 |
|-----------|---------|--------|
| ALFWorld | 多步规划 + 物体交互 | Interlat |
| GSM8K / MATH | 数学推理 | LatentMAS, Coconut |
| AIME 2024/2025 | 竞赛数学 | LatentMAS |
| GPQA-Diamond | 科学推理 | LatentMAS |
| HumanEval / MBPP | 代码生成 | LatentMAS, Vision Wormhole |
| MedQA | 医学问答 | LatentMAS |

**没人测过的**：WebArena（网页操作+工具）、SWE-bench（代码+工具）、任何需要 API 调用的任务。

## 七、已知限制

1. **同架构绑定**：LatentMAS 和 Interlat 都只在同架构模型间工作（Qwen-Qwen, LLaMA-LLaMA），Vision Wormhole 部分解决了异构问题
2. **无工具调用**：所有方案都在"纯推理"场景评测，没有工具/API/代码执行
3. **无持久化**：latent state 是一次性的，跨会话无法复用
4. **可解释性为零**：连续向量通信完全不可审计
5. **单轮为主**：多轮 latent 对话的累积误差没人研究过

## 八、开源实现

| 项目 | 地址 | 可复现度 |
|------|------|---------|
| LatentMAS | github.com/Gen-Verse/LatentMAS | 高，training-free |
| Coconut | github.com/facebookresearch/coconut | 高，Meta 官方 |
| Interlat | 论文有代码但未开源 | 低 |
| Vision Wormhole | 未开源 | 低 |

## 九、结论：这个领域的真实状态

**领域成熟度**：极早期。2025.11 才出现第一批论文，现在总共就 3 篇核心工作。

**已验证的**：
- latent 通信比 token 通信更快（4x）更省（70-84% token 减少）更准（+6-14%）
- 不需要从头训练，冻结基座 + 轻量头就行
- Prefix Injection 足够简单实用

**未验证 / 没人做的**：
- ❌ latent + tool use
- ❌ 跨架构 latent protocol
- ❌ 持久 latent state
- ❌ 多轮累积误差
- ❌ 安全/可解释性
- ❌ 真实生产部署

**你的最优切入点**：Hybrid Latent-Token with Tool Use。8×A100，3-4 周出结果。这是一个所有现有工作都没碰的硬 gap，且有明确的评测方案（ALFWorld/WebArena + 工具调用准确率）。

---

## 参考文献

1. LatentMAS — arXiv:2511.20639 (Princeton/Stanford, 2025.11)
2. Interlat — arXiv:2511.09149 (浙大等, ICLR 2026 submission)
3. Vision Wormhole — arXiv:2602.15382 (Purdue等, 2026.02)
4. Coconut — arXiv:2412.06769 (Meta FAIR, COLM 2025)
5. Thinking States — arXiv:2602.08332 (Google, 2026.02)
6. SoftCoT — arXiv:2502.12134 (ACL 2025)
7. Group Think — arXiv:2505.11107 (MediaTek Research, 2025.05)
8. Scaling Agent Systems — arXiv:2512.08296 (Google DeepMind/MIT, 2025.12)
9. ProtocolBench — arXiv:2510.17149 (UIUC, 2025.10)
10. Latent CoT Survey — arXiv:2505.16782 (2025.05)
