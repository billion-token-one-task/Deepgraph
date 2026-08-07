# 两台 A100 机器的共用约定 (2026-08-07)

> **站点参数不写在本文件里。** 下面用符号代替真实取值:
> `$GPU_HOST` (主机地址)、`$GPU_PORT_A` / `$GPU_PORT_B` (两台机的 SSH 端口)、
> `$GPU_USER` (登录用户)、`~/.ssh/<gpu-key>` (密钥文件)。
> 真实取值在部署机的 `.env` 与私有运维记录中, 不进入本仓库。


写给同时在这两台机器上干活的多个 session / agent。谁先读到谁遵守, 改动请就地更新本文件。
本文件放在两台机器的 `/root/SHARED_GPU_HOSTS.md`。

## 机器事实

- `$GPU_HOST:$GPU_PORT_A` = `A100-A`, `:$GPU_PORT_B` = `A100-B`。
  **是两台独立物理机**, 不是一台机器的两个端口。
- 每台: 4x A100-PCIE-40GB (共 160GB 显存), 128 核, 251GB 内存, 800GB+ 空闲磁盘,
  aarch64 (Kunpeng-920), 计算能力 8.0 (Ampere, **不支持 FP8**), 驱动 570.86.10。
- 登录: `ssh -i ~/.ssh/<gpu-key> -p <$GPU_PORT_A|$GPU_PORT_B> $GPU_USER@$GPU_HOST` (免密)。

## 网络: 最重要的一条

从这两台机器出网, **必须走国内镜像**, 否则慢到不可用 (实测):

| 源 | 速度 |
|---|---|
| pypi.org | 27 kB/s (基本不可用) |
| 腾讯云 pypi 镜像 | 35 MB/s |
| 清华 pypi 镜像 | 31 MB/s |
| hf-mirror.com | 2.1 MB/s |

已在两台机器写好 `/etc/pip.conf` (腾讯为主源, 清华为备源), 所以 **直接 `pip install` 即可**,
不需要每次加 `-i`。下载 HuggingFace 模型前 `export HF_ENDPOINT=https://hf-mirror.com`。

## 谁在用什么

| 目录 | 归属 | 用途 |
|---|---|---|
| `/root/deepgraph-remote-worker/` | Deepgraph 调度器 | **不要碰**, 它会自行清理覆盖 |
| `/root/deepgraph-local-llm/` | Deepgraph 本地推理 ($GPU_PORT_B) | vLLM venv + 模型服务 |
| `/root/minimax-h3-test/` | 另一个 session | 图像/扩散模型测试 |

新开工作请自建 `/root/<你的项目名>/`。

## 可以共用的东西

1. **HuggingFace 缓存** `/root/.cache/huggingface`: 大家都是 root, 天然共享。
   **不要**改成自己的目录 —— 同一个模型只下一次, 在 2 MB/s 的链路上这能省几小时。
2. **vLLM 环境** `/root/deepgraph-local-llm/vllm-venv`: 已装好, 直接用它的
   `bin/python` / `bin/vllm`, 不必自己再装一遍 (省 GB 级下载)。
3. **本地推理端点** ($GPU_PORT_B 起来后): OpenAI 兼容, `http://127.0.0.1:8000/v1`,
   任何 session 都可以直接调, 不需要 API key 和额度。这是最值得共用的一件。
4. `/etc/pip.conf` 的镜像配置: 已全机生效, 谁都受益。

## 显卡分配 (避免互相 OOM)

一台只有 160GB 显存, 谁都别整机独占:

- **$GPU_PORT_B**: GPU 0-3 由 Deepgraph 本地推理使用 (vLLM 常驻)。
  其他 session 请优先用 **$GPU_PORT_A**, 或在 $GPU_PORT_B 上先 `nvidia-smi` 确认有余量。
- **$GPU_PORT_A**: 留给 Deepgraph 的 GPU 实验调度 (随时可能起任务) + 其他 session 的临时活。
- 无论在哪台, 都用 `CUDA_VISIBLE_DEVICES=<n>` 只锁需要的卡, 跑完释放。

注意: Deepgraph 的 `gpu_workers` 心跳表**看不到**外部占用, 反之亦然, 所以互相让路只能靠
`nvidia-smi` 自觉。

## 已知坑

- **aarch64**: 很多 x86 轮子装不上; 装包前确认有 arm64 版本, 否则会退到源码编译 (很慢)。
- **不支持 FP8**: Ampere 没有 FP8 张量核心。FP8 权重的模型 (如 DeepSeek-V4-Flash) 只能
  反量化成 bf16 跑, 体积翻倍 —— 167GB 的模型会变成 ~334GB, 装不下。
- **跨机不能合并显存**: 两台机之间没有 RDMA/InfiniBand, 8 张卡不能当一个 320GB 的池子用。
  单个模型必须能装进**一台**的 160GB。
