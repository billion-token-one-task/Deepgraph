# 怎么用这批 A100 (给人和给 AI 的调用说明)

> **站点参数不写在本文件里。** 下面用符号代替真实取值:
> `$GPU_HOST` (主机地址)、`$GPU_PORT_A` / `$GPU_PORT_B` (两台机的 SSH 端口)、
> `$GPU_USER` (登录用户)、`~/.ssh/<gpu-key>` (密钥文件)。
> 真实取值在部署机的 `.env` 与私有运维记录中, 不进入本仓库。


硬件事实 (2026-08-07 实测, 修正早先的说法): **两台独立物理机**, 不是一台机器的两个端口 --
`$GPU_HOST:$GPU_PORT_A` = 主机 `A100-A`, `:$GPU_PORT_B` = 主机 `A100-B`, 各 4 张
NVIDIA A100-PCIE-40GB。**每台 128 核 / 251GB 内存 / 800GB+ 空闲磁盘, 常态负载接近 0**,
也就是说这是两台几乎完全闲置的大机器, 不只是显卡。宿主 CPU 是 aarch64 (Kunpeng-920), 驱动 570.86.10,
torch 2.5.1+cu124, 实测 fp16 matmul8192 = 159.7 TFLOPS。远端工作根目录 `/root/deepgraph-remote-worker`,
python 用 `python3`, HuggingFace 走 `HF_ENDPOINT=https://hf-mirror.com`。

**登录凭据不写在任何文档或代码里**: 密码在部署机 `/home/billion-token/Deepgraph/.env` 的
`DEEPGRAPH_SSH_CREDENTIAL`, 代码只保存**引用** `env:DEEPGRAPH_SSH_CREDENTIAL` (worker 行的
`credential_ref` 字段)。SSH 走 `SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force` 注入, 不用 sshpass。
要给别人/别的 agent 用, 传给他们的是"去哪读凭据", 不是凭据本身。

下面三种调用方式, 按"你是谁"选。

## 方式 A: 你是 Deepgraph 自己的实验 (推荐, 有记账)

不要自己连 SSH。把工作交给调度器, 它负责选卡、同步代码、跑、收产物、记账:

1. 候选必须挂在一个 active 的 ResourceGrant 上, 且 grant 的 `backend_allowlist` 含 `ssh_gpu`、
   `max_gpu_hours > 0` (V1 的 scripts/auto_advance.py 会自动发这种 grant);
2. `scripts/auto_execute.py` 把候选送进 forge, 产出 `experiment_runs` 行并排队 `gpu_jobs`;
3. web 进程里的 GPU 调度线程认领 `gpu_jobs`, 通过 `orchestrator/ssh_gpu_backend.run_remote_experiment`
   在某张 A100 上执行, 回收 metric 与产物。

卡的账走 agenda 的 `gpu_hours_budget`; 没有 grant 就没有卡, 这是刻意的 fail-closed。

## 方式 B: 你是本仓库里的一段 Python (要走生产通路但自己控制命令)

用系统自己的后端函数, 好处是凭据/端口/依赖安装/产物回传都已经处理好:

```python
from db import database as db
from orchestrator.ssh_gpu_backend import run_remote_experiment

worker = db.fetchone(
    "SELECT * FROM gpu_workers WHERE id=? ",
    ("ssh:$GPU_HOST:$GPU_PORT_A:gpu0",),
)
result = run_remote_experiment(
    worker=dict(worker),
    run_id=999001,                      # 只用于远端目录命名; 非业务表写入
    local_workdir=Path("/tmp/myrun"),   # 会被 rsync 到远端
    local_code_dir=Path("/tmp/myrun/code"),
    time_budget=600,                    # 秒, 硬超时
    command_tokens=["python3", "train.py"],
    local_python="python3",
    benchmark_env={"HF_ENDPOINT": "https://hf-mirror.com"},
)
```

worker 行里的 `metadata` 决定连哪台: `ssh_host` / `ssh_port` / `ssh_user` / `credential_ref` /
`remote_base_dir` / `python_bin` / `visible_device`。选卡时优先挑 `status='idle'` 且
`heartbeat_at` 新鲜的行 (心跳由 web 进程每轮刷新)。

## 方式 C: 你是外部进程 / 另一个 AI, 只想借算力

**给密钥, 不要给密码。** 密码是调度器在用的同一把凭据: 交出去就无法单独收回, 只能改密码并
同步改 .env; 而且拿到的一方多半会把它明文写进某个 askpass 脚本长期留在磁盘上 (2026-08-07
真的发生过一次, 被拦下)。密钥可以单独吊销 -- 删 authorized_keys 一行即可。

一次性配好: 运维在部署机执行 `bash ~/install_gpu_key.sh` (脚本从 .env 取密码、临时 askpass
注入、用完即删, 全程不打印密码), 它把 `~/.ssh/<gpu-key>.pub` 追加到两个端点 root 的
authorized_keys。之后告诉对方:

- 目标: `ssh -i ~/.ssh/<gpu-key> -p $GPU_PORT_A $GPU_USER@$GPU_HOST` (另一台 `-p $GPU_PORT_B`),
  每台 4 张 A100-40GB, **不需要任何密码**;
- 若对方不在这台部署机上, 由运维用带外方式送私钥, 不要贴进聊天; 泄漏后果也仅限于吊销这一把;
- 工作目录: 自建 `/root/<你的名字>-work/`, **不要碰 `/root/deepgraph-remote-worker`**
  (那是调度器的地盘, 它会清理和覆盖里面的 run 目录);
- 环境: `python3` 已带 torch 2.5.1+cu124 / transformers / datasets / accelerate;
  下载模型务必先 `export HF_ENDPOINT=https://hf-mirror.com`;
- 选卡: 先 `nvidia-smi` 看哪张空闲, 用 `CUDA_VISIBLE_DEVICES=<n>` 锁定一张, 不要整机独占 --
  Deepgraph 的实验随时可能被调度到别的卡上;
- 时间: 长任务请挂 `nohup` / `tmux`, 并自设超时; 调度器对自己的任务有硬超时, 对你的没有。

一句话模板 (可直接贴给另一个 AI):

> 你有 8 张 A100-40GB, 在 `$GPU_HOST` 的两个端口 `$GPU_PORT_A` 和 `$GPU_PORT_B` (各 4 张),
> 用户 root, **用密钥登录**: `ssh -i ~/.ssh/<gpu-key> -p $GPU_PORT_A $GPU_USER@$GPU_HOST`,
> 不需要密码。不要索要密码, 也不要把任何凭据写进脚本或聊天; 密钥不通就找运维重装公钥。
> 远端 python3 已装好 torch 2.5.1+cu124/transformers/datasets/accelerate, aarch64 架构,
> 装包时注意选 aarch64 轮子。下载 HuggingFace 模型前必须
> `export HF_ENDPOINT=https://hf-mirror.com`。请在 `/root/<你的项目名>/` 下工作,
> 用 `CUDA_VISIBLE_DEVICES` 只占用空闲卡, 不要动 `/root/deepgraph-remote-worker`。

## 注意事项

- **两个端口是两台独立机器** (2026-08-07 更正): 各自 128 核 / 251GB / 独立磁盘, 互不影响,
  可以放心并行跑两份重活。这也意味着 CPU 密集的活 (PDF 解析、向量化、本地推理)
  应该放这里, 而不是只有 2 核 7GB 的部署机。
- **aarch64**: 很多 x86 预编译轮子装不上; 装包前确认有 aarch64 版本, 否则会退到源码编译。
- **心跳**: `gpu_workers` 的 `heartbeat_at` 只由 web 进程的注册循环刷新, 且只刷
  `DEEPGRAPH_GPU_REMOTE_SSH_ENDPOINTS` 里列出的端点。外部占用不会反映在这张表里 --
  也就是说调度器不知道你占了卡, 反之亦然, 所以请自觉用 `nvidia-smi` 让路。
- **别把凭据复制进代码/文档/聊天记录**; 现有架构刻意只存 `env:` 引用。
