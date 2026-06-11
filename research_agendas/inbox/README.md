# Direction inbox

Drop a direction file (`*.yaml`, schema below) into this directory, then run `python3 -m scripts.agenda_inbox_watcher --once` (or let the systemd timer / cron job do it — see `deploy/agenda-inbox-watcher.example`).
Processed files move to `processed/` with a `<file>.echo.json` confirmation next to them; rejected files move to `failed/` with a `<file>.error.txt` explaining why.

```yaml
direction: "用扩散模型做小样本医学影像分割，关注跨中心泛化"  # required, natural language
keywords: [medical imaging, diffusion, few-shot]            # optional
constraints:                                                # optional, free text
  compute: "单卡以内"
  data: "仅公开数据集"
goal: experiment_plan   # idea_only | experiment_plan | signal | verified_evidence
contact: "nickname or email"  # required
token_budget: 500000    # optional LLM token cap; default 0 = no cap (usage is still recorded)
```
