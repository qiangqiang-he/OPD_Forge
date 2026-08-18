# OPD_Forge 使用说明

OPD_Forge 基于 VERL 实现策略梯度 OPD 与 GKD-OPD。正式实验配置位于：

- `configs/p1_pg_opd/`：PG-OPD、Random PG-OPD（5%/10%/20%/40%）和 TopGap PG-OPD（20%）
- `configs/p2_gkd_opd/`：GKD-OPD、Random GKD-OPD（5%/10%/20%/40%）和 TopGap GKD-OPD（20%）
- `configs/p3_cal_opd/`：Cal-OPD（正、负反馈校准 Teacher 自身偏移）
- `configs/temp/`：对应的 5-step 测试配置

## 安装环境

需要 Linux、NVIDIA GPU/CUDA、Conda 和 `tmux`。先进入实际存放项目的目录；不要照搬某台服务器的绝对路径：

```bash
cd /path/to/OPD_Forge

# 当前 shell 尚未初始化 Conda 时执行；已能使用 conda activate 则跳过
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environments.yaml  # 已存在 verl 环境时跳过
conda activate verl
```

其中 `/path/to/OPD_Forge` 应替换为当前服务器上的项目路径。Conda 可以安装在任意位置，以上命令会通过 `conda info --base` 自动找到其安装目录。

如果不使用导出的完整环境，也可以安装精简依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps TransferQueue==0.1.7
python -m pip install --no-deps -e ./verl
```

## 启动训练

先设置 W&B 密钥，再将所需 YAML 交给统一启动脚本：

```bash
export WANDB_API_KEY="你的密钥"
bash scripts/start_train.sh configs/p1_pg_opd/pg_opd_math33k_100steps.yaml
```

例如启动 GKD-OPD：

```bash
bash scripts/start_train.sh configs/p2_gkd_opd/gkd_opd_math33k_100steps.yaml
```

启动 Cal-OPD：

```bash
bash scripts/start_train.sh configs/p3_cal_opd/cal_opd_math33k_100steps.yaml
```

训练会在独立的 `tmux` 会话中运行。启动后按终端提示执行 `tmux attach -t <会话名>` 查看；按 `Ctrl-b d` 可退出会话而不中止训练。checkpoint 和日志保存在 `outputs/<run_name>/`，目录会自动创建。

`run_name` 和 `group_name` 可直接在 YAML 中设置，也可在启动时覆盖：

```bash
RUN_NAME=my_run GROUP_NAME=my_group \
  bash scripts/start_train.sh configs/p1_pg_opd/topgap_pg_opd_math33k_ratio0p20_100steps.yaml
```

正式运行前请确认配置中的模型、训练集和评估集路径在当前机器上可访问。同一组 GPU 上不要同时启动多个训练。
