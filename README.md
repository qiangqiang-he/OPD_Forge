# OPD_Forge 使用说明

OPD_Forge 基于 VERL 实现策略梯度 OPD 与 GKD-OPD。正式实验配置位于：

- `configs/p1_pg_opd/`：统一 PG-OPD，以及 random（5%/10%/20%/40%）和 topgap（20%）选择配置
- `configs/p2_gkd_opd/`：统一 GKD-OPD，以及 random（5%/10%/20%/40%）和 topgap（20%）选择配置
- `configs/p3_cal_opd/`：Cal-OPD（正、负反馈校准 Teacher 自身偏移）
- `configs/temp/`：对应的 5-step 测试配置

PG-OPD 与 GKD-OPD 都只使用 sampled-token reverse KL。token 子集由
`distillation.distillation_loss.selection_ratio` 和 `selection_method` 控制；
`selection_method` 可取 `random`、`topgap` 或 `bottomgap`。当比例为 `1` 时
训练使用全部有效 token，选择方式不会参与计算。

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

## 评测训练 checkpoints

评测配置统一放在 `configs/eval/`。启动命令只需要脚本和配置路径；脚本会激活
同一工作区中的 `verl` Conda 环境：

```bash
bash scripts/start_eval.sh configs/eval/eval.yaml
```

配置中的 `training_output` 指向一个训练输出目录。评测器会按 step 数值顺序
发现其中全部 `global_step_*`，将各 checkpoint 的 actor FSDP 分片临时合并到该
`training_output` 内，并用 8 个单卡 vLLM worker 逐 checkpoint 评测。每个
checkpoint 结束后（包括异常退出）都会删除对应的临时合并目录；
`eval_results/` 只保存最终的 `<run_name>.json`。

正式占用 GPU 前可执行完整的 CPU 预检：

```bash
bash scripts/start_eval.sh configs/eval/eval.yaml --validate-only
```

预检只读取 YAML、数据集、tokenizer 与 checkpoint 元数据，不合并模型，也不启动
vLLM 或 GPU worker。

## 八卡 smoke

下面的脚本依次运行 thinking GKD-OPD 和 no-thinking PG-OPD，各训练 2 steps，
并在最后一步使用 AMC23 的一个最小样本计算 Avg@4：

```bash
bash tests/run_unified_opd_smoke.sh
```
