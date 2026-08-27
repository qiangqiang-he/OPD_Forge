# Fast OA-OPD 运行配置说明

## 结论

Fast OA-OPD 已经配置完成。按原来的方式指定 YAML 一键启动即可，不需要手动设置环境变量，也不需要在启动前执行额外命令。

```bash
bash scripts/run_training.sh configs/Exploratory/Fast_OA_OPD/fast_oa_opd_qwen3_4b_to_1p7b_no_thinking_200steps.yaml
```

## Teacher 必须开启和关闭的项目

这些要求只作用于 Fast OA-OPD 的 Teacher 推理，不会修改 Student 的训练和 rollout 设置。

| 项目 | 必须状态 | 配置 | 作用 |
|---|---:|---|---|
| Eager 模式 | 开启 | `enforce_eager: true` | 让独立 probe 和 masked probe 使用一致的 eager 数值路径；当前 PG/OA Teacher 默认也使用 eager |
| vLLM 编译 | 关闭 | 由 `enforce_eager: true` 自动关闭 | 当前 PG/OA Teacher 同样关闭，因此这不是 Fast 相对它们新增的性能损失 |
| CUDA Graph | 关闭 | 由 `enforce_eager: true` 自动关闭 | 当前 PG/OA Teacher 同样关闭，无需单独配置 |
| FlashAttention | 开启 | `attention_backend: FLASH_ATTN` | 执行 Fast OA-OPD 所需的变长 causal attention |
| Batch invariant | 开启 | `fast_oa_opd_batch_invariant: true` | 保证 packed masked branch 与独立 probe 的数值等价性 |
| Fast masked probe 调度 | 自定义路径 | 自动生效 | 绕过普通 vLLM request scheduler；每个 Teacher replica 串行执行一次 masked forward，rollout 在 4 个 replica 间负载均衡 |

对应配置如下：

```yaml
distillation:
  teacher_models:
    teacher_model:
      inference:
        enforce_eager: true
        max_num_seqs: 8
        engine_kwargs:
          vllm:
            attention_backend: FLASH_ATTN
            fast_oa_opd_batch_invariant: true
```

## Batch invariant 如何生效

`fast_oa_opd_batch_invariant` 是项目提供的配置开关。Teacher vLLM worker 启动前，配套代码会自动完成：

```text
fast_oa_opd_batch_invariant: true
                    ↓
VLLM_BATCH_INVARIANT=1
```

用户不需要手动导出 `VLLM_BATCH_INVARIANT`。该设置用于控制 masked forward 所用 CUDA kernel 的数值行为。Fast masked probe 通过 worker RPC 执行，不进入普通 vLLM 动态 request batching；外层 load balancer 仍会把不同 rollout 分发给 4 个 Teacher replica。

## 如果配置不正确会怎样

Fast OA-OPD 启动时会检查上述三个必要条件：

- `enforce_eager` 没开：直接报配置错误并停止启动；
- `attention_backend` 不是 `FLASH_ATTN`：直接报配置错误并停止启动；
- `fast_oa_opd_batch_invariant` 没开：直接报配置错误并停止启动。

因此不会在配置错误时静默开始训练。如果绕过检查继续执行，Teacher 的 probe 分数 $V_k^T$ 和差值 $\Delta_k^T$ 可能产生明显数值漂移：本地测试中编译路径的最大偏差约为 `0.03`，改变动态 batch 组成时个别偏差约为 `0.18`。

使用当前 Fast 配置后，新旧实现的数值等价测试最大误差约为 `3.4e-7`。

## 当前配置的实际范围

- Fast Teacher：Eager 开、FlashAttention 开、batch invariant 开；masked probe 绕过普通 vLLM request scheduler，在 4 个 Teacher replica 间做 rollout 级负载均衡；
- Student vLLM rollout：保持原来的编译/CUDA Graph 和自动 attention 配置；
- Student Actor 训练：保持原来的 FSDP 与动态 token micro-batch 配置；
- 原始 `oa_opd.py`：没有被覆盖，仍可作为 reference 使用。
