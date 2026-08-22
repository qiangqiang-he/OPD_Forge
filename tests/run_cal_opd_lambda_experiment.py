"""Test-only Cal-OPD trainer that records local lambda-sweep statistics.

This entrypoint deliberately lives under ``tests``: it does not change the
production trainer or send experiment data to W&B.  Each training step writes
one JSON object to ``cal_opd_lambda_stats.jsonl`` in the run output directory.
When ``cal_opd_lambda_analysis.rollout_only=true``, reward computation,
advantage construction, and actor updates are skipped: the realized student
rollouts and the three matched teacher-forcing passes are used only for local
counterfactual statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import ray
import torch

from algorithms import resolve_algorithm
from algorithms.cal_opd import CalOPDTrainer, normalize_cal_opd_token
from runners.opd_entrypoint import _BaseTaskRunner
from utils.opd_runtime import configure_opd_data_parallel_batch, configure_opd_defaults
from verl.trainer import main_ppo_sync as verl_sync
from verl.trainer.distillation.losses import compute_calibrated_opd_advantage
from verl.workers.utils.padding import no_padding_2_padding


DEFAULT_SWEEP_LAMBDAS = (1.0, 2.0, 3.0)


def _lambda_key(cal_lambda: float) -> str:
    value = float(cal_lambda)
    return str(int(value)) if value.is_integer() else str(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_lambda_statistics(
    *,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    positive_teacher_log_probs: torch.Tensor,
    negative_teacher_log_probs: torch.Tensor,
    sweep_lambdas: tuple[float, ...] = DEFAULT_SWEEP_LAMBDAS,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute matched lambda counterfactuals for one realized rollout batch."""

    with torch.no_grad():
        base_advantage = teacher_log_probs - student_log_probs
        _, teacher_self_deviation = compute_calibrated_opd_advantage(
            student_log_probs,
            teacher_log_probs,
            positive_teacher_log_probs,
            negative_teacher_log_probs,
        )
        valid = response_mask.bool()
        base_nonzero = valid & base_advantage.ne(0)
        calibrated = {
            _lambda_key(cal_lambda): compute_calibrated_opd_advantage(
                student_log_probs,
                teacher_log_probs,
                positive_teacher_log_probs,
                negative_teacher_log_probs,
                cal_lambda=cal_lambda,
            )[0]
            for cal_lambda in sweep_lambdas
        }

        valid_base = base_advantage[valid].float()
        base_l1 = valid_base.abs().sum().item()
        per_lambda: dict[str, Any] = {}
        for key, advantage in calibrated.items():
            valid_calibrated = advantage[valid].float()
            zeroed = base_nonzero & advantage.eq(0)
            per_lambda[key] = {
                "zero_token_count": int((valid & advantage.eq(0)).sum().item()),
                "zero_token_ratio": _safe_ratio(
                    int((valid & advantage.eq(0)).sum().item()), int(valid.sum().item())
                ),
                "newly_zero_from_base_count": int(zeroed.sum().item()),
                "newly_zero_from_base_ratio": _safe_ratio(
                    int(zeroed.sum().item()), int(base_nonzero.sum().item())
                ),
                "positive_newly_zero_count": int(
                    (zeroed & base_advantage.gt(0)).sum().item()
                ),
                "negative_newly_zero_count": int(
                    (zeroed & base_advantage.lt(0)).sum().item()
                ),
                "retained_l1_magnitude_ratio": _safe_ratio(
                    valid_calibrated.abs().sum().item(), base_l1
                ),
            }

        lambda_keys = [_lambda_key(cal_lambda) for cal_lambda in sweep_lambdas]
        transitions = {
            f"base_to_lambda{lambda_keys[0]}": base_nonzero
            & calibrated[lambda_keys[0]].eq(0)
        }
        transitions.update(
            {
                f"lambda{previous}_to_lambda{current}": (
                    valid
                    & calibrated[previous].ne(0)
                    & calibrated[current].eq(0)
                )
                for previous, current in zip(lambda_keys, lambda_keys[1:])
            }
        )
        summary = {
            "valid_token_count": int(valid.sum().item()),
            "base_nonzero_token_count": int(base_nonzero.sum().item()),
            "per_lambda": per_lambda,
        }
    return summary, transitions, base_advantage, teacher_self_deviation


def _aggregate_transition_tokens(
    *,
    response_token_ids: torch.Tensor,
    selection: torch.Tensor,
    base_advantage: torch.Tensor,
    teacher_self_deviation: torch.Tensor,
    tokenizer,
) -> list[dict[str, Any]]:
    selected_ids = response_token_ids[selection].long()
    if selected_ids.numel() == 0:
        return []

    selected_abs_base = base_advantage[selection].float().abs()
    selected_deviation = teacher_self_deviation[selection].float()
    unique_ids, inverse = torch.unique(selected_ids, sorted=True, return_inverse=True)
    counts = torch.zeros(unique_ids.numel(), dtype=torch.int64)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    base_sums = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    base_sums.scatter_add_(0, inverse, selected_abs_base)
    deviation_sums = torch.zeros(unique_ids.numel(), dtype=torch.float32)
    deviation_sums.scatter_add_(0, inverse, selected_deviation)

    decode_inputs = [[int(token_id)] for token_id in unique_ids.tolist()]
    try:
        decoded = tokenizer.batch_decode(
            decode_inputs,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.batch_decode(decode_inputs, skip_special_tokens=False)

    rows = []
    for index, (token_id, token_piece) in enumerate(
        zip(unique_ids.tolist(), decoded, strict=True)
    ):
        count = int(counts[index].item())
        normalized = normalize_cal_opd_token(token_piece)
        rows.append(
            {
                "token_id": int(token_id),
                "token": str(token_piece),
                "normalized_token": normalized[1] if normalized is not None else str(token_piece),
                "count": count,
                "mean_abs_base_advantage": float(base_sums[index].item() / count),
                "mean_teacher_self_deviation": float(
                    deviation_sums[index].item() / count
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-int(row["count"]), -float(row["mean_abs_base_advantage"])),
    )


class CalOPDLambdaStatsTrainer(CalOPDTrainer):
    """Cal-OPD trainer with local, matched counterfactual lambda diagnostics."""

    @property
    def _lambda_analysis_rollout_only(self) -> bool:
        analysis = self.config.get("cal_opd_lambda_analysis", {})
        return bool(analysis.get("rollout_only", False))

    def _compute_reward_colocate(self, batch):
        if self._lambda_analysis_rollout_only:
            return batch
        return super()._compute_reward_colocate(batch)

    def _compute_advantage(self, batch, metrics):
        if self._lambda_analysis_rollout_only:
            return batch
        return super()._compute_advantage(batch, metrics)

    def _update_actor(self, batch, metrics):
        if self._lambda_analysis_rollout_only:
            return batch
        return super()._update_actor(batch, metrics)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        if not self._lambda_analysis_rollout_only:
            super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "old_log_probs",
            "teacher_logprobs",
            "cal_positive_teacher_logprobs",
            "cal_negative_teacher_logprobs",
        ]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        response_token_ids = self._to_padded(
            data["responses"], self.tokenizer.pad_token_id
        ).detach().cpu()
        response_mask = self._to_padded(data["response_mask"], 0).bool()
        student_log_probs = self._to_padded(data["old_log_probs"], 0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        positive_teacher_log_probs = no_padding_2_padding(
            data["cal_positive_teacher_logprobs"], data
        ).squeeze(-1)
        negative_teacher_log_probs = no_padding_2_padding(
            data["cal_negative_teacher_logprobs"], data
        ).squeeze(-1)
        non_padding_rows = torch.tensor(
            [not tag.get("is_padding", False) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        response_mask &= non_padding_rows.unsqueeze(-1)

        analysis_config = self.config.get("cal_opd_lambda_analysis", {})
        sweep_lambdas = tuple(
            float(value)
            for value in analysis_config.get("lambdas", DEFAULT_SWEEP_LAMBDAS)
        )
        if not sweep_lambdas or any(value < 0 for value in sweep_lambdas):
            raise ValueError("cal_opd_lambda_analysis.lambdas must be non-negative")
        if tuple(sorted(set(sweep_lambdas))) != sweep_lambdas:
            raise ValueError(
                "cal_opd_lambda_analysis.lambdas must be unique and increasing"
            )

        summary, transitions, base_advantage, teacher_self_deviation = (
            compute_lambda_statistics(
                response_token_ids=response_token_ids,
                response_mask=response_mask.detach().cpu(),
                student_log_probs=student_log_probs.detach().float().cpu(),
                teacher_log_probs=teacher_log_probs.detach().float().cpu(),
                positive_teacher_log_probs=positive_teacher_log_probs.detach().float().cpu(),
                negative_teacher_log_probs=negative_teacher_log_probs.detach().float().cpu(),
                sweep_lambdas=sweep_lambdas,
            )
        )
        sample_count = int(non_padding_rows.sum().item())
        summary["sample_count"] = sample_count
        summary["mean_valid_tokens_per_response"] = _safe_ratio(
            summary["valid_token_count"], sample_count
        )
        for lambda_summary in summary["per_lambda"].values():
            lambda_summary["mean_newly_zero_tokens_per_response"] = _safe_ratio(
                lambda_summary["newly_zero_from_base_count"], sample_count
            )
        configured_lambda = float(
            self.config.distillation.distillation_loss.cal_lambda
        )
        configured_key = _lambda_key(configured_lambda)
        configured = summary["per_lambda"][configured_key]
        metrics.update(
            {
                "Cal-OPD/local-lambda-analysis/newly_zero_token_count": configured[
                    "newly_zero_from_base_count"
                ],
                "Cal-OPD/local-lambda-analysis/newly_zero_token_ratio": configured[
                    "newly_zero_from_base_ratio"
                ],
                "Cal-OPD/local-lambda-analysis/retained_l1_magnitude_ratio": configured[
                    "retained_l1_magnitude_ratio"
                ],
            }
        )

        record = {
            "step": int(global_steps),
            "configured_lambda": configured_lambda,
            **summary,
            "transitions": {
                name: {
                    "count": int(selection.sum().item()),
                    "mean_count_per_response": _safe_ratio(
                        int(selection.sum().item()), sample_count
                    ),
                    "tokens": _aggregate_transition_tokens(
                        response_token_ids=response_token_ids,
                        selection=selection,
                        base_advantage=base_advantage,
                        teacher_self_deviation=teacher_self_deviation,
                        tokenizer=self.tokenizer,
                    ),
                }
                for name, selection in transitions.items()
            },
        }
        stats_path = (
            Path(str(self.config.trainer.default_local_dir)).resolve()
            / "cal_opd_lambda_stats.jsonl"
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

        if bool(analysis_config.get("save_token_tensors", False)):
            torch.save(
                {
                    "response_token_ids": response_token_ids,
                    "response_mask": response_mask.detach().cpu(),
                    "student_log_probs": student_log_probs.detach().float().cpu(),
                    "teacher_log_probs": teacher_log_probs.detach().float().cpu(),
                    "positive_teacher_log_probs": positive_teacher_log_probs.detach()
                    .float()
                    .cpu(),
                    "negative_teacher_log_probs": negative_teacher_log_probs.detach()
                    .float()
                    .cpu(),
                    "sample_count": torch.tensor(sample_count, dtype=torch.int64),
                },
                stats_path.with_name("cal_opd_token_tensors.pt"),
            )


@ray.remote
class CalOPDLambdaTaskRunner(_BaseTaskRunner):
    """Project OPD task runner selecting the test-only statistics trainer."""

    def run(self, config):
        configure_opd_defaults(config)
        verl_sync.OmegaConf.resolve(config)
        resolve_algorithm(config).validate(config)

        verl_sync.tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            self.resource_pool_manager.max_colocate_count = 2
            trainer = CalOPDLambdaStatsTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            verl_sync.tq.close()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    configure_opd_defaults(config)
    configure_opd_data_parallel_batch(config)
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    resolve_algorithm(config).validate(config)
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    verl_sync.run_ppo(config, task_runner_class=CalOPDLambdaTaskRunner)


if __name__ == "__main__":
    main()
