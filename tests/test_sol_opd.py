"""CPU and mock regressions for solution-privileged Sol-OPD.

The production algorithm remains an eight-GPU 4-Student + 4-Teacher job.
These tests exercise the same prompt, dataset, causal alignment, dual-Teacher,
statistics, and REINFORCE contracts without loading two models together.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERL_SOURCE_ROOT = PROJECT_ROOT / "verl"
if str(VERL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_SOURCE_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_Sol_OPD_Thinking"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "DAPO-17k-English-Qwen3-4B-Instruct-2507-Correct.json"
)
CONFIGS = {
    "pub_sol_opd_qwen3_4b_thinking_2507_to_1p7b_thinking_len8k_100steps": (
        "Qwen3-1.7B",
        "Qwen3-4B-Thinking-2507",
        1,
    ),
    "pub_sol_opd_qwen3_30b_a3b_thinking_2507_to_4b_thinking_len8k_100steps": (
        "Qwen3-4B",
        "Qwen3-30B-A3B-Thinking-2507",
        4,
    ),
    "pub_sol_opd_qwen3_30b_a3b_thinking_2507_to_8b_thinking_len8k_100steps": (
        "Qwen3-8B",
        "Qwen3-30B-A3B-Thinking-2507",
        4,
    ),
}


class CharacterTokenizer:
    """Small reversible tokenizer used to make prompt budgets exact on CPU."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        del kwargs
        assert not add_special_tokens
        return [ord(character) for character in text]

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        del kwargs
        return {
            "input_ids": self.encode(
                text, add_special_tokens=add_special_tokens
            )
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(int(token_id)) for token_id in token_ids)


def _compose(config_name: str):
    from hydra import compose, initialize_config_dir

    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=[search_path])


def _metric_values(metrics) -> dict[str, float]:
    return {key: float(value.aggregate()) for key, value in metrics.items()}


def _first_json_array_record(path: Path) -> dict:
    """Decode only the first object instead of loading the 79 MB dataset."""

    decoder = json.JSONDecoder()
    buffer = ""
    with path.open("r", encoding="utf-8") as stream:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                raise AssertionError(f"Could not decode a record from {path}.")
            buffer += chunk
            candidate = buffer.lstrip()
            if not candidate.startswith("["):
                raise AssertionError(f"Expected a JSON array in {path}.")
            candidate = candidate[1:].lstrip()
            try:
                record, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                raise AssertionError("The first Sol-OPD dataset record is not an object.")
            return record


def test_solution_prompt_is_registered_as_qwen3_thinking_mode():
    from utils.prompts import (
        get_sol_privileged_prompt_name,
        render_prompt,
    )

    question = "Compute 17 + 25."
    solution = r"Add the two integers to obtain \boxed{42}."
    rendered = render_prompt(
        "qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=solution,
    )

    assert rendered.count(question) == 1
    assert rendered.count(solution) == 1
    assert "A reference solution is provided as additional guidance:" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")
    # In this repository Qwen3 thinking mode starts at the bare assistant
    # boundary.  Pre-closing an empty think block would disable thinking.
    assert "<think>\n\n</think>" not in rendered
    assert get_sol_privileged_prompt_name("qwen3_thinking_prompt") == (
        "qwen3_privileged_solution_thinking"
    )
    with pytest.raises(ValueError, match="thinking Student prompt"):
        get_sol_privileged_prompt_name("qwen3_no_thinking_prompt")


def test_real_training_data_exposes_lowercase_nonempty_solution_field():
    assert DATA_PATH.is_file()
    record = _first_json_array_record(DATA_PATH)

    assert {"question", "answer", "solution"}.issubset(record)
    for field_name in ("question", "answer", "solution"):
        assert isinstance(record[field_name], str)
        assert record[field_name].strip()
    assert "Solution" not in record


def _dataset_config():
    return OmegaConf.create(
        {
            "student_prompt": "qwen3_thinking_prompt",
            "teacher_prompt": "qwen3_privileged_solution_thinking",
            "prompt_key": "prompt",
            "filter_overlong_prompts": False,
            "shuffle": False,
            "seed": 42,
        }
    )


def test_dataset_preserves_solution_and_allows_validation_without_it(tmp_path: Path):
    from utils.custom_dataset import CustomDataset

    train_solution = r"First add one and one; the result is \boxed{2}."
    train_path = tmp_path / "train.json"
    train_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is 1+1?",
                    "answer": 2,
                    "solution": train_solution,
                }
            ]
        ),
        encoding="utf-8",
    )
    val_path = tmp_path / "val.json"
    val_path.write_text(
        json.dumps([{"question": "What is 2+2?", "answer": "4"}]),
        encoding="utf-8",
    )

    tokenizer = CharacterTokenizer()
    train = CustomDataset(
        str(train_path), tokenizer, _dataset_config(), max_samples=-1
    )
    validation = CustomDataset(
        str(val_path), tokenizer, _dataset_config(), max_samples=-1
    )

    train_row = train.dataframe[0]
    assert train_row["sol_question"] == "What is 1+1?"
    assert train_row["sol_privileged_solution"] == train_solution
    assert train_solution in train_row["teacher_prompt_text"]
    assert train_solution not in train_row["prompt"]
    assert train_row["reward_model"]["ground_truth"] == "2"

    val_row = validation.dataframe[0]
    assert val_row["sol_question"] == "What is 2+2?"
    assert val_row["solution"] == ""
    assert val_row["sol_privileged_solution"] == ""
    assert val_row["teacher_prompt_text"].endswith(
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize(
    ("config_name", "student_name", "teacher_name", "teacher_tp"),
    [
        (config_name, *expected)
        for config_name, expected in CONFIGS.items()
    ],
)
def test_formal_sol_configs_keep_8192_lengths_and_four_plus_four_resources(
    config_name: str,
    student_name: str,
    teacher_name: str,
    teacher_tp: int,
):
    from algorithms import resolve_algorithm
    from algorithms.sol_opd import validate_sol_opd_config
    from runners.opd_entrypoint import OPDTaskRunner
    from utils.opd_runtime import (
        configure_opd_data_parallel_batch,
        configure_opd_defaults,
    )
    from verl.trainer import main_ppo_sync as verl_sync

    config = _compose(config_name)
    OmegaConf.resolve(config)

    assert str(config.run_name) == config_name
    assert str(config.run_name_prefix) == config_name
    assert str(config.group_name) == "PUB_Sol_OPD_Thinking"
    assert str(config.trainer.experiment_name) == config_name
    assert int(config.trainer.total_training_steps) == 100
    assert int(config.trainer.test_freq) == 20
    assert int(config.trainer.save_freq) == 20

    assert str(config.algorithm.name) == "sol_opd"
    assert str(config.student_prompt) == "qwen3_thinking_prompt"
    assert str(config.teacher_prompt) == (
        "qwen3_privileged_solution_thinking"
    )
    assert list(config.data.train_files) == [
        "./data/DAPO-17k-English-Qwen3-4B-Instruct-2507-Correct.json"
    ]
    assert int(config.data.max_prompt_length) == 2048
    assert int(config.data.max_response_length) == 8192
    assert int(config.rlvr_generation.train_max_new_tokens) == 8192
    assert int(config.rlvr_generation.val_max_new_tokens) == 8192

    rollout = config.actor_rollout_ref.rollout
    assert str(config.actor_rollout_ref.model.path) == f"./models/{student_name}"
    assert int(rollout.tensor_model_parallel_size) == 1
    assert int(rollout.data_parallel_size) == 4
    assert int(rollout.max_model_len) == 10240
    assert int(rollout.max_num_batched_tokens) == 10240
    assert int(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu) == 10241
    assert int(rollout.log_prob_max_token_len_per_gpu) == 10241
    assert int(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu) == 10241

    teacher = config.distillation.teacher_models.teacher_model
    assert str(teacher.model_path) == f"./models/{teacher_name}"
    assert int(teacher.inference.prompt_length) == 8192
    assert int(teacher.inference.response_length) == 8192
    assert int(teacher.inference.max_model_len) == 16385
    assert int(teacher.inference.tensor_model_parallel_size) == teacher_tp
    assert int(teacher.inference.data_parallel_size) == 1

    # These are separate production pools, not a local one-GPU override.
    student_gpus = int(config.trainer.n_gpus_per_node) * int(
        config.trainer.nnodes
    )
    teacher_gpus = int(config.distillation.n_gpus_per_node) * int(
        config.distillation.nnodes
    )
    assert (student_gpus, teacher_gpus, student_gpus + teacher_gpus) == (4, 4, 8)

    loss = config.distillation.distillation_loss
    assert str(loss.loss_mode) == "sol_reverse_kl"
    assert str(loss.policy_loss_mode) == "reinforce"
    assert bool(loss.use_policy_gradient)
    assert not bool(loss.use_task_rewards)
    assert float(loss.selection_ratio) == 1.0
    assert float(loss.loss_max_clamp) == 20.0
    assert loss.log_prob_min_clamp is None
    assert float(loss.sol_opd_epsilon) == pytest.approx(1.0e-6)

    configure_opd_defaults(config)
    assert configure_opd_data_parallel_batch(config) == 256
    spec = resolve_algorithm(config)
    assert spec.name == "sol_opd"
    spec.validate(config)
    validate_sol_opd_config(config)
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    runner = OPDTaskRunner.__ray_metadata__.modified_class()
    runner.add_actor_rollout_worker(config)
    runner.add_critic_worker(config)
    runner.init_resource_pool_mgr(config)
    runner.resource_pool_manager.max_colocate_count = 2
    assert runner.resource_pool_manager.resource_pool_spec == {
        "global_pool": [4],
        "teacher_pool": [4],
    }
    assert runner.resource_pool_manager.get_n_gpus() == 8


def test_solution_prompt_truncation_preserves_structure_question_and_solution_tail():
    from utils.prompts import render_prompt
    from verl.experimental.agent_loop.agent_loop import (
        _tokenize_solution_privileged_prompt,
    )

    tokenizer = CharacterTokenizer()
    question = "QUESTION-MUST-SURVIVE: Compute 6 times 7."
    solution = (
        "SOLUTION-BEGIN-MAY-BE-DROPPED\n"
        + "intermediate reasoning. " * 600
        + r"SOLUTION-TAIL-MUST-SURVIVE: \boxed{42}"
    )
    marker = "UNIQUE-SOLUTION-MARKER"
    marked = render_prompt(
        "qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=marker,
    )
    prefix, suffix = marked.split(marker)

    token_ids = _tokenize_solution_privileged_prompt(
        tokenizer,
        prompt_name="qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=solution,
        max_prompt_length=8192,
    )
    rendered = tokenizer.decode(token_ids)

    assert len(token_ids) <= 8192
    assert rendered.startswith(prefix)
    assert rendered.endswith(suffix)
    retained_solution = rendered[len(prefix) : -len(suffix)]
    assert retained_solution
    assert solution.endswith(retained_solution)
    assert len(retained_solution) < len(solution)
    assert question in rendered
    assert "SOLUTION-TAIL-MUST-SURVIVE" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")

    short_solution = r"Directly obtain \boxed{42}."
    short_expected = render_prompt(
        "qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=short_solution,
    )
    short_ids = _tokenize_solution_privileged_prompt(
        tokenizer,
        prompt_name="qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=short_solution,
        max_prompt_length=8192,
    )
    assert short_ids == tokenizer.encode(short_expected, add_special_tokens=False)


def test_solution_prompt_truncation_fails_closed_instead_of_truncating_question():
    from verl.experimental.agent_loop.agent_loop import (
        _tokenize_solution_privileged_prompt,
    )

    tokenizer = CharacterTokenizer()
    with pytest.raises((RuntimeError, ValueError)):
        _tokenize_solution_privileged_prompt(
            tokenizer,
            prompt_name="qwen3_privileged_solution_thinking",
            question="Q" * 9000,
            privileged_solution=r"Keep \boxed{1}.",
            max_prompt_length=8192,
        )
    with pytest.raises((RuntimeError, ValueError)):
        _tokenize_solution_privileged_prompt(
            tokenizer,
            prompt_name="qwen3_privileged_solution_thinking",
            question="Nonempty question",
            privileged_solution="",
            max_prompt_length=8192,
        )


def test_sol_statistics_match_hand_computed_token_and_rollout_sufficient_stats():
    from verl.trainer.distillation.losses import (
        compute_sol_opd_comparison_statistics,
    )

    # Rollout 1 contributes one amplified, one reduced, one sign-flipped, and
    # one epsilon-unchanged token. Rollout 2 contributes two amplified tokens.
    opd = torch.tensor(
        [[1.0, 2.0, -3.0, 0.25e-6], [1.0, 1.0, 99.0, 99.0]]
    )
    sol = torch.tensor(
        [[2.0, 1.0, 3.0, -0.25e-6], [3.0, 3.0, -99.0, -99.0]]
    )
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )

    values = _metric_values(
        compute_sol_opd_comparison_statistics(opd, sol, mask, epsilon=1.0e-6)
    )
    prefix = "distillation/sol_stats/"

    assert values[f"{prefix}token_count"] == 6
    assert values[f"{prefix}rollout_count"] == 2
    assert values[f"{prefix}ratio_rollout_count"] == 2
    assert values[f"{prefix}same_direction_token_count"] == 4
    assert values[f"{prefix}same_direction_rollout_count"] == 2
    assert values[f"{prefix}opd_abs_sum"] == pytest.approx(8.00000025)
    assert values[f"{prefix}sol_abs_sum"] == pytest.approx(12.00000025)
    assert values[f"{prefix}deviation_abs_sum"] == pytest.approx(12.0000005)
    assert values[f"{prefix}amplified_count"] == 3
    assert values[f"{prefix}reduced_count"] == 1
    assert values[f"{prefix}equal_count"] == 2
    assert values[f"{prefix}same_direction_amplified_count"] == 3
    assert values[f"{prefix}same_direction_reduced_count"] == 1
    assert values[f"{prefix}sign_flipped_count"] == 1
    assert values[f"{prefix}category_amplified_count"] == 3
    assert values[f"{prefix}category_reduced_count"] == 1
    assert values[f"{prefix}category_sign_flipped_count"] == 1
    assert values[f"{prefix}category_unchanged_count"] == 1

    # Token pooling gives 12/8 = 1.5, whereas rollout-first averaging gives
    # mean([6/6, 6/2]) = 2.0.
    assert values[f"{prefix}rollout_magnitude_ratio_sum"] == pytest.approx(4.0)
    assert values[f"{prefix}rollout_deviation_ratio_sum"] == pytest.approx(
        10.0 / 3.0
    )
    assert values[f"{prefix}rollout_amplification_rate_sum"] == pytest.approx(
        1.25
    )
    assert values[f"{prefix}rollout_reduction_rate_sum"] == pytest.approx(0.25)
    assert values[f"{prefix}rollout_equal_rate_sum"] == pytest.approx(0.5)
    assert values[f"{prefix}rollout_sign_flip_rate_sum"] == pytest.approx(0.25)
    assert values[
        f"{prefix}rollout_same_direction_amplification_rate_sum"
    ] == pytest.approx(1.5)
    assert values[
        f"{prefix}rollout_same_direction_reduction_rate_sum"
    ] == pytest.approx(0.5)


def test_sol_statistics_apply_epsilon_and_exclude_zero_denominator_rollouts():
    from verl.trainer.distillation.losses import (
        compute_sol_opd_comparison_statistics,
    )

    opd = torch.tensor([[0.25e-6, -0.25e-6], [1.0, -1.0]])
    sol = torch.tensor([[-2.0e-6, 2.0e-6], [-2.0, -1.0]])
    mask = torch.ones_like(opd, dtype=torch.bool)
    values = _metric_values(
        compute_sol_opd_comparison_statistics(opd, sol, mask, epsilon=1.0e-6)
    )
    prefix = "distillation/sol_stats/"

    # Near-zero OPD values cannot create sign flips even if Sol has the
    # opposite arithmetic sign. Their rollout is excluded from ratio means.
    assert values[f"{prefix}sign_flipped_count"] == 1
    assert values[f"{prefix}amplified_count"] == 3
    assert values[f"{prefix}equal_count"] == 1
    assert values[f"{prefix}rollout_count"] == 2
    assert values[f"{prefix}ratio_rollout_count"] == 1
    assert values[f"{prefix}same_direction_token_count"] == 1
    assert values[f"{prefix}same_direction_rollout_count"] == 1
    assert values[f"{prefix}rollout_magnitude_ratio_sum"] == pytest.approx(1.5)
    assert values[f"{prefix}rollout_deviation_ratio_sum"] == pytest.approx(1.5)


def test_sol_statistics_reject_invalid_inputs():
    from verl.trainer.distillation.losses import (
        compute_sol_opd_comparison_statistics,
    )

    valid = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_sol_opd_comparison_statistics(
            torch.zeros((1, 2)), torch.zeros((1, 3)), valid
        )
    empty = _metric_values(
        compute_sol_opd_comparison_statistics(
            torch.zeros((1, 2)), torch.zeros((1, 2)), torch.zeros_like(valid)
        )
    )
    assert empty["distillation/sol_stats/token_count"] == 0
    assert empty["distillation/sol_stats/rollout_count"] == 0
    assert empty["distillation/sol_stats/ratio_rollout_count"] == 0
    assert all(value == 0.0 for value in empty.values())
    with pytest.raises(ValueError, match="finite"):
        compute_sol_opd_comparison_statistics(
            torch.tensor([[float("nan"), 0.0]]), torch.zeros((1, 2)), valid
        )
    with pytest.raises(ValueError, match="positive"):
        compute_sol_opd_comparison_statistics(
            torch.zeros((1, 2)), torch.zeros((1, 2)), valid, epsilon=0.0
        )


def test_sol_wandb_metrics_publish_token_pooled_and_rollout_mean_namespaces():
    from algorithms.sol_opd import SolOPDTrainer
    from verl.trainer.distillation.losses import (
        compute_sol_opd_comparison_statistics,
    )

    opd = torch.tensor(
        [[1.0, 2.0, -3.0, 0.25e-6], [1.0, 1.0, 0.0, 0.0]]
    )
    sol = torch.tensor(
        [[2.0, 1.0, 3.0, -0.25e-6], [3.0, 3.0, 0.0, 0.0]]
    )
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    sufficient = compute_sol_opd_comparison_statistics(
        opd, sol, mask, epsilon=1.0e-6
    )
    metrics = {
        f"actor/{key}": float(value.aggregate())
        for key, value in sufficient.items()
    }
    metrics.update(
        {
            "actor/distillation/reverse_kl_estimate": 0.2,
            "actor/distillation/student_sampled_token_prob": 0.1,
            "actor/distillation/teacher_sampled_token_prob": 0.3,
            "actor/distillation/sol_unprivileged_teacher_sampled_token_prob": 0.2,
            "actor/distillation/selected_token_ratio": 1.0,
            "actor/distillation/loss": 0.4,
        }
    )

    trainer = object.__new__(SolOPDTrainer)
    trainer.config = {
        "algorithm": {"name": "sol_opd"},
        "student_prompt": "qwen3_thinking_prompt",
        "teacher_prompt": "qwen3_privileged_solution_thinking",
    }
    trainer._add_opd_training_metrics(metrics)

    token = "sol-opd/train/token"
    rollout = "sol-opd/train/rollout_mean"
    assert metrics[f"{token}/overall_magnitude_ratio"] == pytest.approx(1.5)
    assert metrics[f"{token}/overall_magnitude_change_percent"] == pytest.approx(
        50.0
    )
    assert metrics[f"{token}/amplification_rate"] == pytest.approx(0.5)
    assert metrics[f"{token}/reduction_rate"] == pytest.approx(1 / 6)
    assert metrics[f"{token}/equal_rate"] == pytest.approx(1 / 3)
    assert metrics[f"{token}/sign_flip_rate"] == pytest.approx(1 / 6)
    assert metrics[f"{token}/same_direction_amplification_rate"] == pytest.approx(
        0.75
    )
    assert metrics[f"{token}/same_direction_reduction_rate"] == pytest.approx(
        0.25
    )
    assert metrics[f"{token}/same_direction_eligible_rate"] == pytest.approx(2 / 3)
    assert metrics[f"{token}/privilege_induced_deviation_ratio"] == pytest.approx(
        1.5
    )
    assert metrics[f"{rollout}/overall_magnitude_ratio"] == pytest.approx(2.0)
    assert metrics[f"{rollout}/overall_magnitude_change_percent"] == pytest.approx(
        100.0
    )
    assert metrics[f"{rollout}/amplification_rate"] == pytest.approx(0.625)
    assert metrics[f"{rollout}/reduction_rate"] == pytest.approx(0.125)
    assert metrics[f"{rollout}/equal_rate"] == pytest.approx(0.25)
    assert metrics[f"{rollout}/sign_flip_rate"] == pytest.approx(0.125)
    assert metrics[
        f"{rollout}/same_direction_amplification_rate"
    ] == pytest.approx(0.75)
    assert metrics[f"{rollout}/same_direction_reduction_rate"] == pytest.approx(
        0.25
    )
    assert metrics[f"{rollout}/privilege_induced_deviation_ratio"] == pytest.approx(
        5 / 3
    )
    assert metrics[f"{rollout}/ratio_excluded_rate"] == 0.0
    assert metrics[f"{rollout}/same_direction_excluded_rate"] == 0.0

    for namespace in (token, rollout):
        category_sum = sum(
            metrics[f"{namespace}/{name}"]
            for name in (
                "category_same_direction_amplified_rate",
                "category_same_direction_reduced_rate",
                "category_sign_flipped_rate",
                "category_approximately_unchanged_rate",
            )
        )
        assert category_sum == pytest.approx(1.0)
    assert metrics["sol-opd/train/reverse_kl_estimate"] == 0.2
    assert metrics["sol-opd/train/solution_teacher_sampled_token_prob_mean"] == 0.3
    assert metrics["sol-opd/train/unprivileged_teacher_sampled_token_prob_mean"] == 0.2
    assert metrics["sol-opd/train/policy_loss"] == 0.4
    assert not any("sol_stats" in key for key in metrics)
    assert not any(key.startswith("pri-opd/") for key in metrics)


def _sol_loss_config():
    return SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="sol_reverse_kl",
            selection_ratio=1.0,
            selection_method="random",
            sol_opd_epsilon=1.0e-6,
            loss_max_clamp=20.0,
            use_policy_gradient=True,
            policy_loss_mode="reinforce",
            global_batch_info={},
        )
    )


def test_sol_loss_uses_solution_teacher_for_training_and_control_only_for_stats(
    monkeypatch: pytest.MonkeyPatch,
):
    from verl.trainer.distillation import losses

    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    student = torch.tensor(
        [[-2.0, -1.0, -4.0], [-1.0, -3.0, -2.0]], requires_grad=True
    )
    solution_teacher = torch.tensor(
        [[-1.0, -2.0, -2.0], [-2.0, -2.0, -4.0]]
    )
    unprivileged_teacher = torch.tensor(
        [[-1.5, -1.5, -3.0], [-0.5, -3.5, -1.0]]
    )
    response_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )
    data = {
        "teacher_logprobs": solution_teacher.unsqueeze(-1),
        "sol_unprivileged_teacher_logprobs": unprivileged_teacher.unsqueeze(-1),
        "response_mask": response_mask,
    }

    reverse_kl, metrics = losses.compute_solution_privileged_reverse_kl(
        None, _sol_loss_config(), {"log_probs": student}, data
    )

    expected = (student.detach() - solution_teacher) * response_mask
    torch.testing.assert_close(reverse_kl.detach(), expected)
    assert reverse_kl.requires_grad
    assert metrics["distillation/reverse_kl_estimate"].aggregate() == pytest.approx(
        expected[response_mask].mean().item()
    )
    assert "distillation/sol_stats/deviation_abs_sum" in metrics

    alternate_data = dict(data)
    alternate_data["sol_unprivileged_teacher_logprobs"] = (
        unprivileged_teacher - 10.0
    ).unsqueeze(-1)
    alternate, alternate_metrics = losses.compute_solution_privileged_reverse_kl(
        None, _sol_loss_config(), {"log_probs": student}, alternate_data
    )
    torch.testing.assert_close(alternate, reverse_kl)
    assert (
        alternate_metrics[
            "distillation/sol_stats/opd_advantage_sum"
        ].aggregate()
        != metrics["distillation/sol_stats/opd_advantage_sum"].aggregate()
    )


def test_sol_reinforce_backward_has_finite_masked_gradients_and_detached_advantage(
    monkeypatch: pytest.MonkeyPatch,
):
    from verl.trainer.distillation import losses

    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    solution_teacher = torch.tensor(
        [[-1.0, -2.0, -2.0], [-2.0, -2.0, -4.0]]
    )
    response_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )
    actor_config = SimpleNamespace(
        loss_agg_mode="token-mean", global_batch_info={}
    )

    def run(control_shift: float):
        student = torch.tensor(
            [[-2.0, -1.0, -4.0], [-1.0, -3.0, -2.0]],
            requires_grad=True,
        )
        data = {
            "teacher_logprobs": solution_teacher.unsqueeze(-1),
            "sol_unprivileged_teacher_logprobs": (
                solution_teacher + control_shift
            ).unsqueeze(-1),
            "response_mask": response_mask,
            "old_log_probs": student.detach().clone(),
        }
        scalar, _ = losses.distillation_loss(
            actor_config,
            _sol_loss_config(),
            {"log_probs": student},
            data,
        )
        scalar.backward()
        return scalar.detach(), student.grad.detach().clone()

    loss_a, grad_a = run(0.0)
    loss_b, grad_b = run(100.0)
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(grad_a, grad_b)
    assert bool(torch.isfinite(grad_a).all())
    assert bool((grad_a[response_mask] != 0).all())
    assert torch.equal(grad_a[~response_mask], torch.zeros_like(grad_a[~response_mask]))


def test_remove_padding_preserves_both_teacher_streams_and_causal_response_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    from tensordict import TensorDict
    from verl.workers.utils import padding

    def cpu_unpad_input(values, attention_mask):
        indices = attention_mask.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        flat = values.flatten(0, 1)
        unpadded = flat.index_select(0, indices)
        lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        cumulative = torch.zeros(
            lengths.numel() + 1, dtype=torch.int32, device=lengths.device
        )
        cumulative[1:] = torch.cumsum(lengths, dim=0)
        return unpadded, indices, cumulative, int(lengths.max()), None

    monkeypatch.setattr(padding, "unpad_input", cpu_unpad_input)
    monkeypatch.setattr(
        padding,
        "index_first_axis",
        lambda values, indices: values.index_select(0, indices),
    )

    prompts = torch.tensor([[0, 11, 12], [31, 32, 33]])
    responses = torch.tensor([[21, 22, 0], [41, 42, 43]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    attention_mask = torch.tensor(
        [[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    input_ids = torch.cat((prompts, responses), dim=-1)
    position_ids = torch.tensor(
        [[0, 0, 1, 2, 3, 0], [0, 1, 2, 3, 4, 5]], dtype=torch.long
    )

    primary_ids = torch.zeros((2, 6, 1), dtype=torch.int32)
    primary = torch.zeros((2, 6, 1), dtype=torch.float32)
    control = torch.zeros((2, 6, 1), dtype=torch.float32)
    # Causal row p-1 predicts response token p. The first example has one
    # left-padded prompt slot and one right-padded response slot.
    primary_ids[0, 2:4, 0] = torch.tensor([21, 22])
    primary[0, 2:4, 0] = torch.tensor([-1.0, -2.0])
    control[0, 2:4, 0] = torch.tensor([-1.5, -2.5])
    primary_ids[1, 2:5, 0] = torch.tensor([41, 42, 43])
    primary[1, 2:5, 0] = torch.tensor([-3.0, -4.0, -5.0])
    control[1, 2:5, 0] = torch.tensor([-3.5, -4.5, -5.5])

    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "teacher_ids": primary_ids,
            "teacher_logprobs": primary,
            "sol_unprivileged_teacher_logprobs": control,
        },
        batch_size=[2],
    )
    converted = padding.left_right_2_no_padding(data)

    assert converted["teacher_logprobs"].is_nested
    assert converted["sol_unprivileged_teacher_logprobs"].is_nested
    primary_response = padding.no_padding_2_padding(
        converted["teacher_logprobs"], converted
    ).squeeze(-1)
    control_response = padding.no_padding_2_padding(
        converted["sol_unprivileged_teacher_logprobs"], converted
    ).squeeze(-1)
    torch.testing.assert_close(
        primary_response,
        torch.tensor([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]]),
    )
    torch.testing.assert_close(
        control_response,
        torch.tensor([[-1.5, -2.5, 0.0], [-3.5, -4.5, -5.5]]),
    )
    sampled_ids = padding.no_padding_2_padding(
        converted["teacher_ids"], converted
    ).squeeze(-1)
    torch.testing.assert_close(sampled_ids.long(), responses.long())


def test_minimal_batch_padding_resizes_both_sol_teacher_streams():
    from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict
    from verl.workers.utils.padding import no_padding_2_padding

    sequence_length = 5
    response_length = 3
    source = {
        "prompts": torch.tensor([11, 12], dtype=torch.int64),
        "responses": torch.tensor([21, 22, 23], dtype=torch.int64),
        "input_ids": torch.tensor([11, 12, 21, 22, 23], dtype=torch.int64),
        "attention_mask": torch.ones(sequence_length, dtype=torch.int64),
        "position_ids": torch.arange(sequence_length, dtype=torch.int64),
        "response_mask": torch.ones(response_length, dtype=torch.int64),
        "loss_mask": torch.ones(response_length, dtype=torch.int64),
        "rm_scores": torch.zeros(response_length, dtype=torch.float32),
        "rollout_log_probs": torch.zeros(response_length, dtype=torch.float32),
        "teacher_ids": torch.zeros((sequence_length, 1), dtype=torch.int32),
        "teacher_logprobs": torch.full(
            (sequence_length, 1), -1.0, dtype=torch.float32
        ),
        "sol_unprivileged_teacher_logprobs": torch.full(
            (sequence_length, 1), -2.0, dtype=torch.float32
        ),
        "oa_opd_weights": torch.ones(response_length, dtype=torch.float32),
        "num_turns": 1,
    }
    padding_sample, padding_tag = construct_minimal_padding_template(
        source,
        {"prompt_len": 2, "response_len": 3, "seq_len": 5},
        eos_token_id=0,
    )

    assert padding_tag["is_padding"] is True
    assert padding_sample["teacher_logprobs"].shape == (2, 1)
    assert padding_sample["sol_unprivileged_teacher_logprobs"].shape == (2, 1)
    assert padding_sample["oa_opd_weights"].shape == (1,)
    assert not bool(padding_sample["teacher_logprobs"].any())
    assert not bool(padding_sample["sol_unprivileged_teacher_logprobs"].any())
    assert not bool(padding_sample["oa_opd_weights"].any())

    combined = list_of_dict_to_tensordict([source, padding_sample])
    solution_response = no_padding_2_padding(
        combined["teacher_logprobs"], combined
    )
    control_response = no_padding_2_padding(
        combined["sol_unprivileged_teacher_logprobs"], combined
    )
    assert solution_response.shape == control_response.shape == (2, 3, 1)
    assert not bool(solution_response[1].any())
    assert not bool(control_response[1].any())


class _FakeTeacherManager:
    def __init__(self, response_ids: list[int], diagnostic_topk: int):
        self.response_ids = list(response_ids)
        self.diagnostic_topk = diagnostic_topk
        self.calls: list[dict] = []

    def get_teacher_prompt_length(self, routing_key=None) -> int:
        del routing_key
        return 8192

    async def compute_teacher_logprobs_single(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["sequence_ids"][-len(self.response_ids) :] == self.response_ids
        prompt_length = int(kwargs["student_prompt_length"])
        response_length = int(kwargs["response_length"])
        sequence_length = prompt_length + response_length
        causal_start = prompt_length - 1
        sampled_ids = torch.zeros((sequence_length, 1), dtype=torch.int32)
        sampled_ids[causal_start : causal_start + response_length, 0] = (
            torch.tensor(self.response_ids, dtype=torch.int32)
        )
        value = -0.5 if len(self.calls) == 1 else -1.5
        sampled_logprobs = torch.zeros((sequence_length, 1), dtype=torch.float32)
        sampled_logprobs[
            causal_start : causal_start + response_length, 0
        ] = value
        topk_ids = sampled_ids.expand(-1, self.diagnostic_topk).clone()
        topk_logprobs = torch.full(
            (sequence_length, self.diagnostic_topk),
            -2.0,
            dtype=torch.float32,
        )
        topk_logprobs[
            causal_start : causal_start + response_length, 0
        ] = -0.1
        return (
            sampled_ids,
            sampled_logprobs,
            topk_ids,
            topk_logprobs,
            sampled_logprobs.clone(),
            None,
            {"teacher_engine_s": 0.0, "teacher_logprob_extract_s": 0.0},
        )


class _FakeStudentClient:
    def __init__(self, response_ids: list[int], diagnostic_topk: int):
        self.response_ids = list(response_ids)
        self.diagnostic_topk = diagnostic_topk

    async def generate(self, *, prompt_ids, **kwargs):
        del kwargs
        sequence_length = len(prompt_ids)
        response_length = len(self.response_ids)
        causal_start = sequence_length - response_length - 1
        ids = torch.zeros(
            (sequence_length, self.diagnostic_topk), dtype=torch.int32
        )
        ids[causal_start : causal_start + response_length, :] = torch.tensor(
            self.response_ids, dtype=torch.int32
        ).unsqueeze(-1)
        logprobs = torch.full(
            (sequence_length, self.diagnostic_topk), -2.0, dtype=torch.float32
        )
        logprobs[causal_start : causal_start + response_length, 0] = -0.1
        return SimpleNamespace(
            extra_fields={
                "prompt_ids": ids.tolist(),
                "prompt_logprobs": logprobs.tolist(),
            }
        )


def test_dual_teacher_forwards_score_the_same_original_student_response_ids():
    from utils.prompts import render_prompt
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
        AgentLoopWorker,
    )
    from verl.experimental.teacher_loop.teacher_manager import (
        _slice_response_prediction_outputs,
    )

    tokenizer = CharacterTokenizer()
    question = "Compute 20 + 22."
    solution = r"Use addition and conclude \boxed{42}."
    student_prompt = render_prompt("qwen3_thinking_prompt", question=question)
    solution_prompt = render_prompt(
        "qwen3_privileged_solution_thinking",
        question=question,
        privileged_solution=solution,
    )
    prompt_ids = tokenizer.encode(student_prompt, add_special_tokens=False)
    response_ids = [7001, 7002, 7003]
    diagnostic_topk = 2

    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "data_source"
    worker.tokenizer = tokenizer
    worker.config = OmegaConf.create(
        {
            "algorithm": {"name": "sol_opd"},
            "teacher_prompt": "qwen3_privileged_solution_thinking",
            "distillation": {
                "distillation_loss": {"diagnostic_topk": diagnostic_topk}
            },
        }
    )
    manager = _FakeTeacherManager(response_ids, diagnostic_topk)
    worker.teacher_server_manager = manager
    worker.llm_client = _FakeStudentClient(response_ids, diagnostic_topk)

    async def tokenize_preformatted_prompt(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    worker.tokenize_preformatted_prompt = tokenize_preformatted_prompt
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        metrics=AgentLoopMetrics(),
    )
    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=False,
            sample_kwargs={
                "data_source": "math",
                "teacher_prompt_text": solution_prompt,
                "sol_question": question,
                "sol_privileged_solution": solution,
            },
        )
    )

    assert len(manager.calls) == 2
    privileged_call, control_call = manager.calls
    assert not privileged_call.get("sampled_only", False)
    assert control_call["sampled_only"] is True
    assert privileged_call["sequence_ids"][-3:] == response_ids
    assert control_call["sequence_ids"][-3:] == response_ids
    privileged_prefix = privileged_call["sequence_ids"][:-3]
    control_prefix = control_call["sequence_ids"][:-3]
    assert privileged_prefix != control_prefix
    assert tokenizer.decode(privileged_prefix) == solution_prompt
    assert control_prefix == prompt_ids
    for call in manager.calls:
        assert call["student_prompt_length"] == len(prompt_ids)
        assert call["response_length"] == len(response_ids)
        assert call["routing_key"] == "math"

    primary_ids = _slice_response_prediction_outputs(
        output.extra_fields["teacher_ids"], len(response_ids)
    ).squeeze(-1)
    control_ids = _slice_response_prediction_outputs(
        output.extra_fields["sol_unprivileged_teacher_ids"], len(response_ids)
    ).squeeze(-1)
    torch.testing.assert_close(primary_ids.long(), torch.tensor(response_ids))
    torch.testing.assert_close(control_ids.long(), torch.tensor(response_ids))
    assert "teacher_logprobs" in output.extra_fields
    assert "sol_unprivileged_teacher_logprobs" in output.extra_fields


def test_sol_training_teacher_forward_rejects_missing_solution():
    from utils.prompts import render_prompt
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
        AgentLoopWorker,
    )

    tokenizer = CharacterTokenizer()
    question = "Compute 1 + 1."
    prompt_text = render_prompt("qwen3_thinking_prompt", question=question)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = [91, 92]
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_key = "data_source"
    worker.tokenizer = tokenizer
    worker.config = OmegaConf.create(
        {
            "algorithm": {"name": "sol_opd"},
            "teacher_prompt": "qwen3_privileged_solution_thinking",
            "distillation": {"distillation_loss": {"diagnostic_topk": 2}},
        }
    )
    manager = _FakeTeacherManager(response_ids, 2)
    worker.teacher_server_manager = manager
    worker.llm_client = _FakeStudentClient(response_ids, 2)

    async def tokenize_preformatted_prompt(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    worker.tokenize_preformatted_prompt = tokenize_preformatted_prompt
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
    )
    with pytest.raises(RuntimeError, match="solution"):
        asyncio.run(
            worker._compute_teacher_logprobs(
                output,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                validate=False,
                sample_kwargs={
                    "data_source": "math",
                    "teacher_prompt_text": prompt_text,
                    "sol_question": question,
                    "sol_privileged_solution": "",
                },
            )
        )


def test_agent_loop_output_promotes_sol_control_logprobs_without_duplicate_ids():
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopMetrics,
        AgentLoopOutput,
    )

    control_ids = torch.tensor([[0], [101], [102], [0]], dtype=torch.int32)
    control_logprobs = torch.tensor(
        [[0.0], [-1.0], [-2.0], [0.0]], dtype=torch.float32
    )
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[101, 102],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "sol_unprivileged_teacher_ids": control_ids,
            "sol_unprivileged_teacher_logprobs": control_logprobs,
        },
    ).as_dict()

    torch.testing.assert_close(
        output["sol_unprivileged_teacher_logprobs"], control_logprobs
    )
    assert "sol_unprivileged_teacher_ids" not in output
    assert "sol_unprivileged_teacher_logprobs" not in output["extra_fields"]
