"""CPU-only regression tests for EOPD and its six production configs."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from algorithms.eopd import validate_eopd_config
from verl.trainer.distillation import losses
from verl.trainer.distillation.eopd import compute_entropy_gated_forward_kl
from verl.workers.config import DistillationLossConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "EOPD"
LOSS_DEFAULTS = OmegaConf.load(
    PROJECT_ROOT
    / "verl"
    / "verl"
    / "trainer"
    / "config"
    / "distillation"
    / "distillation.yaml"
).distillation_loss


def _load_config(path: Path):
    config = OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml"),
        OmegaConf.load(path),
    )
    config.distillation.distillation_loss = OmegaConf.merge(
        LOSS_DEFAULTS, config.distillation.distillation_loss
    )
    return config


def test_eopd_forward_kl_normalizes_only_teacher_and_uses_strict_gate():
    student_log_probs = torch.log(
        torch.tensor(
            [
                [[0.20, 0.10]],
                [[0.25, 0.05]],
            ],
            requires_grad=True,
        )
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [[0.40, 0.30]],
                [[0.50, 0.20]],
            ]
        )
    )
    teacher_entropy = torch.tensor([[0.8], [0.8001]])

    result = compute_entropy_gated_forward_kl(
        student_log_probs,
        teacher_log_probs,
        teacher_entropy,
        entropy_threshold=0.8,
    )

    normalized_teacher = torch.tensor([0.50, 0.20]) / 0.70
    expected_second = (
        normalized_teacher
        * (normalized_teacher.log() - torch.log(torch.tensor([0.25, 0.05])))
    ).sum()
    torch.testing.assert_close(result, torch.tensor([[0.0], [expected_second]]))

def test_eopd_forward_kl_gradient_does_not_renormalize_student():
    student_log_probs = torch.tensor(
        [[[-1.4, -2.3]]], requires_grad=True
    )
    teacher_log_probs = torch.log(torch.tensor([[[0.4, 0.3]]]))
    result = compute_entropy_gated_forward_kl(
        student_log_probs,
        teacher_log_probs,
        torch.tensor([[1.0]]),
        entropy_threshold=0.8,
    )

    result.sum().backward()
    expected_teacher = torch.tensor([[[-4.0 / 7.0, -3.0 / 7.0]]])
    torch.testing.assert_close(student_log_probs.grad, expected_teacher)


def test_chunked_student_projection_returns_sampled_opd_logprobs_and_gated_fkl():
    from verl.models.transformers.dense_common import _chunked_forward_kl_topk

    hidden = torch.tensor(
        [[[0.2, -0.1, 0.3], [0.4, 0.2, -0.2]]], requires_grad=True
    )
    vocab_weights = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.3, -0.2, 0.1],
            [-0.1, 0.4, 0.2],
            [0.2, 0.1, -0.3],
        ],
        requires_grad=True,
    )
    teacher_ids = torch.tensor([[[0, 1], [2, 3]]])
    teacher_logprobs = torch.log(
        torch.tensor([[[0.4, 0.3], [0.5, 0.2]]])
    )
    labels = torch.tensor([[1, 3]])
    teacher_entropy = torch.tensor([[0.8, 0.9]])

    outputs = _chunked_forward_kl_topk(
        hidden,
        vocab_weights,
        teacher_ids,
        teacher_logprobs,
        temperature=1.0,
        chunk_size=1,
        log_prob_min_clamp=None,
        shift_labels=labels,
        teacher_entropy=teacher_entropy,
        eopd_entropy_threshold=0.8,
    )

    full_student_logprobs = torch.nn.functional.log_softmax(
        torch.nn.functional.linear(hidden, vocab_weights), dim=-1
    )
    expected_sampled = full_student_logprobs.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    expected_fkl = compute_entropy_gated_forward_kl(
        full_student_logprobs.gather(-1, teacher_ids),
        teacher_logprobs,
        teacher_entropy,
        entropy_threshold=0.8,
    )
    torch.testing.assert_close(outputs[0], expected_fkl)
    torch.testing.assert_close(outputs[5], expected_sampled)

    (outputs[0].sum() + outputs[5].sum()).backward()
    assert hidden.grad is not None
    assert vocab_weights.grad is not None


def test_vllm_full_vocab_extractor_returns_exact_entropy_and_only_topk():
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    probabilities = [0.5, 0.3, 0.15, 0.05]
    expected_entropy = -sum(p * math.log(p) for p in probabilities)
    entries = {
        11: SimpleNamespace(logprob=math.log(0.5), rank=1),
        12: SimpleNamespace(logprob=math.log(0.3), rank=2),
        # The bounded worker output uses rank topk + 1 as its entropy carrier.
        13: SimpleNamespace(logprob=expected_entropy, rank=3),
    }
    output = SimpleNamespace(
        prompt_token_ids=[10, 11],
        prompt_logprobs=[None, entries],
    )
    extracted = {}

    extract_prompt_logprobs(
        output,
        num_prompt_logprobs=3,
        result_dict=extracted,
        prompt_logprobs_topk=2,
    )

    assert extracted["prompt_ids"] == [[11, 12], [0, 0]]
    assert len(extracted["prompt_logprobs"][0]) == 2
    assert extracted["prompt_entropies"][0][0] == pytest.approx(expected_entropy)
    assert extracted["prompt_entropies"][1] == [0.0]
    assert extracted["prompt_sampled_logprobs"][0][0] == pytest.approx(
        math.log(0.5)
    )


def test_vllm_eopd_gather_computes_entropy_before_bounded_topk_transfer():
    from vllm.v1.outputs import LogprobsTensors

    from verl.workers.rollout.vllm_rollout.utils import enable_eopd_entropy_gather

    class FakeSampler:
        @staticmethod
        def gather_logprobs(logprobs, num_logprobs, token_ids):
            top_logprobs, top_ids = torch.topk(logprobs, num_logprobs, dim=-1)
            sampled = token_ids.unsqueeze(-1)
            return LogprobsTensors(
                logprob_token_ids=torch.cat((sampled, top_ids), dim=-1).to(
                    torch.int32
                ),
                logprobs=torch.cat(
                    (logprobs.gather(-1, sampled), top_logprobs), dim=-1
                ),
                selected_token_ranks=torch.ones_like(token_ids),
            )

    sampler = FakeSampler()
    enable_eopd_entropy_gather(sampler, topk=2)
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    gathered = sampler.gather_logprobs(
        probabilities.log(), 3, torch.tensor([0], dtype=torch.int64)
    )

    assert gathered.logprob_token_ids.shape == (1, 4)
    torch.testing.assert_close(
        gathered.logprobs[0, 1:3], probabilities.log()[0, :2]
    )
    expected_entropy = torch.special.entr(probabilities).sum()
    torch.testing.assert_close(gathered.logprobs[0, -1], expected_entropy)
    assert gathered.selected_token_ranks.item() == 4


def test_eopd_combines_existing_opd_pg_and_direct_forward_kl(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)

    def fake_policy_loss(**kwargs):
        mask = kwargs["response_mask"].to(kwargs["log_prob"].dtype)
        value = -(
            kwargs["advantages"] * kwargs["log_prob"] * mask
        ).sum() / mask.sum()
        return value, {}

    def fake_agg_loss(*, loss_mat, loss_mask, **_kwargs):
        mask = loss_mask.to(loss_mat.dtype)
        return (loss_mat * mask).sum() / mask.sum()

    monkeypatch.setattr(losses, "get_policy_loss_fn", lambda _name: fake_policy_loss)
    monkeypatch.setattr(losses, "agg_loss", fake_agg_loss)

    student = torch.tensor([[-2.0, -1.0]], requires_grad=True)
    forward_kl = torch.tensor([[0.3, 0.0]], requires_grad=True)
    data = {
        "teacher_sampled_logprobs": torch.tensor([[[-1.0], [-2.0]]]),
        "teacher_entropy": torch.tensor([[[1.0], [0.2]]]),
        "response_mask": torch.ones((1, 2), dtype=torch.bool),
        "old_log_probs": student.detach().clone(),
    }
    actor_config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={},
    )
    loss_config = SimpleNamespace(
        loss_mode="eopd",
        loss_max_clamp=20.0,
        use_policy_gradient=True,
        policy_loss_mode="reinforce",
        global_batch_info={},
        eopd_entropy_threshold=0.8,
        eopd_alpha=2.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    total, metrics = losses.distillation_loss(
        actor_config,
        distillation_config,
        {"log_probs": student, "distillation_losses": forward_kl},
        data,
    )

    assert total.item() == pytest.approx(0.8)
    assert "distillation/eopd_forward_kl_loss" in metrics
    total.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[-0.5, 0.5]]))
    torch.testing.assert_close(forward_kl.grad, torch.tensor([[1.0, 1.0]]))


def test_six_eopd_configs_have_models_lengths_defaults_and_persistence():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == 6
    expected_pairs = {
        ("Qwen3-4B", "Qwen3-1.7B"),
        ("Qwen3-8B", "Qwen3-1.7B"),
        ("Qwen3-4B", "Qwen3-0.6B"),
    }
    observed_pairs = set()
    observed_modes = {"thinking": 0, "no_thinking": 0}

    for path in paths:
        config = _load_config(path)
        teacher_name = Path(
            str(config.distillation.teacher_models.teacher_model.model_path)
        ).name
        student_name = Path(str(config.actor_rollout_ref.model.path)).name
        observed_pairs.add((teacher_name, student_name))

        thinking = str(config.student_prompt) == "qwen3_thinking_prompt"
        observed_modes["thinking" if thinking else "no_thinking"] += 1
        assert config.teacher_prompt == config.student_prompt
        assert config.rlvr_generation.train_max_new_tokens == (
            8192 if thinking else 4096
        )
        assert config.rlvr_generation.val_max_new_tokens == (
            32768 if thinking else 16384
        )
        assert config.data.max_response_length == (32768 if thinking else 16384)

        loss = config.distillation.distillation_loss
        assert config.algorithm.name == "eopd"
        assert config.group_name == "EOPD"
        assert loss.loss_mode == "eopd"
        assert loss.topk == 16
        assert loss.eopd_entropy_threshold == pytest.approx(0.8)
        assert loss.eopd_alpha == pytest.approx(1.0)
        vllm_kwargs = config.distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm
        assert vllm_kwargs.max_logprobs == 17
        assert vllm_kwargs.eopd_entropy_topk == 16
        assert config.distillation.teacher_chunk_size == 128
        assert config.trainer.total_training_steps == 100
        assert config.trainer.test_freq == 20
        assert config.trainer.save_freq == 20
        validate_eopd_config(config)

    assert observed_pairs == expected_pairs
    assert observed_modes == {"thinking": 3, "no_thinking": 3}

    instantiated = DistillationLossConfig(
        loss_mode="eopd",
        topk=16,
        eopd_entropy_threshold=0.8,
        eopd_alpha=1.0,
    )
    assert instantiated.loss_settings.use_topk
    assert instantiated.loss_settings.use_full_vocab_teacher_entropy
