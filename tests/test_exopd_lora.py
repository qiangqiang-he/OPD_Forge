"""CPU regression tests for the LoRA-backed ExOPD reference policy.

These tests deliberately use only tiny in-memory models and worker doubles.  They
exercise the same configuration and controller/worker branches as production
without weakening the four-Student/four-Teacher production layout.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_ExOPD_Thinking"
PUBLICATION_CONFIGS = tuple(sorted(CONFIG_DIR.glob("*.yaml")))


def test_all_five_publication_exopd_configs_are_discovered():
    """Avoid a vacuous parametrized pass when the publication path changes."""

    assert len(PUBLICATION_CONFIGS) == 5


def _compose_publication_config(path: Path):
    from hydra import compose, initialize_config_dir

    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=path.stem, overrides=[search_path])


@pytest.mark.parametrize("config_path", PUBLICATION_CONFIGS, ids=lambda path: path.stem)
def test_publication_exopd_uses_rank64_all_dense_lora(config_path: Path):
    """Every publication run uses one sharded base plus a rank-64 adapter."""

    from algorithms.exopd import validate_exopd_config
    from verl.trainer.ppo.utils import need_reference_policy

    config = _compose_publication_config(config_path)
    model = config.actor_rollout_ref.model
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout

    assert config.algorithm.name == "exopd"
    assert config.group_name == "PUB_ExOPD_Thinking"
    assert str(config.run_name).endswith("_lora_r64")
    assert str(config.run_name_prefix).endswith("_lora_r64")
    assert int(config.trainer.n_gpus_per_node) == 4
    assert int(config.distillation.n_gpus_per_node) == 4
    assert (
        int(config.trainer.n_gpus_per_node)
        + int(config.distillation.n_gpus_per_node)
        == 8
    )

    # FSDP/PEFT consumes the top-level HF-model LoRA fields.  Keep the nested
    # Megatron-style rank disabled so there is a single unambiguous source.
    assert int(model.lora_rank) == 64
    assert int(model.lora_alpha) == 128
    assert str(model.target_modules) == "all-linear"
    assert model.exclude_modules is None
    assert int(model.lora.rank) == 0
    assert not bool(model.lora.merge)

    assert str(actor.strategy) == "fsdp"
    assert int(actor.fsdp_config.fsdp_size) == 4
    assert not bool(actor.fsdp_config.param_offload)
    assert not bool(actor.fsdp_config.optimizer_offload)

    assert bool(rollout.layered_summon)
    assert str(rollout.load_format) == "safetensors"
    assert str(rollout.checkpoint_engine.backend) == "naive"
    assert need_reference_policy(config)
    validate_exopd_config(config)


def test_layered_summon_rejects_dummy_rollout_loading():
    """Layered LoRA transfer cannot start from vLLM's uninitialized dummy base."""

    from algorithms.exopd import validate_exopd_config

    config = _compose_publication_config(PUBLICATION_CONFIGS[0])
    config.actor_rollout_ref.rollout.load_format = "dummy"

    with pytest.raises(ValueError, match="layered_summon.*safetensors"):
        validate_exopd_config(config)


def test_lora_rollout_rejects_non_naive_checkpoint_engine():
    """Adapter-only rollout sync must retain the PEFT metadata."""

    from algorithms.exopd import validate_exopd_config

    config = _compose_publication_config(PUBLICATION_CONFIGS[0])
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"

    with pytest.raises(ValueError, match=r"checkpoint_engine\.backend=naive"):
        validate_exopd_config(config)


def test_adapter_path_requires_positive_top_level_fsdp_rank():
    """An adapter path alone does not cause the FSDP engine to build PEFT."""

    from algorithms.exopd import validate_exopd_config

    config = _compose_publication_config(PUBLICATION_CONFIGS[0])
    config.actor_rollout_ref.model.lora_rank = 0
    config.actor_rollout_ref.model.lora_adapter_path = "/adapter/checkpoint"

    with pytest.raises(ValueError, match="positive top-level model.lora_rank"):
        validate_exopd_config(config)


def test_exopd_advantage_matches_formula_and_stops_all_auxiliary_gradients():
    """The detached token signal may backpropagate only through log pi_S in L."""

    from verl.trainer.distillation.losses import compute_exopd_advantage

    student = torch.tensor([[-2.0, -0.5]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -1.5]], requires_grad=True)
    reference = torch.tensor([[-1.8, -1.0]], requires_grad=True)

    advantage = compute_exopd_advantage(
        student,
        teacher,
        reference,
        exopd_lambda=1.25,
    )
    expected = 1.25 * (teacher.detach() - reference.detach()) - (
        student.detach() - reference.detach()
    )

    torch.testing.assert_close(advantage, expected)
    assert advantage.shape == student.shape
    assert not advantage.requires_grad

    loss = -(advantage * student).mean()
    loss.backward()
    torch.testing.assert_close(student.grad, -advantage / advantage.numel())
    assert teacher.grad is None
    assert reference.grad is None


def test_exopd_per_token_loss_preserves_shape_and_masks_metrics(monkeypatch):
    """Padding stays token-aligned and cannot contaminate reported statistics."""

    from verl.trainer.distillation import losses

    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    student = torch.tensor(
        [[-2.0, -0.5, 40.0, -1.25], [-1.5, 50.0, -0.75, 60.0]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[-1.0], [-1.5], [-40.0], [-0.5]], [[-1.0], [-50.0], [-1.5], [-60.0]]]
    )
    reference = torch.tensor(
        [[-1.8, -1.0, 30.0, -1.0], [-1.2, 45.0, -1.0, 55.0]]
    )
    response_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, False]]
    )
    data = {
        "teacher_logprobs": teacher,
        "ref_log_prob": reference,
        "response_mask": response_mask,
    }
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(exopd_lambda=1.25)
    )

    per_token_loss, metrics = losses.compute_exopd_sampled_token_loss(
        None,
        config,
        {"log_probs": student},
        data,
    )
    expected_advantage = 1.25 * (teacher.squeeze(-1) - reference) - (
        student.detach() - reference
    )

    assert per_token_loss.shape == response_mask.shape == student.shape
    assert per_token_loss.dtype == student.dtype
    assert not per_token_loss.requires_grad
    torch.testing.assert_close(per_token_loss, -expected_advantage)

    valid_advantage = expected_advantage[response_mask]
    advantage_mean = metrics["distillation/exopd_advantage_mean"].aggregate()
    advantage_abs_mean = metrics[
        "distillation/exopd_advantage_abs_mean"
    ].aggregate()
    assert float(advantage_mean) == pytest.approx(float(valid_advantage.mean()))
    assert float(advantage_abs_mean) == pytest.approx(
        float(valid_advantage.abs().mean())
    )
    assert "distillation/exopd_reference_sampled_token_prob" in metrics

    # This is the exact token mask used by the downstream token-mean reducer.
    masked_policy_loss = -(expected_advantage[response_mask] * student[response_mask]).mean()
    masked_policy_loss.backward()
    assert student.grad is not None
    assert bool((student.grad[~response_mask] == 0).all())
    assert bool(torch.isfinite(student.grad[response_mask]).all())


def _minimal_exopd_role_config(*, lora_rank: int):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    # Exercise the top-level fallback used by FSDP configs.
                    "lora_rank": lora_rank,
                    "lora": {"rank": 0},
                    "lora_adapter_path": None,
                },
                "actor": {"use_kl_loss": False},
            },
            "algorithm": {"use_kl_in_reward": False},
            "distillation": {
                "enabled": True,
                "distillation_loss": {"loss_mode": "exopd_reverse_kl"},
            },
        }
    )


@pytest.mark.parametrize(
    ("lora_rank", "expected_role_name"),
    ((64, "ActorRollout"), (0, "ActorRolloutRef")),
)
def test_task_runner_selects_in_actor_reference_role(
    monkeypatch, lora_rank: int, expected_role_name: str
):
    """LoRA ExOPD must not request a nonexistent standalone reference worker."""

    from verl.trainer import main_ppo_sync as sync

    # Ray retains the undecorated class as the parent of its generated actor
    # class.  Calling it locally lets us test role construction without a Ray
    # cluster or GPUs.
    runner_class = sync.TaskRunner.__ray_metadata__.modified_class.__mro__[1]
    runner = runner_class()
    monkeypatch.setattr(sync.ray, "remote", lambda worker_class: worker_class)

    runner.add_actor_rollout_worker(
        _minimal_exopd_role_config(lora_rank=lora_rank)
    )

    assert len(runner.role_worker_mapping) == 1
    (role,) = runner.role_worker_mapping
    assert role.name == expected_role_name
    assert runner.mapping[role] == "global_pool"


@pytest.mark.parametrize("config_path", PUBLICATION_CONFIGS, ids=lambda path: path.stem)
def test_publication_task_runner_builds_four_plus_four_resource_pools(
    monkeypatch,
    config_path: Path,
):
    """Exercise the formal TaskRunner placement contract, not only YAML scalars."""

    from verl.trainer import main_ppo_sync as sync
    from verl.trainer.ppo.utils import Role

    runner_class = sync.TaskRunner.__ray_metadata__.modified_class.__mro__[1]
    runner = runner_class()
    monkeypatch.setattr(sync.ray, "remote", lambda worker_class: worker_class)
    config = _compose_publication_config(config_path)

    runner.add_actor_rollout_worker(config)
    runner.init_resource_pool_mgr(config)

    assert runner.resource_pool_manager.resource_pool_spec == {
        "global_pool": [4],
        "teacher_pool": [4],
    }
    assert runner.resource_pool_manager.get_n_gpus() == 8
    assert runner.mapping[Role.ActorRollout] == "global_pool"
    assert runner.mapping[Role.TeacherModel] == "teacher_pool"
    assert Role.ActorRolloutRef not in runner.role_worker_mapping


def test_init_workers_reuses_actor_group_for_lora_reference(monkeypatch):
    """Regression test for a KeyError on the absent ActorRolloutRef group."""

    from verl.trainer import main_ppo_sync as sync
    from verl.trainer.ppo.utils import Role

    config = _compose_publication_config(PUBLICATION_CONFIGS[0])
    actor_group = SimpleNamespace(
        init_model=lambda: None,
    )
    pool = object()

    class FakeResourcePoolManager:
        resource_pool_dict = {"global_pool": pool}

        def create_resource_pool(self):
            return None

        def get_resource_pool(self, _role):
            return pool

    class FakeRayWorkerGroup:
        def __init__(self, **_kwargs):
            pass

        def spawn(self, prefix_set):
            assert set(prefix_set) == {str(Role.ActorRollout)}
            return {str(Role.ActorRollout): actor_group}

    class FakeRewardLoopManager:
        reward_loop_workers = None

        def __init__(self, **_kwargs):
            pass

    class FakeLLMServerManager:
        @classmethod
        def create(cls, **_kwargs):
            return cls()

        def get_client(self):
            return object()

        def get_replicas(self):
            return []

    class FakeAgentLoopManager:
        @classmethod
        def create(cls, **_kwargs):
            return cls()

    class FakeCheckpointManager:
        def __init__(self, **_kwargs):
            pass

        def sleep_replicas(self):
            return None

    monkeypatch.setattr(sync, "RayClassWithInitArgs", lambda **kwargs: kwargs)
    monkeypatch.setattr(sync, "create_colocated_worker_cls", lambda **_kwargs: object)
    monkeypatch.setattr(sync, "RayWorkerGroup", FakeRayWorkerGroup)
    monkeypatch.setattr(sync, "RewardLoopManager", FakeRewardLoopManager)
    monkeypatch.setattr(sync, "LLMServerManager", FakeLLMServerManager)
    monkeypatch.setattr(sync, "AgentLoopManagerTQ", FakeAgentLoopManager)
    monkeypatch.setattr(sync, "CheckpointEngineManager", FakeCheckpointManager)
    monkeypatch.setattr(sync, "omega_conf_to_dataclass", lambda value: value)

    trainer = object.__new__(sync.PPOTrainer)
    trainer.config = config
    trainer.resource_pool_manager = FakeResourcePoolManager()
    trainer.role_worker_mapping = {Role.ActorRollout: object}
    trainer.use_critic = False
    trainer.use_reference_policy = True
    trainer.use_teacher_policy = False
    trainer.replay_buffer = object()

    # The old implementation indexed all_wg[ActorRolloutRef] here and failed.
    trainer.init_workers()

    assert trainer.ref_in_actor
    assert trainer.actor_rollout_wg is actor_group
    assert trainer.ref_policy_wg is actor_group


@pytest.mark.parametrize("ref_in_actor", [True, False])
def test_shared_reference_does_not_profile_the_actor_group_twice(ref_in_actor: bool):
    """A shared Actor/Reference worker has exactly one profiler lifecycle."""

    from verl.trainer import main_ppo_sync as sync

    class FakeActorGroup:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def start_profile(self, **_kwargs):
            self.started += 1

        def stop_profile(self):
            self.stopped += 1

    actor_group = FakeActorGroup()
    trainer = object.__new__(sync.PPOTrainer)
    trainer.actor_rollout_wg = actor_group
    trainer.ref_policy_wg = actor_group
    trainer.use_reference_policy = True
    trainer.ref_in_actor = ref_in_actor
    trainer.use_critic = False
    trainer.global_steps = 1
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    trainer.config = SimpleNamespace(
        global_profiler=SimpleNamespace(
            profile_continuous_steps=False,
            steps=[],
        )
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    assert actor_group.started == 1
    assert actor_group.stopped == 1


def test_separate_reference_worker_has_its_own_profiler_lifecycle():
    """A genuinely separate Reference worker must still be profiled."""

    from verl.trainer import main_ppo_sync as sync

    class FakeWorkerGroup:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def start_profile(self, **_kwargs):
            self.started += 1

        def stop_profile(self):
            self.stopped += 1

    actor_group = FakeWorkerGroup()
    reference_group = FakeWorkerGroup()
    trainer = object.__new__(sync.PPOTrainer)
    trainer.actor_rollout_wg = actor_group
    trainer.ref_policy_wg = reference_group
    trainer.use_reference_policy = True
    trainer.ref_in_actor = False
    trainer.use_critic = False
    trainer.global_steps = 1
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    trainer.config = SimpleNamespace(
        global_profiler=SimpleNamespace(
            profile_continuous_steps=False,
            steps=[],
        )
    )

    trainer._start_profiling()
    trainer._stop_profiling()

    assert actor_group.started == actor_group.stopped == 1
    assert reference_group.started == reference_group.stopped == 1


class _FakeBatch:
    def __init__(self):
        self.keys = ["sample-0", "sample-1"]
        self.partition_id = 7
        self.extra_info = {}

    def __len__(self):
        return len(self.keys)


class _FakeFields(dict):
    def select(self, *keys):
        return _FakeFields({key: self[key] for key in keys})


def test_reference_flag_is_scoped_before_the_next_policy_pass(monkeypatch):
    """The reference pass disables LoRA, but the following Student pass does not."""

    from verl.trainer import main_ppo_sync as sync

    observed_adapter_flags: list[bool] = []

    class FakeActorGroup:
        def compute_log_prob(self, batch):
            observed_adapter_flags.append(
                bool(batch.extra_info.get("no_lora_adapter", False))
            )
            return [None] * len(batch)

    class ForbiddenReferenceGroup:
        def compute_ref_log_prob(self, _batch):
            raise AssertionError("LoRA ExOPD must reuse the actor worker")

    stored = {}

    def fake_get(**_kwargs):
        return _FakeFields({"log_probs": "nested-logprobs", "response_mask": "mask"})

    def fake_put(*, fields, **_kwargs):
        stored.update(fields)

    monkeypatch.setattr(sync.tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(sync.tq, "kv_batch_put", fake_put)
    monkeypatch.setattr(sync, "response_from_nested", lambda values, _mask: values)

    trainer = object.__new__(sync.PPOTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(temperature=1.0)
        )
    )
    trainer.ref_in_actor = True
    trainer.actor_rollout_wg = FakeActorGroup()
    trainer.ref_policy_wg = ForbiddenReferenceGroup()
    batch = _FakeBatch()

    trainer._compute_ref_log_prob(batch, metrics={})

    assert observed_adapter_flags == [True]
    assert "no_lora_adapter" not in batch.extra_info
    assert stored["ref_log_prob"] == "nested-logprobs"

    # This represents the immediately following old-policy call.  The same
    # controller metadata must now expose the adapter-enabled Student.
    trainer.actor_rollout_wg.compute_log_prob(batch)
    assert observed_adapter_flags == [True, False]


def test_training_worker_enters_disable_adapter_context(monkeypatch):
    """The worker consumes no_lora_adapter and scopes inference inside the context."""

    from verl.workers import engine_workers

    events: list[str] = []

    @contextlib.contextmanager
    def recorded_context(name: str):
        events.append(f"{name}:enter")
        try:
            yield
        finally:
            events.append(f"{name}:exit")

    class FakeEngine:
        def eval_mode(self, *, disable_auto_offload):
            assert not disable_auto_offload
            return recorded_context("eval")

        def disable_adapter(self):
            return recorded_context("adapter-disabled")

        def infer_batch(self, _data, *, loss_function):
            assert loss_function is None
            assert events[-1] == "adapter-disabled:enter"
            events.append("infer")
            return object()

        def is_mp_src_rank_with_outputs(self):
            return False

    missing = object()

    class FakeTensorDictUtils:
        @staticmethod
        def get(data, *, key, default=missing):
            if key in data:
                return data[key]
            if default is not missing:
                return default
            raise KeyError(key)

        @staticmethod
        def pop(data, *, key, default=missing):
            if key in data:
                return data.pop(key)
            if default is not missing:
                return default
            raise KeyError(key)

        @staticmethod
        def assign_non_tensor(data, **values):
            data.update(values)

    monkeypatch.setattr(engine_workers, "tu", FakeTensorDictUtils)
    worker = object.__new__(engine_workers.TrainingWorker)
    worker.engine = FakeEngine()
    worker.model_config = {}
    worker.engine_config = SimpleNamespace(
        use_dynamic_bsz=False,
        infer_max_token_len_per_gpu=128,
        infer_micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
    )
    worker.loss_fn = object()
    data = {
        "global_token_num": 8,
        "compute_loss": False,
        "no_lora_adapter": True,
    }

    infer_batch = inspect.unwrap(engine_workers.TrainingWorker.infer_batch)
    result = infer_batch(worker, data)

    assert result is None
    assert "no_lora_adapter" not in data
    assert events == [
        "eval:enter",
        "adapter-disabled:enter",
        "infer",
        "adapter-disabled:exit",
        "eval:exit",
    ]


@pytest.mark.skipif(
    os.environ.get("RUN_EXOPD_FSDP_GPU_SMOKE") != "1",
    reason="set RUN_EXOPD_FSDP_GPU_SMOKE=1 and launch with single-rank torchrun",
)
def test_real_fsdp1_lora_wrapper_exposes_disable_adapter():
    """The real FSDP1 wrapper must forward PEFT's adapter context to the engine."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import LlamaConfig, LlamaForCausalLM
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    owned_process_group = False
    if not dist.is_initialized():
        required = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        missing = sorted(required.difference(os.environ))
        if missing:
            pytest.fail(
                "Launch this smoke with torchrun; missing distributed variables: "
                + ", ".join(missing)
            )
        dist.init_process_group(backend="nccl")
        owned_process_group = True

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    wrapped = None
    try:
        torch.manual_seed(19)
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        )
        peft_model = get_peft_model(
            base,
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if parameter.requires_grad and "lora_B" in name:
                    parameter.fill_(0.01)

        auto_wrap_policy = get_fsdp_wrap_policy(
            peft_model,
            config={"min_num_params": 0},
            is_lora=True,
        )
        wrapped = FSDP(
            peft_model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            sync_module_states=True,
            use_orig_params=False,
        )
        engine = object.__new__(FSDPEngine)
        engine.module = wrapped

        input_ids = torch.randint(0, 128, (2, 12), device=device)
        wrapped.eval()
        with torch.no_grad():
            adapter_logits_before = wrapped(
                input_ids=input_ids,
                use_cache=False,
            ).logits
            with engine.disable_adapter():
                reference_logits = wrapped(
                    input_ids=input_ids,
                    use_cache=False,
                ).logits
            adapter_logits_after = wrapped(
                input_ids=input_ids,
                use_cache=False,
            ).logits

        torch.testing.assert_close(
            adapter_logits_after,
            adapter_logits_before,
            rtol=0,
            atol=0,
        )
        assert not torch.allclose(adapter_logits_before, reference_logits)
    finally:
        wrapped = None
        gc.collect()
        torch.cuda.empty_cache()
        if owned_process_group:
            dist.destroy_process_group()


def _sampled_token_log_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
    labels = input_ids[:, 1:]
    return torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)


def test_tiny_rank64_exopd_updates_only_lora_parameters():
    """A real PEFT step preserves the disabled-adapter reference exactly."""

    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from verl.trainer.distillation.losses import compute_exopd_advantage

    torch.manual_seed(7)
    base = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    lora_config = model.peft_config["default"]
    assert int(lora_config.r) == 64
    assert int(lora_config.lora_alpha) == 128
    assert len(lora_config.target_modules) == 7
    assert all(
        name.endswith(
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
        for name in lora_config.target_modules
    )

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    assert trainable
    assert frozen
    assert all("lora_" in name for name in trainable)
    assert all("lora_" not in name for name in frozen)

    # PEFT initializes B to zero, so seed a visible policy/reference difference
    # before constructing the detached ExOPD advantage.
    with torch.no_grad():
        for name, parameter in trainable.items():
            if "lora_B" in name:
                parameter.fill_(0.01)

    input_ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad(), model.disable_adapter():
        reference_before = _sampled_token_log_probs(model, input_ids).clone()
    student_log_probs = _sampled_token_log_probs(model, input_ids)
    assert not torch.allclose(student_log_probs.detach(), reference_before)

    # A deterministic synthetic Teacher is sufficient here: the test concerns
    # which Student parameters receive the exact detached ExOPD signal.
    teacher_log_probs = reference_before + torch.linspace(
        -0.2, 0.2, reference_before.shape[-1]
    ).unsqueeze(0)
    advantage = compute_exopd_advantage(
        student_log_probs,
        teacher_log_probs,
        reference_before,
        exopd_lambda=1.25,
    )
    assert not advantage.requires_grad

    frozen_before = {
        name: parameter.detach().clone() for name, parameter in frozen.items()
    }
    trainable_before = {
        name: parameter.detach().clone() for name, parameter in trainable.items()
    }
    optimizer = torch.optim.AdamW(trainable.values(), lr=1.0e-2)
    optimizer.zero_grad(set_to_none=True)
    loss = -(advantage * student_log_probs).mean()
    loss.backward()

    assert all(parameter.grad is not None for parameter in trainable.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert all(parameter.grad is None for parameter in frozen.values())
    optimizer.step()

    assert any(
        not torch.equal(parameter.detach(), trainable_before[name])
        for name, parameter in trainable.items()
    )
    for name, parameter in frozen.items():
        torch.testing.assert_close(parameter.detach(), frozen_before[name], rtol=0, atol=0)

    with torch.no_grad(), model.disable_adapter():
        reference_after = _sampled_token_log_probs(model, input_ids)
    torch.testing.assert_close(reference_after, reference_before, rtol=0, atol=0)
