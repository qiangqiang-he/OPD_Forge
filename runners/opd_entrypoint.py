"""Ray task runner for the project standard OPD trainer."""

from __future__ import annotations

import hydra
import ray

from algorithms import resolve_algorithm
from utils.opd_runtime import configure_opd_defaults
from verl.trainer import main_ppo_sync as verl_sync


_BaseTaskRunner = verl_sync.TaskRunner.__ray_metadata__.modified_class


@ray.remote
class OPDTaskRunner(_BaseTaskRunner):
    """Construct the colocated student pool and separate teacher pool."""

    def run(self, config):
        configure_opd_defaults(config)
        verl_sync.pprint(verl_sync.OmegaConf.to_container(config, resolve=True))
        verl_sync.OmegaConf.resolve(config)
        algorithm = resolve_algorithm(config)
        algorithm.validate(config)

        verl_sync.tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            # With distillation enabled this creates a distinct teacher_pool in
            # addition to the student actor/rollout global_pool.
            self.init_resource_pool_mgr(config)
            # Standard OPD has two colocated student roles (FSDP actor and
            # rollout) and no critic/reference-policy role. Reserving VERL's
            # default three CPU slots per GPU prevents the independent teacher
            # placement group from fitting on an 8-GPU/32-CPU node.
            self.resource_pool_manager.max_colocate_count = 2
            trainer = algorithm.trainer_class(
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
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    resolve_algorithm(config).validate(config)
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    verl_sync.run_ppo(config, task_runner_class=OPDTaskRunner)


if __name__ == "__main__":
    main()
