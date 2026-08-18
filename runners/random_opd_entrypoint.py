"""Backward-compatible entrypoint for retained random OPD variants."""

from __future__ import annotations

import hydra
import ray

from algorithms import resolve_algorithm
from utils.opd_runtime import configure_opd_data_parallel_batch, configure_opd_defaults
from verl.trainer import main_ppo_sync as verl_sync


_BaseTaskRunner = verl_sync.TaskRunner.__ray_metadata__.modified_class


@ray.remote
class RandomOPDTaskRunner(_BaseTaskRunner):
    """Construct pools for Random GKD-OPD or Random PG-OPD."""

    def run(self, config):
        configure_opd_defaults(config)
        verl_sync.pprint(verl_sync.OmegaConf.to_container(config, resolve=True))
        verl_sync.OmegaConf.resolve(config)
        algorithm = resolve_algorithm(config)
        if str(config.algorithm.name) not in {
            "random_gkd_opd",
            "random_pg_opd",
        }:
            raise ValueError(
                "random_opd_entrypoint only accepts random_gkd_opd or random_pg_opd."
            )
        algorithm.validate(config)

        verl_sync.tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
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
    configure_opd_data_parallel_batch(config)
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    algorithm = resolve_algorithm(config)
    if str(config.algorithm.name) not in {
        "random_gkd_opd",
        "random_pg_opd",
    }:
        raise ValueError(
            "random_opd_entrypoint only accepts random_gkd_opd or random_pg_opd."
        )
    algorithm.validate(config)
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    verl_sync.run_ppo(config, task_runner_class=RandomOPDTaskRunner)


if __name__ == "__main__":
    main()
