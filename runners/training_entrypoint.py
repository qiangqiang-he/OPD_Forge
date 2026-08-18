"""Hydra/Ray entrypoint for OPD training."""

from __future__ import annotations

import hydra

from runners.opd_entrypoint import OPDTaskRunner
from utils.opd_runtime import configure_opd_data_parallel_batch, configure_opd_defaults
from verl.trainer import main_ppo_sync as verl_sync


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    configure_opd_defaults(config)
    configure_opd_data_parallel_batch(config)
    verl_sync.auto_set_device(config)
    config.transfer_queue.enable = True
    verl_sync.validate_config(
        config=config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )
    verl_sync.run_ppo(config, task_runner_class=OPDTaskRunner)


if __name__ == "__main__":
    main()
