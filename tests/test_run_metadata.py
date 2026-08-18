"""Regression tests for run naming and W&B grouping defaults."""

import re

from omegaconf import OmegaConf

from utils.opd_runtime import configure_run_metadata


def test_explicit_run_and_group_names_are_preserved():
    config = OmegaConf.create(
        {
            "run_name": "explicit_run",
            "group_name": "PG_OPD",
            "algorithm": {"name": "pg_opd"},
            "trainer": {"experiment_name": "old"},
        }
    )

    run_name, group_name = configure_run_metadata(config)

    assert run_name == "explicit_run"
    assert group_name == "PG_OPD"
    assert config.trainer.experiment_name == "explicit_run"


def test_empty_run_is_generated_and_empty_group_defaults_to_temp():
    config = OmegaConf.create(
        {
            "run_name": "",
            "group_name": "",
            "run_name_prefix": "automatic_pg",
            "algorithm": {"name": "pg_opd"},
            "trainer": {"experiment_name": "old"},
        }
    )

    run_name, group_name = configure_run_metadata(config)

    assert re.fullmatch(r"automatic_pg_\d{8}_\d{6}_p\d+", run_name)
    assert group_name == "temp"
    assert config.run_name == run_name
    assert config.group_name == "temp"
    assert config.trainer.experiment_name == run_name
