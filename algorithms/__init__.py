"""Explicit registry for the retained OPD-family algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Type


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    trainer_class: Type
    validate: Callable


@dataclass(frozen=True)
class AlgorithmRegistration:
    module: str
    trainer_class: str
    validator: str


ALGORITHM_REGISTRY: dict[str, AlgorithmRegistration] = {
    "pg_opd": AlgorithmRegistration(
        "algorithms.pg_opd", "PGOPDTrainer", "validate_pg_opd_config"
    ),
    "random_pg_opd": AlgorithmRegistration(
        "algorithms.random_pg_opd",
        "RandomPGOPDTrainer",
        "validate_random_pg_opd_config",
    ),
    "topgap_pg_opd": AlgorithmRegistration(
        "algorithms.topgap_pg_opd",
        "TopGapPGOPDTrainer",
        "validate_topgap_pg_opd_config",
    ),
    "gkd_opd": AlgorithmRegistration(
        "algorithms.gkd_opd", "GKDOPDTrainer", "validate_gkd_opd_config"
    ),
    "random_gkd_opd": AlgorithmRegistration(
        "algorithms.random_gkd_opd",
        "RandomGKDOPDTrainer",
        "validate_random_gkd_opd_config",
    ),
    "topgap_gkd_opd": AlgorithmRegistration(
        "algorithms.topgap_gkd_opd",
        "TopGapGKDOPDTrainer",
        "validate_topgap_gkd_opd_config",
    ),
    "ps_opd": AlgorithmRegistration(
        "algorithms.ps_opd", "PSOPDTrainer", "validate_ps_opd_config"
    ),
    "cal_opd": AlgorithmRegistration(
        "algorithms.cal_opd", "CalOPDTrainer", "validate_cal_opd_config"
    ),
}


def resolve_algorithm(config) -> AlgorithmSpec:
    """Resolve the exact canonical name in ``algorithm.name``."""

    name = str(config.algorithm.name)
    registration = ALGORITHM_REGISTRY.get(name)
    if registration is None:
        expected = ", ".join(ALGORITHM_REGISTRY)
        raise ValueError(
            f"Unknown algorithm.name={name!r}; expected one of: {expected}."
        )
    module = import_module(registration.module)
    return AlgorithmSpec(
        name=name,
        trainer_class=getattr(module, registration.trainer_class),
        validate=getattr(module, registration.validator),
    )


__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmRegistration",
    "AlgorithmSpec",
    "resolve_algorithm",
]
