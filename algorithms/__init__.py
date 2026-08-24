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
    "gkd_opd": AlgorithmRegistration(
        "algorithms.gkd_opd", "GKDOPDTrainer", "validate_gkd_opd_config"
    ),
    "ps_opd": AlgorithmRegistration(
        "algorithms.ps_opd", "PSOPDTrainer", "validate_ps_opd_config"
    ),
    "cal_opd": AlgorithmRegistration(
        "algorithms.cal_opd", "CalOPDTrainer", "validate_cal_opd_config"
    ),
    "exopd": AlgorithmRegistration(
        "algorithms.exopd", "ExOPDTrainer", "validate_exopd_config"
    ),
    "eopd": AlgorithmRegistration(
        "algorithms.eopd", "EOPDTrainer", "validate_eopd_config"
    ),
    "uni_opd": AlgorithmRegistration(
        "algorithms.uni_opd", "UniOPDTrainer", "validate_uni_opd_config"
    ),
    "fire_opd": AlgorithmRegistration(
        "algorithms.fire_opd", "FiReOPDTrainer", "validate_fire_opd_config"
    ),
    "oa_opd": AlgorithmRegistration(
        "algorithms.oa_opd", "OAOPDTrainer", "validate_oa_opd_config"
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
