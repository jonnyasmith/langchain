"""Every named schema's shape is the extraction contract: strict mode and the prompt both read it.

These run over the whole `SCHEMAS` registry rather than one model by name, so a schema added to
the registry cannot escape the contract that ADR-0001 requires of it.
"""

from types import NoneType
from typing import get_args

import pytest
from pydantic import BaseModel

from extractor.schemas import SCHEMAS


@pytest.mark.parametrize("schema", SCHEMAS.values(), ids=SCHEMAS.keys())
def test_every_field_is_described_because_descriptions_are_prompt_surface(
    schema: type[BaseModel],
) -> None:
    """Descriptions travel to the provider inside the JSON schema and carry the extraction
    semantics, so an undescribed field degrades extraction without failing anything else."""
    undescribed = [name for name, field in schema.model_fields.items() if not field.description]

    assert undescribed == []


@pytest.mark.parametrize("schema", SCHEMAS.values(), ids=SCHEMAS.keys())
def test_every_field_is_required_and_nullable_as_strict_mode_demands(
    schema: type[BaseModel],
) -> None:
    """ADR-0001: optionality is a nullable type with no default. The conventional Pydantic
    spelling — `str | None = None` — makes the field non-required, which strict mode rejects."""
    for name, field in schema.model_fields.items():
        assert field.is_required(), f"{name} has a default, so strict mode drops it from required"
        assert NoneType in get_args(field.annotation), f"{name} cannot hold an absent field"


def test_the_registry_is_not_empty_so_the_contract_tests_cannot_pass_vacuously() -> None:
    """Parametrising over an empty registry would collect nothing and report green."""
    assert SCHEMAS
