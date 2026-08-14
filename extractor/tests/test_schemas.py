"""The `tos` schema's shape is the extraction contract: strict mode and the prompt both read it."""

from types import NoneType
from typing import get_args

from extractor.schemas import TermsOfService


def test_every_field_is_described_because_descriptions_are_prompt_surface() -> None:
    """Descriptions travel to the provider inside the JSON schema and carry the extraction
    semantics, so an undescribed field degrades extraction without failing anything else."""
    undescribed = [
        name for name, field in TermsOfService.model_fields.items() if not field.description
    ]

    assert undescribed == []


def test_every_field_is_required_and_nullable_as_strict_mode_demands() -> None:
    """ADR-0001: optionality is a nullable type with no default. The conventional Pydantic
    spelling — `str | None = None` — makes the field non-required, which strict mode rejects."""
    for name, field in TermsOfService.model_fields.items():
        assert field.is_required(), f"{name} has a default, so strict mode drops it from required"
        assert NoneType in get_args(field.annotation), f"{name} cannot hold an absent field"
