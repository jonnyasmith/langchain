from datetime import date

from pydantic import BaseModel, Field


class TermsOfService(BaseModel):
    """Key legal terms extracted from Terms of Service."""

    governing_law: str | None = Field(
        description="Jurisdiction whose law governs the agreement, or null when unstated."
    )
    arbitration_required: bool | None = Field(
        description="Whether the agreement requires arbitration, or null when unstated."
    )
    arbitration_clause: str | None = Field(
        description="The arbitration clause text, or null when the document has no such clause."
    )
    liability_cap: str | None = Field(
        description="The stated cap on liability, preserving its amount or formula, or null."
    )
    termination_notice_period: str | None = Field(
        description="Notice period required to terminate the agreement, or null when unstated."
    )
    data_retention_period: str | None = Field(
        description="How long data is retained after termination, or null when unstated."
    )
    effective_date: date | None = Field(
        description="Date on which the agreement takes effect, or null when unstated."
    )


SCHEMAS: dict[str, type[BaseModel]] = {"tos": TermsOfService}
