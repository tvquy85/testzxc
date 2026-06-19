from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal

class ForecastDistribution(BaseModel):
    strong_down: float = Field(..., ge=0, le=1)
    mild_down: float = Field(..., ge=0, le=1)
    neutral: float = Field(..., ge=0, le=1)
    mild_up: float = Field(..., ge=0, le=1)
    strong_up: float = Field(..., ge=0, le=1)

    @model_validator(mode='after')
    def check_sum(self) -> 'ForecastDistribution':
        total = self.strong_down + self.mild_down + self.neutral + self.mild_up + self.strong_up
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Probabilities must sum to approximately 1.0, got {total}")
        return self

class RationaleOutput(BaseModel):
    news_rationale: List[str]
    technical_rationale: List[str]
    conflict_resolution: str
    forecast_distribution: ForecastDistribution
    action: Literal["long", "short", "hold"]
    risk_note: str

    @field_validator("news_rationale", "technical_rationale")
    @classmethod
    def non_empty_list(cls, value: List[str]) -> List[str]:
        if not value or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("rationale lists must contain non-empty strings")
        return value

    @model_validator(mode='after')
    def check_strings(self) -> 'RationaleOutput':
        if not self.conflict_resolution.strip():
            raise ValueError("conflict_resolution must be non-empty")
        if not self.risk_note.strip():
            raise ValueError("risk_note must be non-empty")
        return self
