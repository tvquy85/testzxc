from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode='after')
    def check_action_consistency(self) -> 'RationaleOutput':
        # Ensure action somewhat aligns with probabilities
        down_prob = self.forecast_distribution.strong_down + self.forecast_distribution.mild_down
        up_prob = self.forecast_distribution.strong_up + self.forecast_distribution.mild_up
        
        # A simple check: if one side is strongly dominant, action should match
        if down_prob > 0.6 and self.action == "long":
            raise ValueError("Action is 'long' but downward probability > 60%")
        if up_prob > 0.6 and self.action == "short":
            raise ValueError("Action is 'short' but upward probability > 60%")
            
        return self
