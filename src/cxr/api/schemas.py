"""Response models of the API."""

from typing import Literal

from pydantic import BaseModel

DISCLAIMER = "Research use only. Not a medical device; not for clinical diagnosis."


class QualityFlag(BaseModel):
    stat: str
    value: float
    side: Literal["low", "high"]
    limit: float  # the train reference quantile that was crossed


class Quality(BaseModel):
    status: Literal["ok", "warning"]  # warning = at least one flag; the prediction is still returned
    flags: list[QualityFlag]
    stats: dict[str, float]


class Prediction(BaseModel):
    label: str  # NORMAL or PNEUMONIA
    prob_pneumonia: float
    threshold: float  # PNEUMONIA when prob_pneumonia >= threshold (chosen on val)
    model_version: str
    quality: Quality
    disclaimer: str = DISCLAIMER


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
