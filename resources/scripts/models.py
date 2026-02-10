"""Pydantic models for PII detection evaluation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    """Overall evaluation metrics."""

    pii_precision: float
    pii_recall: float
    pii_f1_score: float
    entity_precision_dict: Dict[str, float]
    entity_recall_dict: Dict[str, float]
    total_samples: int
    total_entities: int
    samples_evaluated: int
    samples_discarded: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    config: Optional[Dict[str, Any]] = None


class EvaluationOutput(BaseModel):
    """Complete evaluation output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: EvaluationMetrics
    fps: pd.DataFrame = Field(default_factory=pd.DataFrame)
    fns: pd.DataFrame = Field(default_factory=pd.DataFrame)


class PromptEvaluationInput(BaseModel):
    """Input configuration for prompt-based PII analysis."""

    text: str = Field(..., description="The text to analyze for PII entities")
    language: str = Field(default="en", description="Language code for the text")
