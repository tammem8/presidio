"""Pydantic models for PII detection evaluation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PatternConfig(BaseModel):
    """Pattern recognizer configuration."""

    name: str
    regex: str
    score: float = 0.9


class RecognizerConfig(BaseModel):
    """Custom recognizer configuration."""

    entity_type: str
    patterns: List[PatternConfig]


class EntityMappingConfig(BaseModel):
    """Entity type mapping configuration."""

    source: str
    target: str


class AnalyzerConfig(BaseModel):
    """Complete analyzer configuration including NLP settings."""

    # NLP settings
    nlp_engine_name: str = "spacy"
    model_name: str = "en_core_web_sm"
    language: str = "en"

    # Analyzer settings
    default_score_threshold: float = 0.4
    recognizers_to_keep: Optional[List[str]] = None
    custom_recognizers: List[RecognizerConfig] = Field(default_factory=list)
    entity_mappings: List[EntityMappingConfig] = Field(default_factory=list)
    context_enhancer_count: Optional[int] = None


class EntityMetrics(BaseModel):
    """Metrics for a single entity type."""

    entity_type: str
    count: int
    precision: float
    recall: float
    f1_score: float


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
