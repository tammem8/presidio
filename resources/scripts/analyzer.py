"""Presidio Analyzer Engine setup and configuration."""

from pathlib import Path
from typing import Dict, Optional

from presidio_analyzer import AnalyzerEngine, AnalyzerEngineProvider
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

# Default config directory relative to this file
CONFIG_DIR = Path(__file__).parent.parent / "config"


def create_analyzer_engine(
    analyzer_conf_file: Optional[str] = None,
    nlp_engine_conf_file: Optional[str] = None,
    recognizer_registry_conf_file: Optional[str] = None,
) -> AnalyzerEngine:
    """Create and configure the Presidio Analyzer Engine using YAML config files.

    Args:
        analyzer_conf_file: Path to analyzer config YAML file.
            Defaults to config/analyzer-config.yaml.
        nlp_engine_conf_file: Path to NLP engine config YAML file.
            Defaults to config/nlp-config.yaml.
        recognizer_registry_conf_file: Path to recognizer registry config YAML file.
            Defaults to config/recognizers-config.yaml.

    Returns:
        Configured AnalyzerEngine instance.
    """
    analyzer_conf_file = analyzer_conf_file or str(CONFIG_DIR / "analyzer-config.yaml")
    nlp_engine_conf_file = nlp_engine_conf_file or str(CONFIG_DIR / "nlp-config.yaml")
    recognizer_registry_conf_file = recognizer_registry_conf_file or str(
        CONFIG_DIR / "recognizers-config.yaml"
    )

    print(f"Loading analyzer config from: {analyzer_conf_file}")
    print(f"Loading NLP engine config from: {nlp_engine_conf_file}")
    print(f"Loading recognizer registry config from: {recognizer_registry_conf_file}")
    
    provider = AnalyzerEngineProvider(
        analyzer_engine_conf_file=analyzer_conf_file,
        nlp_engine_conf_file=nlp_engine_conf_file,
        recognizer_registry_conf_file=recognizer_registry_conf_file,
    )
    analyzer = provider.create_engine()

    # # add context-aware enhancer to the analyzer
    # context_enhancer = LemmaContextAwareEnhancer()
    # analyzer.context_aware_enhancer = context_enhancer

    return analyzer