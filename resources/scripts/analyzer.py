"""Presidio Analyzer Engine setup and configuration."""

from typing import Dict

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.pattern import Pattern
from presidio_analyzer.pattern_recognizer import PatternRecognizer
from presidio_evaluator.models import PresidioAnalyzerWrapper

from .models import AnalyzerConfig


def create_analyzer_engine(config: AnalyzerConfig) -> AnalyzerEngine:
    """Create and configure the Presidio Analyzer Engine.

    Args:
        config: Evaluation configuration containing NLP and analyzer settings.

    Returns:
        Configured AnalyzerEngine instance.
    """
    nlp_config = {
        "nlp_engine_name": config.nlp_engine_name,
        "models": [
            {"lang_code": config.language, "model_name": config.model_name}
        ],
        "ner_model_configuration": {
            "labels_to_ignore": [
                "CARDINAL",
                "QUANTITY",
                "MONEY",
                "PRODUCT",
                "WORK_OF_ART",
                "EVENT",
                "FAC",
            ]
        },
    }

    print(f"Creating NLP engine with config: {nlp_config}")
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

    # Set up context aware enhancer if configured
    context_enhancer = None
    if config.context_enhancer_count is not None:
        context_enhancer = LemmaContextAwareEnhancer(
            context_prefix_count=config.context_enhancer_count,
            context_suffix_count=config.context_enhancer_count,
        )
        print(
            f"Using LemmaContextAwareEnhancer with context count: "
            f"{config.context_enhancer_count}"
        )

    analyzer_engine = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=[config.language],
        context_aware_enhancer=context_enhancer,
    )

    # Remove unwanted recognizers if specified
    if config.recognizers_to_keep:
        print(f"Keeping only recognizers: {config.recognizers_to_keep}")
        for rec in analyzer_engine.registry.get_recognizers(
            config.language, all_fields=True
        ):
            if rec.name not in config.recognizers_to_keep:
                analyzer_engine.registry.remove_recognizer(rec.name)

    # Add custom recognizers
    for rec_config in config.custom_recognizers:
        patterns = [
            Pattern(name=p.name, regex=p.regex, score=p.score)
            for p in rec_config.patterns
        ]
        recognizer = PatternRecognizer(
            supported_entity=rec_config.entity_type, patterns=patterns
        )
        analyzer_engine.registry.add_recognizer(recognizer)
        print(f"Added custom recognizer for entity: {rec_config.entity_type}")

    # Print loaded recognizers
    loaded_recognizers = [
        rec.name
        for rec in analyzer_engine.registry.get_recognizers(
            config.language, all_fields=True
        )
    ]
    print(f"Loaded recognizers: {loaded_recognizers}")

    return analyzer_engine


def get_entities_mapping(config: AnalyzerConfig) -> Dict[str, str]:
    """Get entity type mappings including defaults and custom mappings.

    Args:
        config: Analyzer configuration containing entity mappings.

    Returns:
        Dictionary mapping source entity types to target entity types.
    """
    # Start with default Presidio mappings
    mapping = dict(PresidioAnalyzerWrapper.presidio_entities_map)

    # Add custom mappings from config
    for entity_map in config.entity_mappings:
        mapping[entity_map.source] = entity_map.target

    return mapping
