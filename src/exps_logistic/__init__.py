"""Logistic regression experiments for concept classification from CoT rationales."""

from .classifier import ClassifierResult, ConceptClassifier
from .config import ExperimentConfig, parse_args
from .data_utils import (
    create_theta_new_label,
    equal_width_bin,
    filter_by_rep,
    load_data,
    make_gamma_label,
    prepare_labels,
    stratified_split_robust,
)
from .featurizer import (
    Featurizer,
    HFCLSFeaturizer,
    OpenAIEmbeddingFeaturizer,
    SentenceTransformersFeaturizer,
    TfidfFeaturizer,
    build_featurizer,
)
from .metrics import EvaluationMetrics, compute_metrics, empirical_entropy_bits, print_results
from .parsers import (
    extract_problem_text,
    maybe_strip_fences,
    parse_arithmetic_operands,
    parse_knap_stats,
    parse_lcs_lengths,
    parse_list_of_ints,
    parse_list_of_list_ints,
    parse_rod_N,
    strip_md_code_fences,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "parse_args",
    # Classifier
    "ConceptClassifier",
    "ClassifierResult",
    # Data utilities
    "load_data",
    "filter_by_rep",
    "prepare_labels",
    "stratified_split_robust",
    "equal_width_bin",
    "make_gamma_label",
    "create_theta_new_label",
    # Featurizers
    "Featurizer",
    "TfidfFeaturizer",
    "HFCLSFeaturizer",
    "SentenceTransformersFeaturizer",
    "OpenAIEmbeddingFeaturizer",
    "build_featurizer",
    # Metrics
    "EvaluationMetrics",
    "compute_metrics",
    "empirical_entropy_bits",
    "print_results",
    # Parsers
    "strip_md_code_fences",
    "maybe_strip_fences",
    "parse_list_of_ints",
    "parse_list_of_list_ints",
    "extract_problem_text",
    "parse_arithmetic_operands",
    "parse_lcs_lengths",
    "parse_knap_stats",
    "parse_rod_N",
]
