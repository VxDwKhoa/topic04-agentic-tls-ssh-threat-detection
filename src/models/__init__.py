"""Topic 04 ML/DL Models Package (Weeks 5, 6, 7)."""

from src.models.base import BaseEvaluator, BaseThreatModel, BaseTrainer, CheckpointManager
from src.models.rules_baseline import RulesBaselineModel
from src.models.random_forest import RandomForestEvaluator, RandomForestThreatModel, RandomForestTrainer
from src.models.isolation_forest import (
    IsolationForestEvaluator,
    IsolationForestThreatModel,
    IsolationForestTrainer,
)
from src.models.lstm_sequence import LSTMEvaluator, LSTMSequenceModel, LSTMTrainer, PureLSTMNetwork
from src.models.fusion import (
    COMPONENT_NAMES,
    CalibratedFusionModel,
    FusionEvaluator,
    MultiModelInferenceBundle,
    align_lstm_sequence_scores_to_flows,
    discover_checkpoint_file,
)

__all__ = [
    "BaseThreatModel",
    "BaseTrainer",
    "BaseEvaluator",
    "CheckpointManager",
    "RulesBaselineModel",
    "RandomForestThreatModel",
    "RandomForestTrainer",
    "RandomForestEvaluator",
    "IsolationForestThreatModel",
    "IsolationForestTrainer",
    "IsolationForestEvaluator",
    "PureLSTMNetwork",
    "LSTMSequenceModel",
    "LSTMTrainer",
    "LSTMEvaluator",
    "COMPONENT_NAMES",
    "CalibratedFusionModel",
    "FusionEvaluator",
    "MultiModelInferenceBundle",
    "align_lstm_sequence_scores_to_flows",
    "discover_checkpoint_file",
]
