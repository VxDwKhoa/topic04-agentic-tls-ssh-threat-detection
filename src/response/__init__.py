"""Topic 04 Week 10 Policy-Constrained Response Subsystem, Evaluator & Offline Demo."""

from typing import Any

from src.response.planner import HumanApprovalGate, ResponsePlanner
from src.response.retest_learn import Stage7RetestVerifier, Stage8FeedbackManager
from src.response.schemas import (
    ActionRequest,
    AnalystFeedbackRecord,
    ApprovalDecision,
    AuditEvent,
    AutonomousTrainingForbiddenError,
    ContainmentEntry,
    FeedbackSubmissionResult,
    GateDecision,
    RemediationDiff,
    ResponsePlan,
    ResponsePolicyViolationError,
    RetestObservation,
    RetestResult,
    RollbackResult,
    SimulatedActionRecord,
    is_human_reviewer_id,
    validate_week10_payload,
)
from src.response.simulator import InMemoryResponseSimulator

_LAZY_EXPORTS = {
    "ResponsePolicyEvaluator": ("src.response.evaluator", "ResponsePolicyEvaluator"),
    "run_week10_evaluation": ("src.response.evaluator", "run_week10_evaluation"),
    "EndToEndDemoRunner": ("src.response.demo_end_to_end", "EndToEndDemoRunner"),
    "build_arg_parser": ("src.response.demo_end_to_end", "build_arg_parser"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionRequest",
    "AnalystFeedbackRecord",
    "ApprovalDecision",
    "AuditEvent",
    "AutonomousTrainingForbiddenError",
    "ContainmentEntry",
    "EndToEndDemoRunner",
    "FeedbackSubmissionResult",
    "GateDecision",
    "HumanApprovalGate",
    "InMemoryResponseSimulator",
    "RemediationDiff",
    "ResponsePlan",
    "ResponsePlanner",
    "ResponsePolicyEvaluator",
    "ResponsePolicyViolationError",
    "RetestObservation",
    "RetestResult",
    "RollbackResult",
    "SimulatedActionRecord",
    "Stage7RetestVerifier",
    "Stage8FeedbackManager",
    "build_arg_parser",
    "is_human_reviewer_id",
    "run_week10_evaluation",
    "validate_week10_payload",
]
