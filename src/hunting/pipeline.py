"""
Week 8 Scoring Adapter and End-to-End Week 9 Threat-Hunting Pipeline.

Bridges `MultiModelInferenceBundle` + `CalibratedFusionModel.decompose_evidence()`
(Weeks 5-8 checkpoints) with Week 9's `IncidentCorrelator`,
`EvidenceGroundedMitreMapper`, and `EvidenceGraphBuilder`. Also supports
pre-scored DataFrames/flows for synthetic L5/L6 Zeek test fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import pandas as pd

from src.hunting.correlator import IncidentCorrelator
from src.hunting.evidence_graph import EvidenceGraphBuilder
from src.hunting.mitre_mapper import EvidenceGroundedMitreMapper
from src.hunting.schemas import (
    GROUND_TRUTH_COLUMNS,
    FlowEvidence,
    IncidentCluster,
    IncidentEvidenceGraph,
)
from src.models.fusion import (
    CalibratedFusionModel,
    MultiModelInferenceBundle,
    discover_checkpoint_file,
)


class Week8ScoringAdapter:
    """Loads trained Weeks 5-8 checkpoints and outputs GT-free FlowEvidence records."""

    def __init__(
        self,
        inference_bundle: MultiModelInferenceBundle,
        fusion_model: CalibratedFusionModel,
        detector_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.inference_bundle = inference_bundle
        self.fusion_model = fusion_model
        self.detector_versions = dict(
            detector_versions or EvidenceGraphBuilder.DEFAULT_DETECTOR_VERSIONS
        )

    @classmethod
    def from_checkpoint_root(
        cls,
        checkpoint_root: Union[str, Path],
        device: Optional[str] = None,
    ) -> "Week8ScoringAdapter":
        root = Path(checkpoint_root)
        bundle = MultiModelInferenceBundle.from_checkpoint_root(root, device=device)
        fusion_path = discover_checkpoint_file(root, "week8_calibrated_fusion.joblib")
        fusion_payload = joblib.load(fusion_path)
        fusion_model = CalibratedFusionModel.from_checkpoint(fusion_payload)

        versions = {
            "week5_rules_baseline": str(bundle.rules_model.get_metadata().get("model_name", "week5_rules_baseline")),
            "week5_rf_binary": str(bundle.rf_model.get_metadata().get("model_name", "week5_rf_binary")),
            "week6_isolation_forest": str(bundle.if_model.get_metadata().get("model_name", "week6_isolation_forest")),
            "week7_lstm_temporal": str(bundle.lstm_model.get_metadata().get("model_name", "week7_lstm_temporal")),
            "week8_calibrated_fusion": f"threshold={fusion_model.threshold}",
        }
        return cls(inference_bundle=bundle, fusion_model=fusion_model, detector_versions=versions)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        lstm_batch_size: int = 512,
    ) -> List[FlowEvidence]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
        if len(df) == 0:
            return []

        # Strip ground-truth columns before scoring/hunting to guarantee zero GT leakage
        clean_cols = [c for c in df.columns if str(c).lower() not in GROUND_TRUTH_COLUMNS]
        df_clean = df[clean_cols].reset_index(drop=True)

        component_scores, context = self.inference_bundle.compute_component_scores(
            df_clean, lstm_batch_size=lstm_batch_size
        )
        decomposed = self.fusion_model.decompose_evidence(
            component_scores=component_scores,
            rule_reasons=context.get("rule_reasons"),
            anomaly_reasons=context.get("anomaly_reasons"),
        )

        records = df_clean.to_dict(orient="records")
        return [
            FlowEvidence.from_row_and_fusion(row, fusion_record=decomposed[idx], default_index=idx)
            for idx, row in enumerate(records)
        ]


class ThreatHuntingPipeline:
    """Orchestrates flow scoring/normalization -> correlation -> ATT&CK mapping -> evidence graphs."""

    def __init__(
        self,
        scoring_adapter: Optional[Week8ScoringAdapter] = None,
        correlator: Optional[IncidentCorrelator] = None,
        mitre_mapper: Optional[EvidenceGroundedMitreMapper] = None,
        graph_builder: Optional[EvidenceGraphBuilder] = None,
    ) -> None:
        self.scoring_adapter = scoring_adapter
        self.correlator = correlator or IncidentCorrelator()
        self.mitre_mapper = mitre_mapper or EvidenceGroundedMitreMapper()
        self.graph_builder = graph_builder or EvidenceGraphBuilder(
            default_detector_versions=(
                scoring_adapter.detector_versions if scoring_adapter is not None else None
            )
        )

    @staticmethod
    def _is_prescored_dataframe(df: pd.DataFrame) -> bool:
        return "s_total" in df.columns or (
            "component_scores" in df.columns
        ) or all(col in df.columns for col in ("s_rule", "s_classifier", "s_anomaly", "s_sequence"))

    def flows_from_dataframe(
        self,
        df: pd.DataFrame,
        decomposed_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[FlowEvidence]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
        if len(df) == 0:
            return []

        clean_cols = [c for c in df.columns if str(c).lower() not in GROUND_TRUTH_COLUMNS]
        df_clean = df[clean_cols].reset_index(drop=True)

        if decomposed_evidence is not None:
            if len(decomposed_evidence) != len(df_clean):
                raise ValueError(
                    f"Length mismatch between df ({len(df_clean)}) and decomposed_evidence ({len(decomposed_evidence)})."
                )
            records = df_clean.to_dict(orient="records")
            return [
                FlowEvidence.from_row_and_fusion(row, fusion_record=decomposed_evidence[idx], default_index=idx)
                for idx, row in enumerate(records)
            ]

        if self._is_prescored_dataframe(df_clean):
            records = df_clean.to_dict(orient="records")
            return [FlowEvidence.from_row_and_fusion(row, fusion_record=None, default_index=idx) for idx, row in enumerate(records)]

        if self.scoring_adapter is not None:
            return self.scoring_adapter.score_dataframe(df_clean)

        raise RuntimeError(
            "ThreatHuntingPipeline requires either `decomposed_evidence`, pre-scored flow columns "
            "(`s_total` / `component_scores`), or a configured `Week8ScoringAdapter`."
        )

    def run_on_flows(
        self,
        flows: Sequence[FlowEvidence],
        include_low_signal_clusters: Optional[bool] = None,
    ) -> List[Tuple[IncidentCluster, IncidentEvidenceGraph]]:
        clusters = self.correlator.correlate(
            flows, include_low_signal_clusters=include_low_signal_clusters
        )
        results: List[Tuple[IncidentCluster, IncidentEvidenceGraph]] = []
        det_versions = (
            self.scoring_adapter.detector_versions
            if self.scoring_adapter is not None
            else self.graph_builder.default_detector_versions
        )
        for cluster in clusters:
            mappings, unmapped_rationale = self.mitre_mapper.map_cluster(cluster)
            graph = self.graph_builder.build(
                cluster,
                attack_mappings=mappings,
                unmapped_rationale=unmapped_rationale,
                detector_versions=det_versions,
            )
            results.append((cluster, graph))
        return results

    def run(
        self,
        df: pd.DataFrame,
        decomposed_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
        include_low_signal_clusters: Optional[bool] = None,
    ) -> List[Tuple[IncidentCluster, IncidentEvidenceGraph]]:
        flows = self.flows_from_dataframe(df, decomposed_evidence=decomposed_evidence)
        return self.run_on_flows(flows, include_low_signal_clusters=include_low_signal_clusters)


def export_evidence_graphs_json(
    graphs: Sequence[IncidentEvidenceGraph],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize IncidentEvidenceGraph objects cleanly (`allow_nan=False`) to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "metadata": metadata or {},
        "incident_count": len(graphs),
        "incidents": [g.to_dict() for g in graphs],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return out
