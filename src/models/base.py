"""
Abstract OOP contracts (BaseThreatModel, BaseTrainer, BaseEvaluator)
and CheckpointManager for Topic 04 ML/DL models (Weeks 5, 6, 7).
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import torch

from src.utils.config import get_environment_info


class BaseThreatModel(ABC):
    """Abstract base class for all Topic 04 threat detection models."""

    def __init__(self, model_name: str, task_type: str) -> None:
        self.model_name: str = model_name
        self.task_type: str = task_type
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "BaseThreatModel":
        """Train the model on sanitized inputs."""

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Return discrete predictions for input samples."""

    def predict_proba(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Return class probability estimates when supported by the model."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba().")

    def score_samples(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Return continuous anomaly/threat scores when supported by the model."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement score_samples().")

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model hyperparameters and state metadata."""


class BaseTrainer(ABC):
    """Abstract base class for Topic 04 training orchestrators."""

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute training and return structured training results."""


class BaseEvaluator(ABC):
    """Abstract base class for Topic 04 evaluation components."""

    @abstractmethod
    def evaluate(self, model: BaseThreatModel, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a fitted model and return structured metrics."""


class CheckpointManager:
    """
    Saves and loads `.joblib` or PyTorch `state_dict` (`.pt`) artifacts alongside
    UTF-8 `_metadata.json` provenance files (`allow_nan=False`).
    """

    def __init__(self, checkpoint_dir: Union[str, Path]) -> None:
        self.checkpoint_dir: Path = Path(checkpoint_dir)

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        name = str(model_name).strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"Invalid checkpoint model_name: '{model_name}'")
        return name

    @classmethod
    def _contains_tensor(cls, obj: Any) -> bool:
        if isinstance(obj, torch.Tensor):
            return True
        if isinstance(obj, Mapping):
            return any(cls._contains_tensor(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(cls._contains_tensor(v) for v in obj)
        return False

    @classmethod
    def _sanitize_json(cls, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, int):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return val if math.isfinite(val) else None
        if isinstance(obj, np.ndarray):
            return cls._sanitize_json(obj.tolist())
        if isinstance(obj, Mapping):
            return {str(k): cls._sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [cls._sanitize_json(v) for v in obj]
        return str(obj)

    def save_checkpoint(
        self,
        model_name: str,
        artifacts: Any,
        metadata: Optional[Dict[str, Any]] = None,
        fmt: Optional[Literal["joblib", "pt"]] = None,
    ) -> Dict[str, Path]:
        clean_name = self._validate_model_name(model_name)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        resolved_fmt: Literal["joblib", "pt"]
        payload: Any = artifacts
        if fmt is not None:
            if fmt not in ("joblib", "pt"):
                raise ValueError(f"Unsupported checkpoint format '{fmt}'. Expected 'joblib' or 'pt'.")
            resolved_fmt = fmt
            if resolved_fmt == "pt" and isinstance(artifacts, torch.nn.Module):
                payload = artifacts.state_dict()
        else:
            if isinstance(artifacts, torch.nn.Module):
                resolved_fmt = "pt"
                payload = artifacts.state_dict()
            elif isinstance(artifacts, torch.Tensor) or (
                isinstance(artifacts, Mapping) and self._contains_tensor(artifacts)
            ):
                resolved_fmt = "pt"
            else:
                resolved_fmt = "joblib"

        artifact_filename = f"{clean_name}.{resolved_fmt}"
        artifact_path = self.checkpoint_dir / artifact_filename
        metadata_path = self.checkpoint_dir / f"{clean_name}_metadata.json"

        if resolved_fmt == "pt":
            torch.save(payload, artifact_path)
        else:
            joblib.dump(payload, artifact_path, compress=3)

        sanitized_user_meta = self._sanitize_json(metadata or {})
        if not isinstance(sanitized_user_meta, dict):
            sanitized_user_meta = {"metadata": sanitized_user_meta}

        meta_record: Dict[str, Any] = {
            **sanitized_user_meta,
            "model_name": clean_name,
            "format": resolved_fmt,
            "artifact_filename": artifact_filename,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": self._sanitize_json(get_environment_info()),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_record, f, indent=2, ensure_ascii=False, allow_nan=False)

        return {"artifact_path": artifact_path, "metadata_path": metadata_path}

    def load_metadata(self, model_name: str) -> Dict[str, Any]:
        clean_name = self._validate_model_name(model_name)
        metadata_path = self.checkpoint_dir / f"{clean_name}_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_checkpoint(
        self,
        model_name: str,
        fmt: Optional[Literal["joblib", "pt"]] = None,
        map_location: Any = "cpu",
    ) -> Tuple[Any, Dict[str, Any]]:
        clean_name = self._validate_model_name(model_name)
        metadata_path = self.checkpoint_dir / f"{clean_name}_metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.is_file():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        resolved_fmt: Optional[str] = fmt or metadata.get("format")
        if resolved_fmt is None:
            if (self.checkpoint_dir / f"{clean_name}.pt").is_file():
                resolved_fmt = "pt"
            elif (self.checkpoint_dir / f"{clean_name}.joblib").is_file():
                resolved_fmt = "joblib"
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for '{clean_name}' in {self.checkpoint_dir}"
                )

        if resolved_fmt not in ("joblib", "pt"):
            raise ValueError(f"Unsupported checkpoint format '{resolved_fmt}'.")

        artifact_path = self.checkpoint_dir / f"{clean_name}.{resolved_fmt}"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Checkpoint artifact not found: {artifact_path}")

        if resolved_fmt == "pt":
            try:
                artifacts = torch.load(artifact_path, map_location=map_location, weights_only=True)
            except TypeError:
                artifacts = torch.load(artifact_path, map_location=map_location)
        else:
            artifacts = joblib.load(artifact_path)

        return artifacts, metadata

    def list_checkpoints(self) -> List[str]:
        if not self.checkpoint_dir.is_dir():
            return []
        names = set()
        for p in self.checkpoint_dir.iterdir():
            if p.is_file():
                if p.suffix in {".pt", ".joblib"}:
                    names.add(p.stem)
                elif p.name.endswith("_metadata.json"):
                    names.add(p.name[: -len("_metadata.json")])
        return sorted(names)
