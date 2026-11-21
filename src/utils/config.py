"""
Configuration, dual-environment path resolution (Kaggle GPU vs Local),
and reproducibility utilities for Topic 04 (Weeks 5, 6, 7).
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
import torch

KAGGLE_DATASET_DIR = Path("/kaggle/input/datasets/vxdwkhoa/Dataset_NF/NF-UQ-NIDS-v2")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
KAGGLE_CHECKPOINT_DIR = KAGGLE_WORKING_DIR / "Checkpoint"

LOCAL_FALLBACK_DATASET_DIR = Path(r"D:\Crypto\DoAn\NF-UQ-NIDS-v2")
LOCAL_FALLBACK_CHECKPOINT_DIR = Path(r"D:\Crypto\DoAn\Notebook\Checkpoint")

SAMPLE_REL_PATH = Path("data") / "public_loaders" / "sample_data" / "nfuq_sample_1500.parquet"
SAMPLE_MANIFEST_REL_PATH = Path("data") / "public_loaders" / "sample_data" / "nfuq_manifest_and_audit.json"
SCHEMA_REL_PATH = Path("data") / "schemas" / "schema_allowlist.json"


def discover_repo_root(start: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Locate the repository root by walking parent directories for Topic 04 markers."""
    search_starts = []
    if start is not None:
        search_starts.append(Path(start).resolve())
    else:
        search_starts.append(Path(__file__).resolve())
        cwd = Path.cwd().resolve()
        if cwd not in search_starts:
            search_starts.append(cwd)

    for marker in (SCHEMA_REL_PATH, Path("pyproject.toml"), SAMPLE_REL_PATH):
        for anchor in search_starts:
            candidates = [anchor, *anchor.parents]
            for candidate in candidates:
                if (candidate / marker).is_file():
                    return candidate
    return None


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved dataset, schema, manifest, and checkpoint paths."""

    mode: str
    is_sample_mode: bool
    dataset_dir: Path
    checkpoint_dir: Path
    repo_root: Optional[Path]
    sample_path: Optional[Path]
    manifest_path: Optional[Path]
    schema_path: Optional[Path]

    def split_tabular_path(self, split: str) -> Optional[Path]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train', 'val', or 'test'.")
        if self.is_sample_mode:
            return None
        return self.dataset_dir / split / "flows_tabular.parquet"

    @property
    def unseen_holdout_path(self) -> Optional[Path]:
        if self.is_sample_mode:
            return None
        return self.dataset_dir / "test" / "unseen_attack_holdout.parquet"

    def sequences_dir(self, split: str) -> Optional[Path]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train', 'val', or 'test'.")
        if self.is_sample_mode:
            return None
        return self.dataset_dir / split / "sequences_temporal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "is_sample_mode": self.is_sample_mode,
            "dataset_dir": str(self.dataset_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "repo_root": str(self.repo_root) if self.repo_root is not None else None,
            "sample_path": str(self.sample_path) if self.sample_path is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "schema_path": str(self.schema_path) if self.schema_path is not None else None,
        }


class PathResolver:
    """Side-effect-free path resolver supporting Kaggle GPU, local full dataset, and sample mode."""

    @staticmethod
    def _inspect_dataset_candidate(candidate: Path) -> Optional[Tuple[Path, bool, Optional[Path]]]:
        """
        Inspect a dataset candidate path.
        Returns (resolved_dataset_dir, is_sample_mode, sample_path) if valid, else None.
        """
        if not candidate.exists():
            return None

        if candidate.is_file() and candidate.suffix.lower() == ".parquet":
            return candidate, True, candidate

        if candidate.is_dir():
            if (candidate / "train" / "flows_tabular.parquet").is_file():
                return candidate, False, None
            nested = candidate / "NF-UQ-NIDS-v2"
            if (nested / "train" / "flows_tabular.parquet").is_file():
                return nested, False, None
            sample_candidate = candidate / "nfuq_sample_1500.parquet"
            if sample_candidate.is_file():
                return sample_candidate, True, sample_candidate

        return None

    @classmethod
    def _resolve_checkpoint_dir(
        cls,
        checkpoint_dir: Optional[Union[str, Path]],
        repo_root: Optional[Path],
    ) -> Path:
        if checkpoint_dir is not None and str(checkpoint_dir).strip():
            return Path(checkpoint_dir)

        env_ckpt = os.environ.get("TOPIC04_CHECKPOINT_DIR", "").strip()
        if env_ckpt:
            return Path(env_ckpt)

        if KAGGLE_WORKING_DIR.exists():
            return KAGGLE_CHECKPOINT_DIR

        if repo_root is not None and (repo_root.parent / "Notebook").exists():
            return repo_root.parent / "Notebook" / "Checkpoint"

        if LOCAL_FALLBACK_CHECKPOINT_DIR.parent.exists():
            return LOCAL_FALLBACK_CHECKPOINT_DIR

        return Path.cwd() / "Checkpoint"

    @classmethod
    def resolve(
        cls,
        dataset_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        prefer_sample: bool = False,
    ) -> ResolvedPaths:
        repo_root = discover_repo_root()
        schema_path = (
            repo_root / SCHEMA_REL_PATH
            if repo_root is not None and (repo_root / SCHEMA_REL_PATH).is_file()
            else None
        )
        resolved_checkpoint_dir = cls._resolve_checkpoint_dir(checkpoint_dir, repo_root)

        mode: Optional[str] = None
        is_sample_mode = False
        resolved_dataset_dir: Optional[Path] = None
        sample_path: Optional[Path] = None

        # 1. Explicit dataset_dir argument (fail-closed)
        if dataset_dir is not None and str(dataset_dir).strip():
            candidate = Path(dataset_dir)
            if not candidate.exists():
                raise FileNotFoundError(f"Explicit dataset_dir does not exist: {candidate}")
            inspected = cls._inspect_dataset_candidate(candidate)
            if inspected is None:
                raise FileNotFoundError(
                    f"Explicit dataset_dir '{candidate}' does not contain train/flows_tabular.parquet or a parquet sample."
                )
            resolved_dataset_dir, is_sample_mode, sample_path = inspected
            mode = "custom"

        # 2. TOPIC04_DATA_DIR environment variable (fail-closed)
        elif os.environ.get("TOPIC04_DATA_DIR", "").strip():
            env_dir = Path(os.environ["TOPIC04_DATA_DIR"].strip())
            if not env_dir.exists():
                raise FileNotFoundError(f"TOPIC04_DATA_DIR does not exist: {env_dir}")
            inspected = cls._inspect_dataset_candidate(env_dir)
            if inspected is None:
                raise FileNotFoundError(
                    f"TOPIC04_DATA_DIR '{env_dir}' does not contain train/flows_tabular.parquet or a parquet sample."
                )
            resolved_dataset_dir, is_sample_mode, sample_path = inspected
            mode = "custom"

        # 3. Auto-discovery (unless prefer_sample forces sample fallback directly)
        if resolved_dataset_dir is None and not prefer_sample:
            # 3a. Kaggle mount (supports real directory layout or mocked Path.exists)
            inspected = cls._inspect_dataset_candidate(KAGGLE_DATASET_DIR)
            if inspected is not None:
                resolved_dataset_dir, is_sample_mode, sample_path = inspected
                mode = "kaggle"
            elif KAGGLE_DATASET_DIR.exists():
                nested = KAGGLE_DATASET_DIR / "NF-UQ-NIDS-v2"
                resolved_dataset_dir = (
                    nested if nested.exists() and not (KAGGLE_DATASET_DIR / "train").exists() else KAGGLE_DATASET_DIR
                )
                is_sample_mode = False
                sample_path = None
                mode = "kaggle"
            else:
                # 3b. Local full dataset
                local_candidates = []
                if repo_root is not None:
                    local_candidates.append(repo_root.parent / "NF-UQ-NIDS-v2")
                if LOCAL_FALLBACK_DATASET_DIR not in local_candidates:
                    local_candidates.append(LOCAL_FALLBACK_DATASET_DIR)

                for local_cand in local_candidates:
                    inspected = cls._inspect_dataset_candidate(local_cand)
                    if inspected is not None:
                        resolved_dataset_dir, is_sample_mode, sample_path = inspected
                        mode = "local_full"
                        break

        # 4. Repository sample fallback
        if resolved_dataset_dir is None:
            if repo_root is not None and (repo_root / SAMPLE_REL_PATH).is_file():
                sample_file = repo_root / SAMPLE_REL_PATH
                resolved_dataset_dir = sample_file
                sample_path = sample_file
                is_sample_mode = True
                mode = "local_sample"
            else:
                raise FileNotFoundError(
                    "Unable to locate NF-UQ-NIDS-v2 dataset (Kaggle, local, or sample fallback)."
                )

        # Locate manifest without side effects
        manifest_path: Optional[Path] = None
        if not is_sample_mode:
            candidate_manifest = resolved_dataset_dir / "metadata_manifest.json"
            if candidate_manifest.is_file():
                manifest_path = candidate_manifest
        elif sample_path is not None:
            candidate_manifest = sample_path.parent / "nfuq_manifest_and_audit.json"
            if candidate_manifest.is_file():
                manifest_path = candidate_manifest

        return ResolvedPaths(
            mode=mode or "custom",
            is_sample_mode=is_sample_mode,
            dataset_dir=resolved_dataset_dir,
            checkpoint_dir=resolved_checkpoint_dir,
            repo_root=repo_root,
            sample_path=sample_path,
            manifest_path=manifest_path,
            schema_path=schema_path,
        )


@dataclass
class ExperimentConfig:
    """Reproducible experiment hyperparameters and runtime configuration."""

    seed: int = 42
    max_train_rows: Optional[int] = None
    device: str = "auto"
    seq_len: int = 8
    stride: int = 4
    batch_size: int = 256
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_train_rows is not None and self.max_train_rows < 1:
            raise ValueError(f"max_train_rows must be >= 1 or None, got {self.max_train_rows}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_environment_info() -> Dict[str, str]:
    """Return Python, core ML library versions, and CUDA availability metadata."""
    try:
        xgb_version = importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        xgb_version = "not_installed"

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "none"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pyarrow": str(pyarrow.__version__),
        "scikit_learn": str(sklearn.__version__),
        "torch": str(torch.__version__),
        "joblib": str(joblib.__version__),
        "xgboost": str(xgb_version),
        "cuda_available": str(cuda_available),
        "cuda_device_count": str(cuda_device_count),
        "cuda_device_name": str(cuda_device_name),
    }


def set_global_seed(seed: int = 42, deterministic_torch: bool = False) -> Dict[str, str]:
    """Seed Python, NumPy, and PyTorch RNGs and return environment provenance."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = get_environment_info()
    env["seed"] = str(seed)
    return env
