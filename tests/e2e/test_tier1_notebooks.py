"""
Tier 1 E2E Tests: Notebook Existence, nbformat v4 Validation, OOP Cell Separation,
Kaggle/Local Path Fallback, Provenance Logging, and Notebook <-> experiments/ Parity.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import nbformat
import pytest

from .conftest import (
    EXPERIMENTS_DIR,
    KAGGLE_DATASET_DIR,
    NOTEBOOK_DIR,
    NOTEBOOK_FILENAMES,
)


def _read_code_cells(nb_path: Path) -> List[str]:
    nb = nbformat.read(str(nb_path), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


@pytest.mark.parametrize("nb_name", NOTEBOOK_FILENAMES)
def test_notebook_exists_in_notebook_dir(nb_name: str):
    path = NOTEBOOK_DIR / nb_name
    assert path.exists(), f"Missing primary notebook: {path}"


@pytest.mark.parametrize("nb_name", NOTEBOOK_FILENAMES)
def test_notebook_exists_in_experiments_mirror_dir(nb_name: str):
    path = EXPERIMENTS_DIR / nb_name
    assert path.exists(), f"Missing mirrored notebook: {path}"


def test_all_notebooks_valid_nbformat_v4_and_non_empty():
    for base_dir in (NOTEBOOK_DIR, EXPERIMENTS_DIR):
        for nb_name in NOTEBOOK_FILENAMES:
            path = base_dir / nb_name
            assert path.exists(), f"Missing notebook: {path}"
            nb = nbformat.read(str(path), as_version=4)
            nbformat.validate(nb)
            code_cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
            assert len(code_cells) >= 6, f"{path} should separate OOP stages into >= 6 code cells"


def test_notebooks_exact_cell_parity_between_notebook_and_experiments():
    for nb_name in NOTEBOOK_FILENAMES:
        primary = NOTEBOOK_DIR / nb_name
        mirror = EXPERIMENTS_DIR / nb_name
        assert primary.exists(), f"Missing primary notebook: {primary}"
        assert mirror.exists(), f"Missing mirrored notebook: {mirror}"

        nb_p = nbformat.read(str(primary), as_version=4)
        nb_m = nbformat.read(str(mirror), as_version=4)
        assert len(nb_p.cells) == len(nb_m.cells), f"Cell count mismatch for {nb_name}"
        for idx, (cp, cm) in enumerate(zip(nb_p.cells, nb_m.cells)):
            assert cp.cell_type == cm.cell_type, f"{nb_name} cell {idx} type mismatch"
            assert cp.source == cm.source, f"{nb_name} cell {idx} source mismatch between Notebook/ and experiments/"


def test_notebooks_separate_oop_classes_into_distinct_code_cells():
    for nb_name in NOTEBOOK_FILENAMES:
        path = NOTEBOOK_DIR / nb_name
        assert path.exists(), f"Missing notebook: {path}"
        code_cells = _read_code_cells(path)

        cells_with_classes: List[List[str]] = []
        for src in code_cells:
            # Strip IPython magics/shell escapes before AST parsing
            cleaned = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
            )
            tree = ast.parse(cleaned)
            class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
            if class_names:
                cells_with_classes.append(class_names)

        # Must separate distinct OOP roles (Config/Path, Loader/Privacy, Transformer/Dataset, Model/Trainer/Evaluator)
        assert len(cells_with_classes) >= 4, (
            f"{nb_name} must define OOP classes across >= 4 distinct code cells; got {cells_with_classes}"
        )


def test_notebooks_contain_kaggle_and_local_fallback_paths():
    for nb_name in NOTEBOOK_FILENAMES:
        path = NOTEBOOK_DIR / nb_name
        assert path.exists(), f"Missing notebook: {path}"
        full_code = "\n".join(_read_code_cells(path))
        assert KAGGLE_DATASET_DIR in full_code or "vxdwkhoa/Dataset_NF/NF-UQ-NIDS-v2" in full_code, (
            f"{nb_name} must configure Kaggle dataset path"
        )
        assert "NF-UQ-NIDS-v2" in full_code
        assert "nfuq_sample_1500.parquet" in full_code, f"{nb_name} must support sample parquet fallback"
        assert "Checkpoint" in full_code, f"{nb_name} must resolve Checkpoint directory"


def test_notebooks_log_seed_and_versions_and_have_no_l7_hardcoding():
    forbidden_l7_tokens = (
        "http_body",
        "http_headers_raw",
        "ssh_terminal_keystrokes",
        "decrypted_tls_payload",
        "tls_session_keys",
    )
    for nb_name in NOTEBOOK_FILENAMES:
        path = NOTEBOOK_DIR / nb_name
        assert path.exists(), f"Missing notebook: {path}"
        full_code = "\n".join(_read_code_cells(path))
        lower_code = full_code.lower()
        assert "seed" in lower_code and "42" in full_code, f"{nb_name} must explicitly record seed"
        assert "__version__" in full_code or "version" in lower_code, f"{nb_name} must log library versions"
        for tok in forbidden_l7_tokens:
            assert tok not in lower_code, f"{nb_name} references prohibited L7 field '{tok}'"


def test_week7_notebook_explicitly_declares_seq_len_stride_and_no_graph_features():
    path = NOTEBOOK_DIR / "week7_pytorch_lstm_temporal.ipynb"
    assert path.exists(), f"Missing notebook: {path}"
    full_code = "\n".join(_read_code_cells(path))
    assert "seq_len" in full_code
    assert "stride" in full_code
    assert "LSTM" in full_code
    for forbidden_graph in ("GraphConv", "GCNConv", "GATConv", "new_edge_score", "adjacency_matrix"):
        assert forbidden_graph not in full_code, f"Week 7 notebook must not include graph feature/layer '{forbidden_graph}'"
