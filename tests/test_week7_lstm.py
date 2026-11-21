"""Unit tests for Week 7 Pure PyTorch LSTM Temporal Sequence Model."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.features.data_loader import NFUQDataLoader
from src.features.datasets import TemporalSequenceDataset
from src.features.transforms import FlowFeatureTransformer
from src.models.base import CheckpointManager
from src.models.lstm_sequence import LSTMEvaluator, LSTMSequenceModel, LSTMTrainer
from src.utils.config import PathResolver, set_global_seed


def test_week7_lstm_train_eval_and_checkpoint_parity(tmp_path: Path):
    set_global_seed(42)
    paths = PathResolver.resolve(checkpoint_dir=tmp_path, prefer_sample=True)
    splits = NFUQDataLoader(paths).load_tabular_splits()

    transformer = FlowFeatureTransformer().fit(splits["train"])
    train_ds = TemporalSequenceDataset(splits["train"], transformer, seq_len=4, stride=2)
    val_ds = TemporalSequenceDataset(splits["val"], transformer, seq_len=4, stride=2)
    test_ds = TemporalSequenceDataset(splits["test"], transformer, seq_len=4, stride=2)

    assert len(train_ds) > 0 and len(test_ds) > 0

    model = LSTMSequenceModel(input_dim=40, hidden_dim=16, num_layers=1, seq_len=4, stride=2, device="cpu")
    history = LSTMTrainer(model, epochs=2, batch_size=16, lr=1e-3, seed=42, device="cpu").fit(train_ds, val_ds)
    assert len(history["train_loss"]) == 2

    evaluator = LSTMEvaluator()
    metrics = evaluator.evaluate(model, test_ds)
    for k in ("roc_auc", "pr_auc", "f1", "seq_len", "stride", "inference_latency_ms_per_seq"):
        assert k in metrics and np.isfinite(float(metrics[k]))
    assert metrics["seq_len"] == 4 and metrics["stride"] == 2

    orig_probs = model.predict_proba(test_ds)

    mgr = CheckpointManager(tmp_path)
    mgr.save_checkpoint(
        "week7_lstm_temporal",
        {"state_dict": model.state_dict()},
        {"seed": 42, "seq_len": 4, "stride": 2, "hidden_dim": 16, "num_layers": 1, "metrics": metrics},
    )
    loaded, meta = mgr.load_checkpoint("week7_lstm_temporal")
    reloaded = LSTMSequenceModel(
        input_dim=40,
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        seq_len=meta["seq_len"],
        stride=meta["stride"],
        device="cpu",
    )
    reloaded.load_state_dict(loaded["state_dict"])
    assert np.allclose(orig_probs, reloaded.predict_proba(test_ds), atol=1e-5)
