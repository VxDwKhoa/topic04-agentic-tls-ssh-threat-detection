"""
Week 7 Pure PyTorch LSTM Temporal Sequence Model, Trainer & Evaluator.
Topic 04 — Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response.
Adheres strictly to:
- Section 10 & 20 (Week 7): Pure temporal sequence model (`torch.nn.LSTM`) on 40 L3/L4 flow
  features grouped by `src_host_token` and ordered by `flow_order_id`, with explicit `seq_len`
  and `stride` metadata, and NO graph features or GNN layers.
- Reproducibility & Checkpointing: `state_dict` saved to `.pt` + `_metadata.json` with seed,
  library versions, `seq_len`, `stride`, and inference latency.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.base import BaseEvaluator, BaseThreatModel, BaseTrainer
from src.utils.config import set_global_seed


class PureLSTMNetwork(nn.Module):
    """Pure PyTorch nn.LSTM temporal sequence classifier (no graph layers)."""

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout) if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.head(last_hidden).squeeze(-1)
        return logits


class LSTMSequenceModel(BaseThreatModel):
    """
    OOP wrapper around PureLSTMNetwork implementing BaseThreatModel for 3D temporal
    flow sequences `(N, seq_len, 40)` with explicit `seq_len` and `stride` tracking.
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 8,
        stride: int = 4,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name="week7_lstm_sequence", task_type="sequence_binary")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.threshold = float(threshold)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.network = PureLSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def state_dict(self) -> Dict[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.network.eval()
        self.is_fitted = True

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
        **kwargs: Any,
    ) -> "LSTMSequenceModel":
        trainer = LSTMTrainer(
            model=self,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=str(self.device),
        )
        if y is not None and (isinstance(X, (np.ndarray, torch.Tensor, list, tuple)) or not hasattr(X, "__getitem__")):
            x_tensor = torch.as_tensor(X, dtype=torch.float32).detach().cpu()
            y_tensor = torch.as_tensor(y, dtype=torch.float32).detach().cpu()
            ds = TensorDataset(x_tensor, y_tensor)
            trainer.fit(ds)
        else:
            trainer.fit(X)
        return self

    def _make_inference_loader(self, X: Any, batch_size: int = 256) -> DataLoader:
        if isinstance(X, DataLoader):
            return X
        use_cuda = self.device.type == "cuda"
        if isinstance(X, np.ndarray):
            tensor_x = torch.as_tensor(X, dtype=torch.float32)
            return DataLoader(
                TensorDataset(tensor_x),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
            )
        if isinstance(X, torch.Tensor):
            pin_mem = use_cuda and not X.is_cuda
            return DataLoader(
                TensorDataset(X.to(torch.float32)),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_mem,
            )
        # PyTorch Dataset (e.g. TemporalSequenceDataset)
        is_cpu_ds = True
        if len(X) > 0:
            sample = X[0]
            first_x = sample[0] if isinstance(sample, (list, tuple)) else sample
            if isinstance(first_x, torch.Tensor) and first_x.is_cuda:
                is_cpu_ds = False
        return DataLoader(
            X,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(use_cuda and is_cpu_ds),
        )

    @torch.no_grad()
    def predict_proba(self, X: Any, batch_size: int = 256, **kwargs: Any) -> np.ndarray:
        self.network.eval()
        loader = self._make_inference_loader(X, batch_size=batch_size)
        non_blocking = self.device.type == "cuda"
        all_probs: List[np.ndarray] = []
        for batch in loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
            logits = self.network(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            if probs.ndim == 0:
                probs = probs.reshape(1)
            all_probs.append(probs.astype(np.float64))
        if not all_probs:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(all_probs, axis=0)

    def predict(self, X: Any, threshold: Optional[float] = None, batch_size: int = 256, **kwargs: Any) -> np.ndarray:
        thr = float(threshold) if threshold is not None else self.threshold
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs >= thr).astype(np.int64)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "threshold": self.threshold,
            "device": str(self.device),
            "graph_features_used": False,
            "is_fitted": self.is_fitted,
        }


class LSTMTrainer(BaseTrainer):
    """Deterministic PyTorch trainer for LSTMSequenceModel (CPU & Kaggle CUDA GPU)."""

    def __init__(
        self,
        model: LSTMSequenceModel,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.seed = int(seed)
        self.device = torch.device(device) if device else model.device
        self.model.device = self.device
        self.model.network.to(self.device)

    @staticmethod
    def _extract_labels(dataset: Any) -> np.ndarray:
        if hasattr(dataset, "y"):
            if dataset.y is None:
                raise ValueError("Dataset labels (`dataset.y`) cannot be None for LSTM training.")
            y_val = dataset.y
            if isinstance(y_val, torch.Tensor):
                return y_val.detach().cpu().numpy().astype(np.float32, copy=False)
            return np.asarray(y_val, dtype=np.float32)
        if hasattr(dataset, "tensors") and len(getattr(dataset, "tensors", ())) >= 2:
            y_val = dataset.tensors[1]
            if isinstance(y_val, torch.Tensor):
                return y_val.detach().cpu().numpy().astype(np.float32, copy=False)
            return np.asarray(y_val, dtype=np.float32)
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError("Dataset items must yield (X, y) pairs for LSTM training.")
            lbl = item[1]
            if isinstance(lbl, torch.Tensor):
                labels.append(float(lbl.detach().cpu().item()))
            else:
                labels.append(float(lbl))
        return np.asarray(labels, dtype=np.float32)

    def fit(self, train_dataset: Any, val_dataset: Optional[Any] = None) -> Dict[str, Any]:
        """Alias for train() matching PyTorch / E2E test conventions."""
        return self.train(train_dataset, val_dataset=val_dataset)

    def train(self, train_dataset: Any, val_dataset: Optional[Any] = None) -> Dict[str, Any]:
        set_global_seed(self.seed)
        if len(train_dataset) == 0:
            raise ValueError("Cannot train LSTM on an empty TemporalSequenceDataset.")

        use_cuda = self.device.type == "cuda"
        first_item = train_dataset[0]
        first_x = first_item[0] if isinstance(first_item, (list, tuple)) else first_item
        is_cpu_train = not (isinstance(first_x, torch.Tensor) and first_x.is_cuda)
        pin_mem = use_cuda and is_cpu_train

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=pin_mem,
        )

        # Class-imbalance pos_weight computed strictly from train_dataset
        y_train = self._extract_labels(train_dataset)
        n_pos = float(np.sum(y_train == 1.0))
        n_neg = float(np.sum(y_train == 0.0))
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([min(n_neg / n_pos, 20.0)], dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            self.model.network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        t0 = time.perf_counter()

        for _ in range(self.epochs):
            self.model.network.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, dtype=torch.float32, non_blocking=use_cuda)
                yb = yb.to(self.device, dtype=torch.float32, non_blocking=use_cuda)

                optimizer.zero_grad()
                logits = self.model.network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.network.parameters(), self.grad_clip_norm)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            history["train_loss"].append(epoch_loss / max(1, n_batches))

            if val_dataset is not None and len(val_dataset) > 0:
                val_first = val_dataset[0]
                val_x = val_first[0] if isinstance(val_first, (list, tuple)) else val_first
                is_cpu_val = not (isinstance(val_x, torch.Tensor) and val_x.is_cuda)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size * 2,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=(use_cuda and is_cpu_val),
                )
                self.model.network.eval()
                v_loss = 0.0
                v_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, dtype=torch.float32, non_blocking=use_cuda)
                        yb = yb.to(self.device, dtype=torch.float32, non_blocking=use_cuda)
                        logits = self.model.network(xb)
                        v_loss += float(criterion(logits, yb).item())
                        v_batches += 1
                history["val_loss"].append(v_loss / max(1, v_batches))

        self.model.is_fitted = True
        self.model.network.eval()
        return {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "epochs": self.epochs,
            "train_seconds": float(time.perf_counter() - t0),
            "n_train_sequences": int(len(train_dataset)),
            "device": str(self.device),
        }


class LSTMEvaluator(BaseEvaluator):
    """Evaluator for Week 7 PyTorch LSTM sequence classification & inference latency."""

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        if len(np.unique(y_true)) < 2:
            return float(np.mean(y_true == 1))
        return float(average_precision_score(y_true, y_score))

    @staticmethod
    def _extract_labels(dataset: Any) -> np.ndarray:
        if hasattr(dataset, "y"):
            if dataset.y is None:
                raise ValueError("Dataset labels (`dataset.y`) cannot be None for LSTM evaluation.")
            y_val = dataset.y
            if isinstance(y_val, torch.Tensor):
                return y_val.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(y_val, dtype=np.int64)
        if hasattr(dataset, "tensors") and len(getattr(dataset, "tensors", ())) >= 2:
            y_val = dataset.tensors[1]
            if isinstance(y_val, torch.Tensor):
                return y_val.detach().cpu().numpy().astype(np.int64, copy=False)
            return np.asarray(y_val, dtype=np.int64)
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError("Dataset items must yield (X, y) pairs for LSTM evaluation.")
            lbl = item[1]
            if isinstance(lbl, torch.Tensor):
                labels.append(int(lbl.detach().cpu().item()))
            else:
                labels.append(int(lbl))
        return np.asarray(labels, dtype=np.int64)

    def evaluate(
        self,
        model: BaseThreatModel,
        dataset: Any,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        y_true = self._extract_labels(dataset)
        seq_len = int(getattr(dataset, "seq_len", getattr(model, "seq_len", 8)))
        stride = int(getattr(dataset, "stride", getattr(model, "stride", 4)))
        n_seqs = int(len(y_true))
        if n_seqs == 0:
            return {
                "roc_auc": 0.5,
                "pr_auc": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "fpr": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "seq_len": seq_len,
                "stride": stride,
                "n_sequences": 0,
                "inference_latency_ms_per_seq": 0.0,
                "events_per_sec": 0.0,
            }

        is_cuda = hasattr(model, "device") and getattr(model.device, "type", None) == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = model.predict_proba(dataset, batch_size=batch_size)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - t0, 1e-9)

        preds = (probs >= getattr(model, "threshold", 0.5)).astype(np.int64)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "roc_auc": self._safe_roc_auc(y_true, probs),
            "pr_auc": self._safe_pr_auc(y_true, probs),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "fpr": fpr,
            "confusion_matrix": cm.tolist(),
            "seq_len": seq_len,
            "stride": stride,
            "n_sequences": n_seqs,
            "inference_latency_ms_per_seq": float((elapsed / max(1, n_seqs)) * 1000.0),
            "events_per_sec": float(n_seqs / elapsed),
        }
