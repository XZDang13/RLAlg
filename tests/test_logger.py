import numpy as np
import pytest
import torch

import RLAlg.logger as logger_module
from RLAlg.logger import MetricsTracker, WandbLogger


def test_metrics_tracker_list_accepts_python_and_tensor_scalars():
    tracker = MetricsTracker()
    tracker.add_list_metrics("loss")

    tracker.add_values("loss", 1.5)
    tracker.add_values("loss", torch.tensor(2.5))
    tracker.add_values("loss", np.array(3.5, dtype=np.float32))

    assert tracker._storages["loss"] == [1.5, 2.5, 3.5]


def test_metrics_tracker_list_rejects_non_scalar_tensor():
    tracker = MetricsTracker()
    tracker.add_list_metrics("loss")

    with pytest.raises(ValueError, match="scalar tensor"):
        tracker.add_values("loss", torch.tensor([1.0, 2.0]))


def test_wandb_logger_raises_when_wandb_unavailable(monkeypatch):
    monkeypatch.setattr(logger_module, "wandb", None)

    with pytest.raises(ImportError, match="wandb is not installed"):
        WandbLogger.init_project("proj")
