from typing import Optional, Union

import numpy as np
import torch
try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - exercised via monkeypatch in tests
    wandb = None

class WandbLogger:
    @staticmethod
    def _require_wandb():
        if wandb is None:
            raise ImportError("wandb is not installed. Install wandb to use WandbLogger.")

    @staticmethod
    def init_project(project_name:str, name:str|None=None, config:dict|None=None):
        WandbLogger._require_wandb()
        wandb.init(project=project_name, name=name, config=config)
        
    @staticmethod
    def log_metrics(metrics:dict, step:int):
        WandbLogger._require_wandb()
        wandb.log(metrics, step=step)

    @staticmethod
    def finish_project():
        WandbLogger._require_wandb()
        wandb.finish()


class MetricsTracker:
    List = 0
    Tensor = 1

    def __init__(self):
        self._storages = {}
        self._types = {}
    def add_batch_metrics(self, names: str, size:int):
        self._storages[names] = torch.zeros(size, dtype=torch.float32)
        self._types[names] = MetricsTracker.Tensor

    def add_list_metrics(self, names: str):
        self._storages[names] = []
        self._types[names] = MetricsTracker.List

    def add_values(self, name: str, values: Union[torch.Tensor, np.ndarray, int, float]):
        if isinstance(values, torch.Tensor):
            values = values.cpu()
        
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).float()

        if self._types[name] == MetricsTracker.List:
            if isinstance(values, torch.Tensor):
                if values.numel() != 1:
                    raise ValueError(f"List metric '{name}' requires a scalar tensor, got shape {tuple(values.shape)}.")
                scalar = values.item()
            elif isinstance(values, (int, float)):
                scalar = values
            else:
                raise TypeError(f"Unsupported value type for list metric '{name}': {type(values)}")
            self._storages[name].append(float(scalar))
        elif self._types[name] == MetricsTracker.Tensor:
            self._storages[name] += values 

    def reset(self, name: str, indices: Optional[torch.Tensor] = None):
        if self._types[name] == MetricsTracker.List:
            self._storages[name] = []
        elif self._types[name] == MetricsTracker.Tensor:
            if indices is None:
                indices = torch.ones_like(self._storages[name], dtype=torch.bool)
            
            self._storages[name][indices] = 0.0

    def get_mean(self, name: str, terminate: Optional[torch.Tensor] = None):
        value = 0

        if self._types[name] == MetricsTracker.List:
            value = np.mean(self._storages[name])
        elif self._types[name] == MetricsTracker.Tensor:
            if terminate is None:
                terminate = torch.ones_like(self._storages[name], dtype=torch.bool)

            indices = torch.nonzero(terminate, as_tuple=False).cpu().squeeze(-1)
            value = self._storages[name][indices].mean().item()

        return value 
