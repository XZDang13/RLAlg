from typing import Dict
from collections import namedtuple
import numpy as np
import torch

BATCH = Dict[str, torch.Tensor]

class ReplayBuffer:
    def __init__(self, max_size:int, batch_size:int) -> None:
        self.episode_buffer = []
        self.max_size = max_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.episode_buffer)

    def add(self, record:namedtuple):
        self.episode_buffer.append(record)
        if len(self.episode_buffer) > self.max_size:
            del self.episode_buffer[0]

    def sample(self) -> BATCH:
        idxs = np.random.randint(0, len(self.episode_buffer), size=self.batch_size)

        batch = {}
        for idx in idxs:
            step = self.episode_buffer[idx]
            for key in step._fields:
                if key not in batch:
                    batch[key] = []

                batch[key].append(getattr(step, key))

        for key, value in batch.items():
            batch[key] = torch.as_tensor(value, dtype=torch.float32)

        return batch