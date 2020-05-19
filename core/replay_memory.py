import random
from typing import List, NamedTuple

import numpy as np


class BatchOutput(NamedTuple):
    state: List[List[float]]
    action: List[List[float]]
    repeat_n: List[int]
    repeat_idx: List[int]
    reward: List[float]
    next_state: List[List[float]]
    mask: List[bool]


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, repeat_n, repeat_idx, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, repeat_n, repeat_idx, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, repeat_n, repeat_idx, reward, next_state, done = map(np.stack, zip(*batch))
        return BatchOutput(state, action, repeat_n, repeat_idx, reward, next_state, done)

    def __len__(self):
        return len(self.buffer)
