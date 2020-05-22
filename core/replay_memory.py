import random
from typing import List, NamedTuple

import numpy as np


class BatchOutput(NamedTuple):
    state: List[List[float]]
    action: List[List[float]]
    reward: List[List[float]]
    next_state: List[List[List[float]]]
    next_state_mask: List[List[float]]
    terminal: List[List[bool]]


class ReplayMemory:
    def __init__(self, capacity:int):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, next_state_mask, terminal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_state_mask, terminal)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_state_mask, terminal = map(np.stack, zip(*batch))
        return BatchOutput(state, action, reward, next_state, next_state_mask, terminal)

    def __len__(self):
        return len(self.buffer)
