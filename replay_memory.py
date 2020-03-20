import random
import torch
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, index, batch_size=4):
        return self.memory[index-batch_size:index]

    def sample_run(self, index):
        i = index
        while i > 0:
            val = Transition(*zip(self.memory[i])).reward
            val = int(val[0])
            if val == -1:
                return self.memory[i+1:index]
            else:
                i -= 1
        return self.memory[:index]

    def __len__(self):
        return len(self.memory)

