"""
Replay Buffer
-------------
Uniform experience replay buffer for storing and sampling
transitions (state, action, reward, next_state, done).
Buffer is NEVER reset between agent resets - this is the
key design choice from Nikishin et al. (2022).
"""
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100_000):
        self.max_size = max_size
        self.ptr = 0          # where to write next
        self.size = 0         # how many transitions stored

        # pre-allocate memory for everything
        self.states      = np.zeros((max_size, state_dim))
        self.actions     = np.zeros((max_size, action_dim))
        self.rewards     = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones       = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        # write at current pointer position
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        # move pointer, wrap around if full
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # pick random indices
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size