"""
DQN Agent
---------
Deep Q-Network implementation with periodic reset mechanism.
Network: 3 layers, 256 units, ReLU activations.
Reset: entire network reinitialized every 50,000 steps.
Reference: Mnih et al. (2015), Nikishin et al. (2022).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ─── Neural Network ───────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)  # one Q value per action
        )

    def forward(self, x):
        return self.network(x)


# ─── DQN Agent ────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):

        self.action_dim    = action_dim
        self.gamma         = gamma        # discount factor
        self.epsilon       = epsilon      # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # two networks — online and target
        self.q_network     = QNetwork(state_dim, action_dim)
        self.target_network= QNetwork(state_dim, action_dim)
        self.optimizer     = optim.Adam(self.q_network.parameters(), lr=lr)

        # copy weights to target network
        self.update_target()

    def select_action(self, state):
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)  # explore
        state  = torch.FloatTensor(state).unsqueeze(0)
        q_vals = self.q_network(state)
        return q_vals.argmax().item()                  # exploit

    def train(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return  # not enough data yet

        # sample from buffer
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # convert to tensors
        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions).squeeze()
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # target Q values (TD target)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        # loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        # copy online network weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def reset(self):
        # ── THE KEY PAPER CONTRIBUTION ──
        # reset ALL network weights but buffer is preserved outside
        self.q_network      = QNetwork(self.q_network.network[0].in_features,
                                       self.q_network.network[-1].out_features)
        self.target_network = QNetwork(self.q_network.network[0].in_features,
                                       self.q_network.network[-1].out_features)
        self.optimizer      = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.update_target()
        print(">> Agent reset! Buffer preserved.")