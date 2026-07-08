
"""
Simplified SPR Agent
--------------------
Self-Predictive Representations implementation with momentum
encoder, projection head, predictor head, and Q-head.
Reset: Q-head only reinitialized every 20,000 steps.
Reference: Schwarzer et al. (2021), Nikishin et al. (2022).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ─── Encoder ──────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# ─── Projection Head ──────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        return self.network(x)


# ─── Predictor Head ───────────────────────────────────────────
class PredictorHead(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.network(x)


# ─── Q Head ───────────────────────────────────────────────────
class QHead(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=2):
        super().__init__()
        self.network = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.network(x)


# ─── SPR Agent ────────────────────────────────────────────────
class SPRAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 hidden_dim=256, proj_dim=128, momentum=0.99):

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.momentum      = momentum
        self.state_dim     = state_dim
        self.hidden_dim    = hidden_dim
        self.proj_dim      = proj_dim

        # ── Online networks ──
        self.encoder    = Encoder(state_dim, hidden_dim)
        self.projector  = ProjectionHead(hidden_dim, proj_dim)
        self.predictor  = PredictorHead(proj_dim)
        self.q_head     = QHead(hidden_dim, action_dim)

        # ── Momentum (target) encoder ──
        self.momentum_encoder    = Encoder(state_dim, hidden_dim)
        self.momentum_projector  = ProjectionHead(hidden_dim, proj_dim)

        # copy weights to momentum networks
        self._copy_to_momentum()

        # freeze momentum networks
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projector.parameters():
            param.requires_grad = False

        # ── Optimizer ──
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.projector.parameters()) +
            list(self.predictor.parameters()) +
            list(self.q_head.parameters()),
            lr=lr
        )

    def _copy_to_momentum(self):
        self.momentum_encoder.load_state_dict(self.encoder.state_dict())
        self.momentum_projector.load_state_dict(self.projector.state_dict())

    def _update_momentum(self):
        # slowly update momentum encoder
        for online, target in zip(self.encoder.parameters(),
                                  self.momentum_encoder.parameters()):
            target.data = self.momentum * target.data + \
                         (1 - self.momentum) * online.data

        for online, target in zip(self.projector.parameters(),
                                  self.momentum_projector.parameters()):
            target.data = self.momentum * target.data + \
                         (1 - self.momentum) * online.data

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state  = torch.FloatTensor(state).unsqueeze(0)
        latent = self.encoder(state)
        q_vals = self.q_head(latent)
        return q_vals.argmax().item()

    def train(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return

        # sample from buffer
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions).squeeze()
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # ── Q loss (TD learning) ──
        latent      = self.encoder(states)
        q_values    = self.q_head(latent)
        current_q   = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_latent  = self.momentum_encoder(next_states)
            next_q       = self.q_head(next_latent)
            max_next_q   = next_q.max(1)[0].unsqueeze(1)
            target_q     = rewards + self.gamma * max_next_q * (1 - dones)

        q_loss = F.mse_loss(current_q, target_q)

        # ── SPR loss (self-predictive) ──
        # online: project + predict current state
        online_proj    = self.projector(latent)
        online_pred    = self.predictor(online_proj)

        # momentum: project next state (target)
        with torch.no_grad():
            momentum_next  = self.momentum_encoder(next_states)
            momentum_proj  = self.momentum_projector(momentum_next)
            # normalize targets
            momentum_proj  = F.normalize(momentum_proj, dim=-1)

        online_pred    = F.normalize(online_pred, dim=-1)

        # cosine similarity loss
        spr_loss = -(online_pred * momentum_proj).sum(dim=-1).mean()

        # ── Total loss ──
        total_loss = q_loss + 0.1 * spr_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # update momentum encoder
        self._update_momentum()

        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        # ── THE KEY PAPER CONTRIBUTION ──
        # reset only the LAST layer (Q head) as per paper
        self.q_head = QHead(self.hidden_dim, self.action_dim)

        # rebuild optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.projector.parameters()) +
            list(self.predictor.parameters()) +
            list(self.q_head.parameters()),
            lr=3e-4
        )
        print(">> SPR Agent reset! Only Q-head reinitialized. Buffer preserved.")