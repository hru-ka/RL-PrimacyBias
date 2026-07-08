"""
SAC Agent
---------
Soft Actor-Critic implementation with periodic reset mechanism.
Uses double Q-learning and automatic entropy tuning.
Reset: all networks reinitialized every 200,000 steps.
Reference: Haarnoja et al. (2018), Nikishin et al. (2022).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ─── Actor Network ────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean    = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x       = self.network(state)
        mean    = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std     = log_std.exp()

        # reparameterization trick
        normal  = torch.distributions.Normal(mean, std)
        z       = normal.rsample()
        action  = torch.tanh(z) * self.max_action

        # log probability
        log_prob = normal.log_prob(z) - torch.log(
            1 - action.pow(2) / self.max_action**2 + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ─── Critic Network ───────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 (double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ─── SAC Agent ────────────────────────────────────────────────
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma      = gamma        # discount factor
        self.tau        = tau          # soft update rate
        self.max_action = max_action
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # networks
        self.actor          = Actor(state_dim, action_dim, max_action)
        self.critic         = Critic(state_dim, action_dim)
        self.critic_target  = Critic(state_dim, action_dim)

        # copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # entropy temperature (alpha)
        self.log_alpha  = torch.zeros(1, requires_grad=True)
        self.alpha      = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy  = -action_dim

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor(state)
        return action.detach().numpy()[0]

    def train(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return

        # sample from buffer
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # ── Critic update ──
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update ──
        new_actions, log_prob = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha update ──
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # ── Soft update target ──
        for param, target_param in zip(self.critic.parameters(),
                                       self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset(self):
        # ── THE KEY PAPER CONTRIBUTION ──
        # reset all networks, keep buffer outside
        self.actor         = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic        = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha     = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        print(">> SAC Agent reset! Buffer preserved.")