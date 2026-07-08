"""
Training Loops
--------------
Training loops for DQN, SAC, and SPR agents.
Each loop supports use_resets flag to enable/disable
the periodic reset mechanism from Nikishin et al. (2022).
"""
import numpy as np
import gymnasium as gym
from buffer import ReplayBuffer
from dqn import DQNAgent
from sac import SACAgent


def train_dqn(env_name, use_resets=False, reset_interval=50000,
              max_steps=200000, batch_size=256, seed=42):

    # setup
    env = gym.make(env_name)
    np.random.seed(seed)
    torch_seed = seed

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent  = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, 1)  # action_dim=1 for discrete

    episode_rewards = []
    episode_steps   = []
    total_steps     = 0
    episode         = 0

    print(f"\n{'='*50}")
    print(f"DQN | env: {env_name} | resets: {use_resets}")
    print(f"{'='*50}")

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed+episode)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition
            buffer.add(state, [action], reward, next_state, float(done))

            # train agent
            agent.train(buffer, batch_size)

            # ── RESET LOGIC ──
            if use_resets and total_steps > 0 and total_steps % reset_interval == 0:
                agent.reset()

            # update target network every 1000 steps
            if total_steps % 1000 == 0:
                agent.update_target()

            state         = next_state
            episode_reward += reward
            total_steps   += 1

        episode += 1
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)

        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Step: {total_steps} | Episode: {episode} | Avg Reward: {avg:.1f}")

    env.close()
    return episode_steps, episode_rewards


def train_sac(env_name, use_resets=False, reset_interval=200000,
              max_steps=1000000, batch_size=256, seed=42):

    # setup
    env = gym.make(env_name)
    np.random.seed(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent  = SACAgent(state_dim, action_dim, max_action)
    buffer = ReplayBuffer(state_dim, action_dim)

    episode_rewards = []
    episode_steps   = []
    total_steps     = 0
    episode         = 0

    print(f"\n{'='*50}")
    print(f"SAC | env: {env_name} | resets: {use_resets}")
    print(f"{'='*50}")

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed+episode)
        episode_reward = 0
        done = False

        while not done:
            # random actions at start to fill buffer
            if total_steps < 10000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition
            buffer.add(state, action, reward, next_state, float(done))

            # train agent
            if total_steps >= 1000:
                agent.train(buffer, batch_size)

            # ── RESET LOGIC ──
            if use_resets and total_steps > 0 and total_steps % reset_interval == 0:
                agent.reset()

            state          = next_state
            episode_reward += reward
            total_steps    += 1

        episode += 1
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)

        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Step: {total_steps} | Episode: {episode} | Avg Reward: {avg:.1f}")

    env.close()
    return episode_steps, episode_rewards
def train_spr(env_name, use_resets=False, reset_interval=20000,
              max_steps=100000, batch_size=256, seed=42):

    env = gym.make(env_name)
    np.random.seed(seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    from spr import SPRAgent
    agent  = SPRAgent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, 1)

    episode_rewards = []
    episode_steps   = []
    total_steps     = 0
    episode         = 0

    print(f"\n{'='*50}")
    print(f"SPR | env: {env_name} | resets: {use_resets}")
    print(f"{'='*50}")

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed+episode)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, [action], reward, next_state, float(done))
            agent.train(buffer, batch_size)

            # ── RESET LOGIC ──
            if use_resets and total_steps > 0 and total_steps % reset_interval == 0:
                agent.reset()

            state          = next_state
            episode_reward += reward
            total_steps    += 1

        episode += 1
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)

        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Step: {total_steps} | Episode: {episode} | Avg Reward: {avg:.1f}")

    env.close()
    return episode_steps, episode_rewards