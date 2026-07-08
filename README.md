# The Primacy Bias in Deep Reinforcement Learning: A Replication Study

**Author:** Hrutuja Girish Kargirwar - 925351
**Course:** Reinforcement Learning - Università degli Studi di Milano-Bicocca
**Year:** 2026

---

## Overview

This project replicates the findings of Nikishin et al. (2022) 
"The Primacy Bias in Deep Reinforcement Learning" using PyTorch.
We implement three RL algorithms (DQN, SAC, SPR) with and without 
periodic network resets across six Gymnasium environments.

---

## Project Structure

*RL-PrimacyBias/
├── buffer.py    # Replay buffer (uniform sampling)
├── dqn.py       # Deep Q-Network agent + reset mechanism
├── sac.py       # Soft Actor-Critic agent + reset mechanism
├── spr.py       # Simplified SPR agent + reset mechanism
├── train.py     # Training loops for all algorithms
├── utils.py     # Smoothing and plotting utilities
└── main.py      # Main experiment runner
---*

## Requirements

- Python 3.13+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

---

## Installation

```bash
# Clone the repository
git clone [your github link]
cd RL-PrimacyBias

# Install dependencies
pip install torch numpy matplotlib
pip install gymnasium
pip install "gymnasium[classic-control]"
pip install "gymnasium[box2d]"
```

---

## How to Run

### Run all experiments
```bash
python main.py
```

### Run specific algorithm only

**DQN experiments:**
```python
# In main.py, uncomment:
run_dqn_experiments()
```

**SAC experiments:**
```python
# In main.py, uncomment:
run_sac_experiments()
```

**SPR experiments:**
```python
# In main.py, uncomment:
run_spr_experiments()
```

---

## Environments

| Environment | Algorithm | Type |
|-------------|-----------|------|
| CartPole-v1 | DQN, SPR | Discrete |
| MountainCar-v0 | DQN | Discrete |
| LunarLander-v3 | SPR | Discrete |
| LunarLanderContinuous-v3 | SAC | Continuous |
| Pendulum-v1 | SAC | Continuous |
| BipedalWalker-v3 | SAC | Continuous |

---

## Key Hyperparameters

| Parameter | DQN | SAC | SPR |
|-----------|-----|-----|-----|
| Learning rate | 3e-4 | 3e-4 | 3e-4 |
| Batch size | 256 | 256 | 256 |
| Buffer size | 100,000 | 100,000 | 100,000 |
| Reset interval | 50,000 | 200,000 | 20,000 |
| Layers reset | All | All | Q-head only |
| Hidden units | 256 | 256 | 256 |
| Discount (γ) | 0.99 | 0.99 | 0.99 |

---

## Results

After running, plots are saved as PNG files:
- `cartpole_results.png`
- `mountaincar_results.png`
- `lunarlander_continuous_results.png`
- `pendulum_results.png`
- `bipedalwalker_results.png`
- `CartPole-v1_spr_results.png`
- `LunarLander-v3_spr_results.png`

---

## Reference

Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.L., & Courville, A. (2022).
The Primacy Bias in Deep Reinforcement Learning.
ICML 2022. https://arxiv.org/abs/2205.07802
