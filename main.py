"""
Main Experiment Runner
----------------------
Runs all experiments for the replication study of
"The Primacy Bias in Deep Reinforcement Learning"
Nikishin et al. (2022).

Usage:
    python main.py

Results saved as PNG files in the current directory.
"""
from train import train_dqn, train_sac, train_spr
from utils import plot_results


# ─── DQN Experiments ──────────────────────────────────────────
def run_dqn_experiments():

    # CartPole
    print("\nRunning CartPole...")
    steps_no_reset, rewards_no_reset = train_dqn(
        env_name="CartPole-v1",
        use_resets=False,
       max_steps=100000
    )
    steps_reset, rewards_reset = train_dqn(
        env_name="CartPole-v1",
        use_resets=True,
        max_steps=100000
    )
    plot_results(
        {
            "DQN (no resets)": (steps_no_reset, rewards_no_reset),
            "DQN + resets":    (steps_reset,    rewards_reset)
        },
        title="CartPole-v1 — DQN with and without resets",
        save_path="cartpole_results.png"
    )

    # MountainCar
    print("\nRunning MountainCar...")
    steps_no_reset, rewards_no_reset = train_dqn(
        env_name="MountainCar-v0",
        use_resets=False,
        max_steps=100000
    )
    steps_reset, rewards_reset = train_dqn(
        env_name="MountainCar-v0",
        use_resets=True,
        max_steps=100000
    )
    plot_results(
        {
            "DQN (no resets)": (steps_no_reset, rewards_no_reset),
            "DQN + resets":    (steps_reset,    rewards_reset)
        },
        title="MountainCar-v0 — DQN with and without resets",
        save_path="mountaincar_results.png"
    )


# ─── SAC Experiments ──────────────────────────────────────────
def run_sac_experiments():

    # LunarLanderContinuous
    #print("\nRunning LunarLanderContinuous-v3...")
    #steps_no_reset, rewards_no_reset = train_sac(
    #    env_name="LunarLanderContinuous-v3",
    #    use_resets=False,
    #    max_steps=200000
    #)
    #steps_reset, rewards_reset = train_sac(
    #    env_name="LunarLanderContinuous-v3",
    #    use_resets=True,
    #    max_steps=200000
    #)
    #plot_results(
    #   {
    #        "SAC (no resets)": (steps_no_reset, rewards_no_reset),
    #        "SAC + resets":    (steps_reset,    rewards_reset)
    #    },
    #    title="LunarLanderContinuous-v3 — SAC with and without resets",
    #    save_path="lunar_lander_continuous_results.png"
    #)

    # Pendulum
    #print("\nRunning Pendulum...")
    #steps_no_reset, rewards_no_reset = train_sac(
    #    env_name="Pendulum-v1",
    #    use_resets=False,
    #    max_steps=200000
    #)
    #steps_reset, rewards_reset = train_sac(
    #    env_name="Pendulum-v1",
    #    use_resets=True,
    #    max_steps=200000
    #)
    #plot_results(
    #    {
    #        "SAC (no resets)": (steps_no_reset, rewards_no_reset),
    #        "SAC + resets":    (steps_reset,    rewards_reset)
    #    },
    #    title="Pendulum-v1 — SAC with and without resets",
    #    save_path="pendulum_results.png"
    #)
    pass

#----------------------SPR Experiments -----------------------------

def run_spr_experiments():
    # CartPole-v1
    #print("\nRunning SPR on CartPole-v1..")

    #steps_no_reset, rewards_no_reset = train_spr(
    #    env_name="CartPole-v1",
    #    use_resets=False,
    #    max_steps=100000
    #)
    #steps_reset, rewards_reset = train_spr(
    #    env_name="CartPole-v1",
    #    use_resets=True,
    #    max_steps=100000
    #)
    #plot_results(
    #    {
    #        "SPR (no resets)": (steps_no_reset, rewards_no_reset),
    #        "SPR + resets":    (steps_reset, rewards_reset)
    #    },
    #    title="CartPole-v1 — SPR with and without resets",
    #    save_path="CartPole-v1_spr_results.png"
    #)

    # LunarLAnder-v3
    print("\nRunning SPR on LunarLander-v3..")

    steps_no_reset, rewards_no_reset = train_spr(
        env_name="LunarLander-v3",
        use_resets=False,
        max_steps=100000
    )
    steps_reset, rewards_reset = train_spr(
        env_name="LunarLander-v3",
        use_resets=True,
        max_steps=100000
    )
    plot_results(
        {
            "SPR (no resets)": (steps_no_reset, rewards_no_reset),
            "SPR + resets":    (steps_reset, rewards_reset)
        },
        title="LunarLander-v3 — SPR with and without resets",
        save_path="LunarLander-v3_spr_results.png"
    )

# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    #print("Starting DQN experiments...")
    #run_dqn_experiments()

    #print("\nStarting SAC experiments...")
    #run_sac_experiments()

    print("\nStarting SPR experiments...")
    run_spr_experiments()

    print("\nAll done! Check the plots.")