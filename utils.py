"""
Utilities
---------
Smoothing and plotting functions for visualizing
training curves across experiments.
"""
import numpy as np
import matplotlib.pyplot as plt


def smooth(data, window=10):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def plot_results(results, title, save_path=None):
    plt.figure(figsize=(10, 5))

    for label, (steps, rewards) in results.items():
        smoothed = smooth(rewards)
        plt.plot(steps, smoothed, label=label)

    plt.xlabel("Environment Steps")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f">> Plot saved to {save_path}")

    plt.show()