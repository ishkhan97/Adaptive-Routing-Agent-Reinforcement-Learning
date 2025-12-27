import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(results, label=None):
    rewards = results["episode_rewards"]

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label=label or "Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Over Time")
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reward_variance(results_list):
    plt.figure(figsize=(8, 4))

    for results in results_list:
        rewards = results["episode_rewards"]
        label = (
            f"seed={results['seed']}, step_reward={results['reward_step']}"
        )
        plt.plot(rewards, alpha=0.7, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Variance Across Runs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_episode_length(results):
    lengths = results["episode_lengths"]

    plt.figure(figsize=(8, 4))
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Episode Length Over Time")
    plt.tight_layout()
    plt.show()
