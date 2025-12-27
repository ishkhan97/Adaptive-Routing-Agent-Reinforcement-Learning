import numpy as np
import torch
from env import GridWorldEnv
from agent import DQNAgent
from baseline import shortest_path_length

def evaluate(
    agent,
    env,
    num_episodes=50,
):
    """
    Evaluate a trained agent without exploration.
    """

    episode_rewards = []
    episode_lengths = []
    successes = 0

    # Disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            steps += 1

            if terminated:
                successes += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    # Restore epsilon
    agent.epsilon = original_epsilon

    results = {
        "avg_reward": np.mean(episode_rewards),
        "avg_steps": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }

    return results

def compare_with_baseline(env, rl_results):
    """
    Compare RL agent performance with optimal shortest-path solution.
    """

    optimal_steps = shortest_path_length(
        grid_size=env.grid_size,
        start=(0, 0),
        goal=(env.grid_size - 1, env.grid_size - 1),
        obstacles=env.obstacles,
    )

    comparison = {
        "optimal_steps": optimal_steps,
        "rl_avg_steps": rl_results["avg_steps"],
        "efficiency_ratio": (
            optimal_steps / rl_results["avg_steps"]
            if optimal_steps is not None
            else None
        ),
    }

    return comparison

if __name__ == "__main__":
    print("Starting evaluation...")
    # Train environment
    train_env = GridWorldEnv()

    agent = DQNAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.n,
    )

    # Load trained weights
    agent.q_net.load_state_dict(torch.load("dqn.pt"))
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    print("Model loaded successfully")

    # New environment with different obstacles (generalization test)
    eval_env = GridWorldEnv(
        obstacles={(1, 2), (2, 3), (3, 3), (4, 1)}
    )
    print("Running evaluation loop...")
    rl_results = evaluate(agent, eval_env)
    print("Evaluation results:", rl_results)

    baseline_comparison = compare_with_baseline(eval_env, rl_results)
    print("Baseline comparison:", baseline_comparison)