import random
import numpy as np
import torch
from env import GridWorldEnv
from agent import DQNAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    
    num_episodes=500,
    seed=42,
    reward_step=-1.0,
    learning_rate=1e-3,
):
    print("Starting training...")
    """
    Train a DQN agent on the GridWorld environment.
    """

    set_seed(seed)

    env = GridWorldEnv(reward_step=reward_step)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=learning_rate,
    )

    episode_rewards = []
    episode_lengths = []
    losses = []

    for episode in range(num_episodes):
        print(f"Episode {episode} running")
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            agent.store_transition(
                state, action, reward, next_state, done
            )

            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Steps: {steps:3d} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "seed": seed,
        "reward_step": reward_step,
        "learning_rate": learning_rate,
    }
    torch.save(agent.q_net.state_dict(), "dqn.pt")
    print("Model saved to dqn.pt")

    return results


if __name__ == "__main__":
    results = train()
