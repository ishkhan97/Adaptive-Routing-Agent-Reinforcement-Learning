# Adaptive Routing with Reinforcement Learning

This project investigates how reinforcement learning agents learn navigation and routing policies under varying reward structures, environment complexity, and training regimes.

Rather than optimizing for benchmark performance, the focus is on understanding learning dynamics, stability, and generalization.

## Problem Setting

We study a grid-based routing task with obstacles and movement costs. The agent must navigate from a fixed start location to a goal while minimizing path length and avoiding obstacles.

This setting captures core challenges common to routing, planning, and decision-making systems.

## Methods

- Custom Gymnasium GridWorld environment
- Deep Q-Network (DQN) implemented in PyTorch
- Experience replay and target networks for training stability
- Curriculum learning to progressively increase task difficulty

## Baselines

To contextualize RL performance, we implement a classical shortest-path solver (BFS) that computes the optimal path length when full environment information is available.

This provides an upper bound on achievable efficiency and highlights tradeoffs between planning-based and learning-based approaches.

## Evaluation

We evaluate agents using:
- Episode reward and length
- Success rate on unseen environments
- Variance across random seeds
- Efficiency relative to the shortest-path baseline

## Key Findings

- Reward shaping strongly influences convergence stability
- RL agents often learn suboptimal but robust policies
- Curriculum learning improves early training stability
- Classical planning outperforms RL in fully observable static environments, but lacks adaptability

## Takeaways

This project demonstrates that reinforcement learning should be evaluated not only by final performance, but by stability, sensitivity, and comparison to classical methods.
