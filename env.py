import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    """
    Simple gridworld environment for routing and navigation experiments.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=6,
        obstacles=None,
        reward_step=-1.0,
        reward_goal=10.0,
        reward_obstacle=-5.0,
        max_steps=100,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_obstacle = reward_obstacle

        # Default obstacle layout
        self.obstacles = obstacles or {(1, 1), (2, 2), (3, 1), (4, 3)}

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation space: agent(x,y), goal(x,y)
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(4,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32)

        self.steps = 0
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        self.steps += 1

        new_pos = self.agent_pos.copy()

        if action == 0:  # up
            new_pos[1] -= 1
        elif action == 1:  # down
            new_pos[1] += 1
        elif action == 2:  # left
            new_pos[0] -= 1
        elif action == 3:  # right
            new_pos[0] += 1

        reward = self.reward_step
        terminated = False
        truncated = False

        # Check bounds
        if not self._in_bounds(new_pos):
            reward += self.reward_obstacle
            new_pos = self.agent_pos  # stay in place

        # Check obstacle
        elif tuple(new_pos) in self.obstacles:
            reward += self.reward_obstacle
            new_pos = self.agent_pos  # stay in place

        self.agent_pos = new_pos

        # Check goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += self.reward_goal
            terminated = True

        # Max steps
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
            ],
            dtype=np.int32,
        )

    def _in_bounds(self, pos):
        return (
            0 <= pos[0] < self.grid_size
            and 0 <= pos[1] < self.grid_size
        )

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        for obs in self.obstacles:
            grid[obs[1], obs[0]] = "#"

        grid[self.goal_pos[1], self.goal_pos[0]] = "G"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        print("\n".join(" ".join(row) for row in grid))
        print()
