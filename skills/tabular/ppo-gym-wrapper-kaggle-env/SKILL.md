---
name: tabular-ppo-gym-wrapper-kaggle-env
description: Wrap a Kaggle competitive game environment as an OpenAI Gym env with continuous action space for training PPO agents via stable-baselines3
---

# PPO Gym Wrapper for Kaggle Environments

## Overview

Kaggle game AI competitions (Halite, Kore, Lux) use `kaggle_environments` which don't conform to the Gym API. Wrapping them in a `gym.Env` subclass with defined observation and action spaces enables training with stable-baselines3 PPO (or SAC, A2C). The wrapper handles state encoding, action translation, opponent management, and episode termination.

## Quick Start

```python
import gym
from gym import spaces
import numpy as np
from kaggle_environments import make
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class KaggleGymEnv(gym.Env):
    def __init__(self, opponent="random"):
        super().__init__()
        self.env = make("kore_fleets", debug=True)
        self.opponent = opponent
        self.observation_space = spaces.Box(-1, 1, shape=(21*21*4+3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def reset(self):
        self.trainer = self.env.train([None, self.opponent])
        obs = self.trainer.reset()
        return self._encode(obs)

    def step(self, action):
        game_action = self._decode(action)
        obs, reward, done, info = self.trainer.step(game_action)
        return self._encode(obs), reward, done, info

env = Monitor(KaggleGymEnv())
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

## Workflow

1. Subclass `gym.Env`, define observation and action spaces
2. In `reset()`, create a trainer via `env.train([None, opponent])` — `None` marks the learning agent
3. In `step()`, decode continuous actions to game actions, call `trainer.step()`
4. Encode raw observations into the observation space format
5. Wrap with `Monitor` for logging, train with PPO or similar algorithm
6. Export the trained policy as a Kaggle submission agent

## Key Decisions

- **Action space**: continuous Box(-1, 1) is simpler than MultiDiscrete; decode to game actions in `step()`
- **Opponent**: start with "random", then self-play or a heuristic agent for curriculum
- **Observation shape**: flatten the grid tensor + append scalar features (turn number, total kore, ship count)
- **Reward**: use shaped rewards (delta board value) rather than sparse win/loss for faster learning
- **Self-play**: alternate the trained agent as opponent every N episodes for robustness

## References

- [Reinforcement Learning baseline in Python](https://www.kaggle.com/code/lesamu/reinforcement-learning-baseline-in-python)
