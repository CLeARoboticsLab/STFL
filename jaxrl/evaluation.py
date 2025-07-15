from typing import Dict
import numpy as np
import gym

from jaxrl.networks.common import Model
import jax.numpy as jnp
from typing import Callable
import jax
from functools import partial

def evaluate(agent, env: gym.Env, num_episodes: int, episode_length: int) -> Dict[str, float]:
    if 'brax' in str(type(env)).lower():
        print("No jax")
    else:
        n_seeds = env.num_envs
        returns = []
        for _ in range(num_episodes):
            observations, dones = env.reset(), np.array([False] * n_seeds)
            rets, length = np.zeros(n_seeds), 0
            while not dones.all():
                actions = agent.sample_actions(observations, temperature=0.0)
                prev_dones = dones
                observations, rewards, dones, infos = env.step(actions)
                rets += rewards * (1 - prev_dones)
                length += 1
                if length >= episode_length:
                    break
            returns.append(rets)
        return {'return': np.array(returns).mean(axis=0)}
