from typing import Optional

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers
from jaxrl.wrappers import VideoRecorder, SequentialMultiEnvWrapper
import jax.numpy as jnp
from jaxrl.datasets import Batch

def combine_batches(batch1, batch2):
    if batch1 is None:
        return batch2
    elif batch2 is None:
        return batch1
    else:
        combined_observations = jnp.concatenate([batch1.observations, batch2.observations], axis=2)
        combined_actions = jnp.concatenate([batch1.actions, batch2.actions], axis=2)
        combined_rewards = jnp.concatenate([batch1.rewards, batch2.rewards], axis=2)
        combined_masks = jnp.concatenate([batch1.masks, batch2.masks], axis=2)
        combined_next_observations = jnp.concatenate([batch1.next_observations, batch2.next_observations], axis=2)
        
        return Batch(
            observations=combined_observations,
            actions=combined_actions,
            rewards=combined_rewards,
            masks=combined_masks,
            next_observations=combined_next_observations
        )


def make_one_env(env_name: str,
                 seed: int,
                 save_folder: Optional[str] = None,
                 add_episode_monitor: bool = True,
                 action_repeat: int = 1,
                 frame_stack: int = 1,
                 from_pixels: bool = False,
                 pixels_only: bool = True,
                 image_size: int = 84,
                 sticky: bool = False,
                 gray_scale: bool = False,
                 flatten: bool = True,
                 **gym_kwargs) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name, **gym_kwargs)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = VideoRecorder(env, save_folder=save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def make_env(env_name: str,
             seed: int,
             eval_episodes: Optional[int] = None,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             eval_episode_length: int = 1000,
             num_envs: Optional[int] = None,
             gym_kwargs: Optional[dict] = None) -> gym.Env:

    if gym_kwargs is None:
        gym_kwargs = {}  # Initialize as empty dict if not provided
    
    if num_envs is None:
        return make_one_env(env_name, seed, save_folder, add_episode_monitor, action_repeat, frame_stack, from_pixels, pixels_only, image_size, sticky, gray_scale, flatten, **gym_kwargs)
    else:
        env_fn_list = [lambda: make_one_env(env_name, seed+i, save_folder, add_episode_monitor, action_repeat, frame_stack,
                                            from_pixels, pixels_only, image_size, sticky, gray_scale, flatten, **gym_kwargs)
                        for i in range(num_envs)]
        return SequentialMultiEnvWrapper(env_fn_list)
