"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import random

from jax import random, vmap

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.agents.mbpo.transition_model import EnsembleModel
from jaxrl.agents.mbpo.transition_model import update as ensemble_update
from jaxrl.agents.mbpo.transition_model import val_loss

from jaxrl.datasets import Batch, InputNormalizationParams
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit)
def tile_and_shuffle(data, rng_keys, ensemble_size = 7):
    data = jnp.expand_dims(data, axis=3)

    tiling_pattern = [1] * data.ndim
    tiling_pattern[3] = ensemble_size
    tiled_data = jnp.tile(data, tiling_pattern)

    if len(data.shape) == 5:
        reshaped_data = tiled_data.reshape(tiled_data.shape[0] * tiled_data.shape[3], tiled_data.shape[1], *(tiled_data.shape[2], tiled_data.shape[4]))
    else:
        reshaped_data = tiled_data.reshape(tiled_data.shape[0] * tiled_data.shape[3], tiled_data.shape[1], tiled_data.shape[2])

    flattened_keys = rng_keys.reshape(-1, 2)
    shuffled_data = vmap(lambda array, prng_key: random.permutation(prng_key, array, axis=1), in_axes=(0, 0))(reshaped_data, flattened_keys)

    shuffled_data = shuffled_data.reshape(*tiled_data.shape)

    return tiled_data

@functools.partial(jax.jit)
def tile_val(data, ensemble_size = 7):
    data = jnp.expand_dims(data, axis=2)

    tiling_pattern = [1] * data.ndim
    tiling_pattern[2] = ensemble_size
    tiled_data = jnp.tile(data, tiling_pattern)
    return tiled_data


@functools.partial(jax.jit)
def fit(obs: jnp.ndarray, act: jnp.ndarray):
    obs_mu = jnp.mean(obs, axis=0)
    obs_std = jnp.std(obs, axis=0) + 1e-6
    act_mu = jnp.mean(act, axis=0)
    act_std = jnp.std(act, axis=0) + 1e-6
    return InputNormalizationParams(obs_mu=obs_mu, obs_std=obs_std,
                        act_mu=act_mu, act_std=act_std)

@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None))
@functools.partial(jax.jit, static_argnames=('ensemble_size', 'obs_size'))
def _generate_transition(key: PRNGKey, ensemble_model: Model, actor: Model, batch: Batch, input_norm_params: InputNormalizationParams, elite_indices: jnp.ndarray, ensemble_size: int, obs_size: int) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:

    batch_size = batch.observations.shape[0]
    key_normal, key_choice, key_action, key = jax.random.split(key, 4)

    parallel_apply_actor = jax.vmap(lambda obs, seed: actor.apply(actor.params, obs).sample(seed=seed))

    actor_keys = jax.random.split(key_action, num=batch_size)

    actions = parallel_apply_actor(batch.observations, actor_keys)

    norm_obs = (batch.observations - input_norm_params.obs_mu) / input_norm_params.obs_std
    norm_act = (actions - input_norm_params.act_mu) / input_norm_params.act_std

    inputs = jnp.concatenate([norm_obs, norm_act], -1)

    inputs = jnp.expand_dims(inputs, axis=1)
    inputs = jnp.tile(inputs, (1, ensemble_size, 1))

    parallel_apply_ensemble = jax.vmap(lambda x: ensemble_model.apply(ensemble_model.params, x, ret_log_var=False))

    means, vars = parallel_apply_ensemble(inputs)

    stds = jnp.sqrt(vars)

    y = means + stds * jax.random.normal(key_normal, means.shape)
    keys = jax.random.split(key_choice, batch_size)

    choose_random_model_output = jax.vmap(
        lambda key, array: array[jax.random.choice(key, elite_indices)]
    )

    y = choose_random_model_output(keys, y)

    delta_obs = y[:, :obs_size]
    next_obs = batch.observations + delta_obs
    reward = jnp.squeeze(y[:, obs_size:])
    return key, next_obs, reward, actions

@functools.partial(jax.jit, static_argnames=('ensemble_size', 'obs_size', 'batched_termination_fn'))
def _generate_transitions(
    carry: Tuple[PRNGKey, Model, Model, Batch, InputNormalizationParams, jnp.ndarray, jnp.ndarray], 
    unused,
    ensemble_size: int,
    obs_size: int,
    batched_termination_fn: Callable
) -> Tuple[Tuple[PRNGKey, Model, Model, Batch, InputNormalizationParams, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    
    del unused    
    rng, ensemble_model, actor, replay_batches, input_norm_params, elite_indices, active_mask = carry

    rng, next_obs, rewards, actions = _generate_transition(
        rng, ensemble_model, actor, replay_batches,
        input_norm_params, elite_indices, ensemble_size, obs_size
    )
    dones = jnp.zeros_like(rewards)
    masks = 1.0 - dones

    outputs = (
        replay_batches.observations, 
        actions,
        rewards,
        masks,
        dones,
        next_obs,
        active_mask
    )
    active_mask = jnp.multiply(active_mask, masks)

    replay_batches = replay_batches._replace(observations=next_obs)

    new_carry = rng, ensemble_model, actor, replay_batches, input_norm_params, elite_indices, active_mask

    return new_carry, outputs

@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))
def _update(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float, target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            soft_critic=True)
    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }

@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0))
def _ensemble_update(
    rng: PRNGKey, ensemble: Model, batch: Batch,
    input_norm_params: InputNormalizationParams
) -> Tuple[PRNGKey, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_ensemble, ensemble_info = ensemble_update(key,
                                            ensemble,
                                            batch, input_norm_params)

    return rng, new_ensemble, {
        **ensemble_info,
    }

@functools.partial(jax.vmap, in_axes=(0, 0, 0))
def _calculate_val_loss(
    ensemble: Model, val_batch: Batch,
    input_norm_params: InputNormalizationParams):

    out_val_loss = val_loss(ensemble, val_batch, input_norm_params)

    return out_val_loss



@functools.partial(jax.jit, static_argnames=('discount', 'tau', 'target_entropy', 'num_updates'))
def _do_multiple_updates(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                         temp: Model, batches: Batch, discount: float, tau: float,
                         target_entropy: float, step, num_updates: int) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
                rng, actor, critic, target_critic, temp,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), discount, tau, target_entropy)
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info
    step, rng, actor, critic, target_critic, temp, info = one_step(0, (step, rng, actor, critic, target_critic, temp, {}))
    return jax.lax.fori_loop(1, num_updates, one_step,
                             (step, rng, actor, critic, target_critic, temp, info))



@functools.partial(jax.jit, static_argnames=('num_updates'))
def _do_multiple_ensemble_updates(rng: PRNGKey, ensemble: Model, batches: Batch, val_batch: Batch, num_updates: int,
                                input_norm_params: InputNormalizationParams) -> Tuple[PRNGKey, Model, InfoDict]:
    def one_step(i, state):
        rng, ensemble, info = state
        new_rng, new_ensemble, info = _ensemble_update(
                rng, ensemble,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), input_norm_params)

        return new_rng, new_ensemble, info
    rng, ensemble, info = one_step(0, (rng, ensemble, {}))
    rng, ensemble, info = jax.lax.fori_loop(1, num_updates, one_step,
                             (rng, ensemble, info))
    
    val_mse_loss = _calculate_val_loss(ensemble, val_batch, input_norm_params)
    info['min_val_mse_loss'] = jnp.min(val_mse_loss, axis = 1)
    info['max_val_mse_loss'] = jnp.max(val_mse_loss, axis = 1)

    return rng, ensemble, info, val_mse_loss

class MBPOLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 termination_fn: Callable,

                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 ensemble_lr: float = 3e-4,

                 ensemble_size: int = 7,
                 ensemble_hidden: int = 200,

                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 num_seeds: int = 5,
                 num_elites: int = 5,
                 critic_layer_norm: bool = False) -> None:        
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.seeds = jnp.arange(seed, seed + num_seeds)
        action_dim = actions.shape[-1]
        obs_dim = observations.shape[-1]

        self.termination_fn = termination_fn

        self.batched_termination_fn = jax.jit(jax.vmap(
                    jax.vmap(self.termination_fn, in_axes=(0, 0, 0)),
                    in_axes=(0, 0, 0)                            
                ))
        
        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.discount = discount


        self.critic_layer_norm = critic_layer_norm
        
        self.ensemble_size = ensemble_size
        self.ensemble_hidden = ensemble_hidden
        self.obs_dim = obs_dim

        self.num_seeds = num_seeds
        self.num_elites = num_elites

        self.scan_fn = functools.partial(
            _generate_transitions,
            ensemble_size=self.ensemble_size,
            obs_size=self.obs_dim,
            batched_termination_fn=self.batched_termination_fn
        )

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key, ensemble_key = jax.random.split(rng, 5)
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
            actor = Model.create(actor_def,
                                 inputs=[actor_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))

            critic_def = critic_net.DoubleCritic(hidden_dims, layer_norm=self.critic_layer_norm)
            critic = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions],
                                  tx=optax.adam(learning_rate=critic_lr))
            target_critic = Model.create(
                critic_def, inputs=[critic_key, observations, actions])

            temp = Model.create(temperature.Temperature(init_temperature),
                                inputs=[temp_key],
                                tx=optax.adam(learning_rate=temp_lr))
            
            inputs = jnp.concatenate([observations, actions], -1)
            inputs = jnp.tile(inputs, (self.ensemble_size,) + (1,))
            ensemble_def = EnsembleModel(obs_size=obs_dim, 
                                        action_size=action_dim, 
                                        reward_size=1, 
                                        ensemble_size=self.ensemble_size,
                                        num_elites=5,
                                        hidden_size=self.ensemble_hidden)
            ensemble = Model.create(ensemble_def,
                                inputs=[ensemble_key, inputs],
                                tx=optax.adam(learning_rate=ensemble_lr))
            
            return actor, critic, target_critic, temp, ensemble, rng
        

        def __reset_ensemble(seed):
            rng = jax.random.PRNGKey(seed)
            rng, ensemble_key = jax.random.split(rng, 2)
            
            inputs = jnp.concatenate([observations, actions], -1)
            inputs = jnp.tile(inputs, (self.ensemble_size,) + (1,))
            ensemble_def = EnsembleModel(obs_size=obs_dim, 
                                        action_size=action_dim, 
                                        reward_size=1, 
                                        ensemble_size=self.ensemble_size,
                                        num_elites=5, 
                                        hidden_size=self.ensemble_hidden)
            ensemble = Model.create(ensemble_def,
                                inputs=[ensemble_key, inputs],
                                tx=optax.adam(learning_rate=ensemble_lr))

            return ensemble, rng

        self._reset_ensemble = jax.jit(jax.vmap(__reset_ensemble))


        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.temp, self.ensemble, self.rng = self.init_models(self.seeds)
        self.step = 1

    def save_state(self, path: str):
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))
        self.target_critic.save(os.path.join(path, 'target_critic'))
        self.temp.save(os.path.join(path, 'temp'))
        with open(os.path.join(path, 'step'), 'w') as f:
            f.write(str(self.step))

    def load_state(self, path: str):
        self.actor = self.actor.load(os.path.join(path, 'actor'))
        self.critic = self.critic.load(os.path.join(path, 'critic'))
        self.target_critic = self.target_critic.load(os.path.join(path, 'target_critic'))
        self.temp = self.temp.load(os.path.join(path, 'temp'))
        with open(os.path.join(path, 'step'), 'r') as f:
            self.step = int(f.read())


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int = 1) -> InfoDict:
        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step, num_updates)

        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        return info
    
    def update_ensemble(self, batch: Batch, val_batch: Batch, input_norm_params: InputNormalizationParams,
                         num_updates: int = 1) -> InfoDict:
        rng, ensemble, info, val_mse_loss = _do_multiple_ensemble_updates(
            self.rng, self.ensemble,
            batch, val_batch, num_updates, input_norm_params)
    
        self.rng = rng
        self.ensemble = ensemble
        return info, val_mse_loss

    def train_mbpo(
            self,
            replay_buffer, 
            FLAGS,
            holdout_ratio: float = 0.2,
            num_epochs: int = 5000,
            val_batch_size: int = 50000):

        num_datapoints = replay_buffer.size

        num_holdout = int(num_datapoints * holdout_ratio)

        replay_index_permutation = np.random.permutation(replay_buffer.size)
        train_indxs = np.random.choice(replay_index_permutation[num_holdout:], size=(num_epochs, FLAGS.batch_size))
        holdout_indxs = np.random.choice(replay_index_permutation[:num_holdout], size=(val_batch_size))

        normalization_indxs = replay_index_permutation[num_holdout:]

        replay_batches = Batch(observations=replay_buffer.observations[:, train_indxs],
                     actions=replay_buffer.actions[:, train_indxs],
                     rewards=replay_buffer.rewards[:, train_indxs],
                     masks=replay_buffer.masks[:, train_indxs],
                     next_observations=replay_buffer.next_observations[:, train_indxs])
        
        val_batch = Batch(observations=replay_buffer.observations[:, holdout_indxs],
                     actions=replay_buffer.actions[:, holdout_indxs],
                     rewards=replay_buffer.rewards[:, holdout_indxs],
                     masks=replay_buffer.masks[:, holdout_indxs],
                     next_observations=replay_buffer.next_observations[:, holdout_indxs])
    
        self.input_norm_params = vmap(fit)(replay_buffer.observations[:, normalization_indxs], replay_buffer.actions[:, normalization_indxs])

        ensemble_prng_keys = vmap(lambda rng_key: random.split(rng_key, self.ensemble_size))(self.rng)
        self.rng = ensemble_prng_keys[:, 0, :]
        configured_reshape_and_tile = partial(tile_and_shuffle, rng_keys = ensemble_prng_keys)

        replay_batches = jax.tree_map(configured_reshape_and_tile, replay_batches)
        val_batch = jax.tree_map(tile_val, val_batch)
        self.ensemble_infos, ensemble_mse_loss = self.update_ensemble(replay_batches, val_batch, self.input_norm_params, num_epochs)

        self.elite_indices = jnp.argsort(ensemble_mse_loss, axis=1)[:,:self.num_elites]

        return self.ensemble_infos

    def generate_transitions(self, replay_buffer, synthetic_buffer, batch_size: int = 1000, rollout_length: int = 1):
        replay_batches = replay_buffer.sample_parallel_multibatch(batch_size, 1)

        replay_batches = jax.tree_util.tree_map(lambda x: x.squeeze(axis=1), replay_batches)

        active_mask = jnp.ones((self.num_seeds, batch_size))
        initial_carry = self.rng, self.ensemble, self.actor, replay_batches, self.input_norm_params, self.elite_indices, active_mask

        final_carry, scan_outputs = jax.lax.scan(
            self.scan_fn,
            initial_carry,
            xs=None, 
            length=rollout_length
        )
        
        self.rng, _, _, _, _, _, _ = final_carry
        
        (observations, actions, rewards, masks, dones, next_obs, active_mask) = scan_outputs

        new_obs_shape = (observations.shape[1], observations.shape[0] * observations.shape[2], 
                    observations.shape[3])
        new_action_shape = (actions.shape[1], actions.shape[0] * actions.shape[2], 
                    actions.shape[3])
        new_reward_shape = (rewards.shape[1], rewards.shape[0] * rewards.shape[2])

        observations = np.array(observations.reshape(new_obs_shape))
        actions = np.array(actions.reshape(new_action_shape))
        rewards = np.array(rewards.reshape(new_reward_shape))
        masks = np.array(masks.reshape(new_reward_shape))
        dones = np.array(dones.reshape(new_reward_shape))
        next_obs = np.array(next_obs.reshape(new_obs_shape))
        active_mask = np.array(active_mask.reshape(new_reward_shape))

        dones = self.batched_termination_fn(observations, actions, next_obs).astype(jnp.float32)
        masks = 1.0 - dones

        synthetic_buffer.batch_insert(observations, actions, rewards, masks, dones, next_obs)
        return synthetic_buffer

    def reset_sac(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, _, self.rng = self.init_models(self.seeds)

    def reset_ensemble(self):
        self.ensemble, self.rng = self._reset_ensemble(self.seeds)