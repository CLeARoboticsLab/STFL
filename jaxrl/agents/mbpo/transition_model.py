from typing import Tuple
from flax import linen as nn
from jax import numpy as jnp
import jax
from typing import Callable

from jaxrl.datasets.dataset import Batch, InputNormalizationParams
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey



class EnsembleLayerNorm(nn.Module):
    hidden_dim: int
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        def ln_fn(xi):
            return nn.LayerNorm(epsilon=self.epsilon)(xi)
        return jax.vmap(ln_fn)(x)
    
class EnsembleDense(nn.Module):
    features: int
    ensemble_size: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init,
                            (self.ensemble_size, x.shape[-1], self.features))
        bias = self.param('bias', self.bias_init,
                          (self.ensemble_size, self.features))
        return jax.vmap(jnp.matmul)(x, kernel) + bias
    
class EnsembleModel(nn.Module):
    obs_size: int
    action_size: int
    reward_size: int
    ensemble_size: int
    num_elites: int
    hidden_size: int
    act: Callable = nn.swish
    layer_norm: bool = False

    def apply_layer_norm(self, x):
        return EnsembleLayerNorm(self.hidden_size)(x) if self.layer_norm else x


    @nn.compact
    def __call__(self, x, ret_log_var: bool = False):

        output_dim = self.obs_size + self.reward_size
        init_max_logvar = 0.5 * jnp.ones((1, output_dim))
        init_min_logvar = -10. * jnp.ones((1, output_dim))
    
        x = EnsembleDense(self.hidden_size, self.ensemble_size, name = 'ed1')(x)
        x = self.apply_layer_norm(x)
        x = self.act(x)
        x = EnsembleDense(self.hidden_size, self.ensemble_size, name = 'ed2')(x)
        x = self.act(x)
        x = self.apply_layer_norm(x)
        x = EnsembleDense(self.hidden_size, self.ensemble_size, name = 'ed3')(x)
        x = self.act(x)
        x = self.apply_layer_norm(x)
        x = EnsembleDense(self.hidden_size, self.ensemble_size, name = 'ed4')(x)
        x = self.act(x)
        x = self.apply_layer_norm(x)
        x = EnsembleDense(output_dim * 2, self.ensemble_size, name = 'ed5')(x)

        max_logvar = init_max_logvar
        min_logvar = init_min_logvar

        mean, logvar = jnp.split(x, 2, axis=-1)
        logvar = max_logvar - nn.softplus(max_logvar - logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, jnp.exp(logvar)
        

def val_loss(ensemble_model: Model, val_batch: Batch,
            input_norm_params: InputNormalizationParams):
    
    val_norm_obs = (val_batch.observations - input_norm_params.obs_mu) / input_norm_params.obs_std
    val_norm_act = (val_batch.actions - input_norm_params.act_mu) / input_norm_params.act_std

    val_inputs = jnp.concatenate([val_norm_obs, val_norm_act], -1)

    val_delta_obs = val_batch.next_observations - val_batch.observations
    val_rewards =  jnp.expand_dims(val_batch.rewards, axis=-1)
    val_model_targets = jnp.concatenate([val_delta_obs, val_rewards], -1)

    def val_loss_fn(model_params: Params):

        def model_fn(inputs):
            means, logvars = ensemble_model.apply(model_params,
                                                 inputs, ret_log_var=True)
            return means, logvars
        means, logvars = jax.vmap(model_fn)(val_inputs)

        mse_loss = jnp.mean((means - val_model_targets)**2, axis=(0, 2))

        return mse_loss

    val_mse_loss = val_loss_fn(ensemble_model.params)

    return val_mse_loss


def update(key: PRNGKey, ensemble_model: Model, batch: Batch, 
           input_norm_params: InputNormalizationParams) -> Tuple[Model, InfoDict]:
    
    norm_obs = (batch.observations - input_norm_params.obs_mu) / input_norm_params.obs_std
    norm_act = (batch.actions - input_norm_params.act_mu) / input_norm_params.act_std

    inputs = jnp.concatenate([norm_obs, norm_act], -1)

    delta_obs = batch.next_observations - batch.observations
    rewards =  jnp.expand_dims(batch.rewards, axis=-1)
    model_targets = jnp.concatenate([delta_obs, rewards], -1)


    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        def model_fn(inputs):
            means, logvars = ensemble_model.apply(model_params,
                                                 inputs, ret_log_var=True)
            return means, logvars
        means, logvars = jax.vmap(model_fn)(inputs)

        inv_vars = jnp.exp(-logvars)

        mse_loss = jnp.mean(inv_vars * (means - model_targets)**2, axis=(0, 2))
        var_loss = jnp.mean(logvars, axis=(0, 2))
    
        total_loss = jnp.sum(mse_loss) + jnp.sum(var_loss)
        int_total_loss = jnp.sum(mse_loss) + jnp.sum(var_loss)

        weight_decays = {
            'ed1': 0.000025,
            'ed2': 0.00005,
            'ed3': 0.000075,
            'ed4': 0.000075,
            'ed5': 0.0001,
        }

        for layer, decay in weight_decays.items():
            weights = model_params['params'][layer]['kernel']
            total_loss += 0.5 * decay * jnp.sum(weights**2)


        ensemble_next_obs = means[:,:,:-1]
        true_next_obs = model_targets[:,:,:-1]
        differences = ensemble_next_obs - true_next_obs
        norms = jnp.linalg.norm(differences, axis=-1)
        real_norms = jnp.linalg.norm(true_next_obs, axis=-1)
        percent_errors = jnp.mean(jnp.abs(norms / real_norms) * 100, axis=(0,1))

        mean_loss = jnp.mean((means - model_targets)**2, axis=(0, 2))

        return total_loss, {
        'model_percent_error': percent_errors,
        'mbpo_model_loss': total_loss,  # Total model loss after including weight decay
        'int_total_loss': int_total_loss,  # Intermediate total loss before weight decay
        'mse_loss': jnp.sum(mse_loss),  # Total MSE loss summed over ensembles
        'unsum_mse_loss': mse_loss,  # Total MSE loss summed over ensembles

        'var_loss': jnp.sum(var_loss),  # Total variance loss summed over ensembles
        'mean_loss': mean_loss,  # Mean loss without weighting by inverse variance or mask
        'min_inv_vars': jnp.min(inv_vars),  # Minimum of inverse variances
        'max_logvars': jnp.max(logvars),  # Maximum of log variances
        'min_logvars': jnp.min(logvars),  # Minimum of log variances
        'sum_weight_decay': sum(0.5 * decay * jnp.sum(weights**2) for layer, decay in weight_decays.items())
            }

    new_model, info = ensemble_model.apply_gradient(model_loss_fn)
    info['model_gnorm'] = info.pop('grad_norm')

    return new_model, info
