import jax.numpy as jnp

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        x = next_obs[0]
        not_done = 	jnp.isfinite(next_obs).all(axis=-1) \
        			* (x >= 0.2) \
        			* (x <= 1.0)

        done = ~not_done
        return done