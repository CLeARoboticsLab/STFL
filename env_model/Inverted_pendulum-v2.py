import jax.numpy as jnp

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        notdone = jnp.isfinite(next_obs).all(axis=-1) \
        		  * (jnp.abs(next_obs[1]) <= .2)
        done = ~notdone

        return done