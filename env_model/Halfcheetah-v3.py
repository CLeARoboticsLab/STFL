import jax.numpy as jnp

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        done = False
        return done