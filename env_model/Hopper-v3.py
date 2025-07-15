import jax.numpy as jnp

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        height = next_obs[0]
        angle = next_obs[1]
        not_done = (
            jnp.abs(next_obs[1:] < 100).all(axis=-1)  
            * (height > 0.7)                
            * (height < jnp.inf)             
            * (jnp.abs(angle) < 0.2)         
        )

        done = ~not_done
        return done