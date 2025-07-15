class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        z = next_obs[0]
        done = (z < 1.0) + (z > 2.0)
        return done