class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):

        height = next_obs[0]
        angle = next_obs[1]
        not_done =  (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        return done